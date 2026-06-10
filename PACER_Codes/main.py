# zsCLIP prompt
# Copyright (c) Alibaba Group
import argparse
import torch
import torchvision.datasets as datasets
import torch.nn.functional as F
import clip
import os
from datasets import build_dataset
import json
from datasets.utils import DatasetWrapper
from utils import *

from dassl.data.transforms import build_transform
from dassl.utils import set_random_seed

from datasets.imagenet import ImageNet


model_names = ['RN50', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
parser = argparse.ArgumentParser(description='InMaP for ImageNet')
parser.add_argument('--data_path', default='Your/Dataset/Path', type=str,
                    help='dataset path')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: RN50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--iters_proxy', default=30, type=int, metavar='N',
                    help='number of total iterations for learning vision proxy')
parser.add_argument('--iters_sinkhorn', default=20, type=int, metavar='N',
                    help='number of total iterations for optimizing Sinkhorn distance')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=10, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--tau_t', default=0.01, type=float)
parser.add_argument('--tau_i', default=0.04, type=float)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--dataset', default='caltech-101', type=str)
parser.add_argument('--shots', default=16, type=int)
parser.add_argument('--cache_k', default=0.3, type=float)
parser.add_argument('--beta', default=1.0, type=float)
parser.add_argument('--logits_ratio', default=0.3, type=float)


from types import SimpleNamespace

transform_cfg = SimpleNamespace()
transform_cfg.INPUT = SimpleNamespace()
transform_cfg.INPUT.SIZE = [224, 224]
transform_cfg.INPUT.INTERPOLATION = "bicubic"
transform_cfg.INPUT.PIXEL_MEAN = [0.48145466, 0.4578275, 0.40821073]
transform_cfg.INPUT.PIXEL_STD = [0.26862954, 0.26130258, 0.27577711]
transform_cfg.INPUT.TRANSFORMS = ["random_resized_crop", "random_flip", "normalize"]
transform_cfg.INPUT.NO_TRANSFORM = False
transform_cfg.INPUT.RRCROP_SCALE = (0.08, 1.0)
transform_cfg.INPUT.CROP_PADDING = 4
transform_cfg.INPUT.COLOR_JITTER = True
transform_cfg.INPUT.AUTO_AUGMENT = False
transform_cfg.INPUT.RE_PROB = 0.25
transform_cfg.INPUT.RE_MODE = "pixel"
transform_cfg.INPUT.RE_COUNT = 1

DATASET_NAME = {
    "oxford_pets": "oxfordpets",
    "oxford_flowers": "flowers102",
    "fgvc_aircraft": "fgvcaircraft",
    "dtd": "dtd",
    "eurosat": "eurosat",
    "stanford_cars": "stanfordcars",
    "food101": "food101",
    "sun397": "sun397",
    "caltech-101": "caltech101",
    "ucf101": "ucf101",
}


def main():
    set_random_seed(1)
    args = parser.parse_args()
    print(args)

    weak_transform = build_transform(transform_cfg, is_train=False)

    print('load pre-trained model')
    model, preprocess = clip.load(args.arch)
    model = model.cuda()
    model.eval()

    print('load data')
    cfg = {'subsample_classes': 'all'}
    dataset = build_dataset(cfg, args.dataset, args.data_path, args.shots)
    class_names = dataset.classnames

    # build text classifier
    prompt_file1 = "CuPL_prompts_" + DATASET_NAME[args.dataset] + ".json"
    prompt_file1 = os.path.join('./gpt3_prompts', prompt_file1)
    with open(prompt_file1) as f:
        cupl_prompts = json.load(f)
    cupl_prompt_features = gpt_llm_fea(class_names, cupl_prompts, model, dataset.template)

    prompt_file2 = DATASET_NAME[args.dataset] + "_prompt.json"
    prompt_file2 = os.path.join('./gpt_file_cafo', prompt_file2)
    with open(prompt_file2) as f:
        cafo_prompts = json.load(f)
    cafo_prompt_features = gpt_llm_fea(class_names, cafo_prompts, model, dataset.template)

    merged_dict = {key: torch.cat([cupl_prompt_features[key], cafo_prompt_features[key]]) for key in
                   cupl_prompt_features}

    text_classifier = build_text_classifier_from_features(merged_dict)

    test_set = dataset.test
    weak_test_dataset = DatasetWrapper(test_set, input_size=224, transform=weak_transform, is_train=False)
    weak_loader = torch.utils.data.DataLoader(weak_test_dataset, batch_size=args.batch_size, num_workers=args.workers)
    weak_image_feat, weak_image_label = get_image_features(model, weak_loader)
    n = len(weak_image_label)
    class_num = len(dataset.classnames)

    text_classifier = text_classifier.float()  # [dim, C]
    weak_logits_t = weak_image_feat @ text_classifier  # [N, C]


    acc1, acc5 = accuracy(weak_logits_t, weak_image_label, topk=(1, 5))
    top1 = (acc1 / n) * 100
    print(f"accuracy with text proxy inited vision proxy: {top1:.2f}")

    OT_weak_logits_t = sinkhorn(weak_logits_t, args.tau_t, args.iters_sinkhorn)
    softmax_raw_logits = F.softmax(weak_logits_t / 0.01, dim=1)

    LT_rate = calculate_LT_rate(softmax_raw_logits, OT_weak_logits_t)
    filter_k = compute_filter_k(LT_rate, mode="sigmoid")

    # create cache model
    cache_num = int(len(weak_logits_t) * args.cache_k / class_num)  # 0.3
    # print("cache_num:{}".format(cache_num))
    cache_keys, cache_values = create_cache_model(weak_image_feat, weak_logits_t, class_num, cache_num)
    cache_keys = cache_keys / cache_keys.norm(dim=-1, keepdim=True) # feature
    cache_values = cache_values.long()  # label

    # update classifier
    image_classifier = image_opt(args, filter_k, class_num, cache_keys, cache_values, weak_image_feat, text_classifier, args.lr, args.iters_proxy, args.tau_i)
    logits_i = weak_image_feat @ image_classifier
    acc1, acc5 = accuracy(logits_i, weak_image_label, topk=(1, 5))
    top1 = (acc1 / n) * 100
    print(f"accuracy with prototype proxies optimized vision proxy: {top1:.2f}")

    checkpoint_path = os.path.join('checkpoint', f'{args.dataset}_image_classifier.pth')
    torch.save(image_classifier, checkpoint_path)



def image_opt(args, filter_k, class_num, cache_keys, cache_values, feat, init_classifier, lr=10, iter=2000, tau_i=0.04):
    ins, dim = feat.shape
    raw_logits = feat @ init_classifier
    OT_raw_logits = sinkhorn(raw_logits, args.tau_t, args.iters_sinkhorn)
    softmax_raw_logits = F.softmax(raw_logits * 100., dim=1)

    filter_num = int(len(raw_logits) * filter_k / class_num)

    high_confidence_unique_indices, low_confidence_indices = confidence_filter(softmax_raw_logits, class_num, filter_num)

    # Uniform Subset
    P1 = sinkhorn(raw_logits[high_confidence_unique_indices], args.tau_t, args.iters_sinkhorn)
    # one-hot label
    P1_onehot = P1.clone()
    val, idx = torch.max(P1_onehot, dim=1)
    OT_mask = val > args.alpha
    P1_onehot[OT_mask, :] = 0
    P1_onehot[OT_mask, idx[OT_mask]] = 1
    more_alpha = torch.where(OT_mask)[0]
    less_alpha = torch.where(~OT_mask)[0]

    tuning_ratio = len(less_alpha) / len(high_confidence_unique_indices)

    # build extra logits
    R_fF = feat[low_confidence_indices] @ cache_keys.t()  # [low_confidence, m]
    sim = get_sim_based_kl(cache_keys, feat[low_confidence_indices], init_classifier)  # [low_confidence, m]
    R_fF = R_fF * sim  # [low_confidence, m]
    cache_logits = ((-1) * (args.beta - args.beta * R_fF)).exp() @ cache_values.float()
    cache_logits = F.softmax(cache_logits, dim=1)

    # Imbalance Subset
    enhanced_low_confidence_logits = cache_logits * args.logits_ratio + OT_raw_logits[low_confidence_indices] * (1 - args.logits_ratio)

    classifier = init_classifier.clone()
    pre_norm = float('inf')
    for i in range(0, iter):
        prob = F.softmax(feat @ classifier / tau_i, dim=1)
        high_confidence_plabel = prob.clone()
        high_confidence_plabel[high_confidence_unique_indices] = P1_onehot
        high_grad = feat.T @ (prob - high_confidence_plabel)

        low_confidence_plabel = prob.clone()
        low_confidence_plabel[low_confidence_indices] = enhanced_low_confidence_logits
        low_grad = feat.T @ (prob - low_confidence_plabel)

        grad = high_grad + low_grad * tuning_ratio

        temp = torch.norm(grad)
        if temp > pre_norm:
            lr /= 2.
        pre_norm = temp
        classifier -= (lr / (ins * tau_i)) * grad # lr*grad/(ins*tau_i)
        classifier = F.normalize(classifier, dim=0)
    return classifier

if __name__ == '__main__':
    main()
