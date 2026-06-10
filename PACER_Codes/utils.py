import clip
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.utils import Datum  # adapted: use InMaP's Datum
from tqdm import tqdm
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip
from collections import defaultdict
from datasets.imagenet import imagenet_classes
import math

def gpt_llm_fea(classnames, gpt_prompts, clip_model, template):
    with torch.no_grad():
        prompt_list = [len(gpt_prompts[cls.replace('_', ' ')]) for cls in classnames]
        if max(prompt_list) == min(prompt_list):
            print('text descriptions are not equal, and minimal is :{}'.format(min(prompt_list)))
        else:
            print('text descriptions for each class is : {}'.format(min(prompt_list)))
        num_des = min(prompt_list)
        prompt_fea = defaultdict()

        for classname in tqdm(classnames):
            classname = classname.replace('_', ' ')
            text = list()

            prompt_text = gpt_prompts[classname][:num_des]
            for item in prompt_text:
                text.extend([item])
            text.extend([t.format(classname) for t in template])

            texts = clip.tokenize(text, truncate=True).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            prompt_fea[classname] = class_embeddings
        return prompt_fea

def build_text_classifier_from_features(prompt_features):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in prompt_features.keys():
            class_embeddings = prompt_features[classname]
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

def get_image_features(model, loader):
    with torch.no_grad():
        image_feat = []
        image_label = []
        for i, (images, target) in enumerate(loader):
            images = images.cuda()
            target = target.cuda()
            image_features = model.encode_image(images)
            image_feat.append(F.normalize(image_features, dim=1))
            image_label.append(target)
    image_feat = torch.cat(image_feat, dim=0)
    image_label = torch.cat(image_label, dim=0)
    image_feat = image_feat.float()  # can keep fp16 for efficiency on GPU

    return image_feat, image_label

def calculate_LT_rate(softmax_raw_logits, OT_raw_logits):
    ins, _ = softmax_raw_logits.shape
    # softmax_raw_logits ===> h_ratio & l_ratio
    val, idx = torch.max(softmax_raw_logits, dim=1)
    h_mask = val > 0.5
    h_confidence = torch.where(h_mask)[0]
    h_ratio = (len(h_confidence)) / ins

    l_mask = val <= 0.5
    l_confidence = torch.where(l_mask)[0]
    l_ratio = (len(l_confidence)) / ins

    # OT_raw_logits ===> high_ratio & low_ratio
    val, idx = torch.max(OT_raw_logits, dim=1)
    high_mask = val > 0.5
    high_confidence = torch.where(high_mask)[0]

    low_mask = val <= 0.5
    low_confidence = torch.where(low_mask)[0]

    high_ratio = len(high_confidence) / ins
    low_ratio = len(low_confidence) / ins

    LT_rate = (low_ratio - l_ratio) / high_ratio

    return abs(LT_rate)

def get_sim_based_kl(new_cache_keys, new_test_feature, new_clip_weights):
	new_cache_logits = F.softmax(10. * new_cache_keys @ new_clip_weights, dim=1)
	new_test_logits = F.softmax(10. * new_test_feature @ new_clip_weights, dim=1)
	kl = torch.empty(len(new_test_logits), len(new_cache_logits), dtype=torch.float16).to(torch.device("cuda:0"))
	for i in range(len(new_test_logits)):
		kl[i] = (new_test_logits[i] * torch.log(new_test_logits[i] / new_cache_logits)).sum(dim=1)
	kl_mean = kl.mean(dim=1).unsqueeze(1)
	kl_std = kl.std(dim=1).unsqueeze(1)
	kl_norm = (kl - kl_mean) / kl_std
	kl_norm_max = kl_norm.max(dim=1).values.unsqueeze(1)
	kl_norm_min = kl_norm.min(dim=1).values.unsqueeze(1)
	sim = 2 * (1 - (kl_norm - kl_norm_min) / (kl_norm_max - kl_norm_min))
	return sim


def sinkhorn(M, tau_t=0.01, iter=20):
    row, col = M.shape
    P = F.softmax(M / tau_t, dim=1)
    P /= row
    for it in range(0, iter):
        # total weight per column must be 1/col or q_j
        P /= torch.sum(P, dim=0, keepdim=True)
        P /= col
        # total weight per row must be 1/row
        P /= torch.sum(P, dim=1, keepdim=True)
        P /= row
    P *= row  # keep each row sum to 1 as the pseudo label
    return P


def confidence_filter(logits, class_num, k):
    high_confidence_indices_list = []
    for i in range(class_num):
        # logits calculate KL divergence
        _, index_i = logits[:, i:i + 1].squeeze().sort(descending=True)

        indices = index_i[0:k]
        indices = indices.flatten()
        high_confidence_indices_list.append(indices)

    all_high_confidence_indices = torch.cat(high_confidence_indices_list, dim=0)

    high_confidence_unique_indices_list = []
    seen = set()
    for idx in all_high_confidence_indices:
        if idx.item() not in seen:
            high_confidence_unique_indices_list.append(idx)
            seen.add(idx.item())
    high_confidence_unique_indices = torch.stack(high_confidence_unique_indices_list)

    mask = torch.zeros(len(logits), dtype=torch.bool, device=logits.device)
    mask[high_confidence_unique_indices] = True

    low_confidence_indices = torch.where(~mask)[0]
    return high_confidence_unique_indices, low_confidence_indices


def confidence_prototype_filter(logits, class_num, k):
    high_confidence_indices_list = []
    label = []
    for i in range(class_num):
        # logits calculate KL divergence
        _, index_i = logits[:, i:i + 1].squeeze().sort(descending=True)
        logits_i = logits[index_i[0:k * 2]]

        kl_i_to_i = torch.zeros(len(logits_i))
        for j in range(len(logits_i)):
            kl_i_to_i[j] = torch.sum(logits_i[j] * torch.log(logits_i[j] / logits_i))
        _, index = kl_i_to_i.sort()
        indices = index_i[index[0:k]]
        indices = indices.flatten()
        high_confidence_indices_list.append(indices)

    all_high_confidence_indices = torch.cat(high_confidence_indices_list, dim=0)

    high_confidence_unique_indices_list = []
    seen = set()
    for idx in all_high_confidence_indices:
        if idx.item() not in seen:
            high_confidence_unique_indices_list.append(idx)
            seen.add(idx.item())
    high_confidence_unique_indices = torch.stack(high_confidence_unique_indices_list)

    mask = torch.zeros(len(logits), dtype=torch.bool, device=logits.device)
    mask[high_confidence_unique_indices] = True

    low_confidence_indices = torch.where(~mask)[0]

    return high_confidence_unique_indices, low_confidence_indices

def prototype_selection_class_freq(F_confi, L_confi, U_total=200, alpha=1.0, eps=1e-8):
    device = F_confi.device
    N, D = F_confi.shape
    if L_confi.dtype != torch.float:
        L_confi = L_confi.float()

    num_classes = L_confi.shape[1]
    # compute class indices and counts
    class_indices = [torch.where(L_confi[:, c] == 1)[0] for c in range(num_classes)]
    n_list = torch.tensor([len(idx) for idx in class_indices], dtype=torch.float, device=device)  # [C]
    total_samples = n_list.sum().item()
    if total_samples == 0:
        raise ValueError("No confident samples available in L_confi.")

    freq_weights = n_list / (n_list.sum() + eps)  # normalized frequency per class

    # compute provisional allocation
    alloc_float = alpha * freq_weights * U_total
    u_c_list = torch.round(alloc_float).long().to(device)

    # ensure at least 1 prototype per class that has any samples, and 0 for empty classes
    for c in range(num_classes):
        if n_list[c] == 0:
            u_c_list[c] = 0
        else:
            if u_c_list[c] < 1:
                u_c_list[c] = 1

    # adjust to keep sum within U_total
    sum_alloc = int(u_c_list.sum().item())
    # if over budget, decrement from largest allocated classes
    if sum_alloc > U_total:
        to_remove = sum_alloc - U_total
        candidates = [(int(u_c_list[c].item()), c) for c in range(num_classes) if u_c_list[c] > 1]
        candidates.sort(reverse=True)  # largest allocations first
        idx = 0
        while to_remove > 0 and idx < len(candidates):
            c = candidates[idx][1]
            dec = min(int(u_c_list[c].item() - 1), to_remove)  # keep at least 1
            u_c_list[c] -= dec
            to_remove -= dec
            idx += 1
        # if still to_remove > 0 (rare), remove 1 from any class >1 until satisfied
        if to_remove > 0:
            for c in range(num_classes):
                if u_c_list[c] > 1 and to_remove > 0:
                    u_c_list[c] -= 1
                    to_remove -= 1

    # if under budget, add to classes proportional to freq_weights
    sum_alloc = int(u_c_list.sum().item())
    if sum_alloc < U_total:
        to_add = U_total - sum_alloc
        freq_idx = torch.argsort(freq_weights, descending=True)
        idx = 0
        while to_add > 0:
            c = int(freq_idx[idx % num_classes].item())
            if n_list[c] > 0:
                u_c_list[c] += 1
                to_add -= 1
            idx += 1

    D_proxy = []
    F_proxy_all = []
    L_proxy_all = []
    for c in range(num_classes):
        indices = class_indices[c]
        if len(indices) == 0:
            continue
        feats = F_confi[indices]  # [n_c, D]
        norm = feats.norm(dim=1, keepdim=True).clamp(min=eps)
        norm_feats = feats / norm
        # cosine similarity centrality (sum of cosines to other instances in class)
        sim = torch.mm(norm_feats, norm_feats.t())  # [n_c, n_c]
        pi_local = sim.sum(dim=1)  # [n_c]
        u_c = int(u_c_list[c].item())
        u_c = min(u_c, len(indices))  # cannot exceed available samples
        if u_c == 0:
            continue
        _, top_local = torch.topk(pi_local, k=u_c)
        selected_global_idx = indices[top_local]
        F_proxy_c = F_confi[selected_global_idx]
        L_proxy_c = L_confi[selected_global_idx]
        D_proxy.append((F_proxy_c, L_proxy_c))
        F_proxy_all.append(F_proxy_c)
        L_proxy_all.append(L_proxy_c)

    if len(F_proxy_all) > 0:
        F_proxy_all = torch.cat(F_proxy_all, dim=0)
        L_proxy_all = torch.cat(L_proxy_all, dim=0)
    else:
        F_proxy_all = F_confi.new_empty((0, D))
        L_proxy_all = L_confi.new_empty((0, num_classes))

    return D_proxy, F_proxy_all, L_proxy_all, u_c_list

def create_cache_model(weak_image_feat, logits, class_num, cache_num):
    softmax_logits = F.softmax(logits * 100., dim=1)
    P1_onehot = softmax_logits.clone()
    val, idx = torch.max(P1_onehot, dim=1)
    OT_mask = val > 0.5
    selected_indices = torch.where(OT_mask)[0]
    plabel = torch.argmax(logits, dim=1)
    plabels = plabel[selected_indices]

    F_confi = weak_image_feat[selected_indices]
    L_confi = F.one_hot(plabels, num_classes=class_num).float()

    D_proxy, _, _, _ = prototype_selection_class_freq(F_confi, L_confi, U_total=cache_num, alpha=1.0)

    cache_keys = []
    cache_values = []
    for F_proxy_c, L_proxy_c in D_proxy:
        cache_keys.append(F_proxy_c)
        cache_values.append(L_proxy_c)
    cache_keys = torch.cat(cache_keys, dim=0)
    cache_values = torch.cat(cache_values, dim=0)

    return cache_keys, cache_values

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def compute_filter_k(LT_rate, threshold=0.1, min_k=0.25, max_k=1.0, mode="linear"):
    if mode == "linear":
        if LT_rate > 0.1:
            return min_k
        ratio = 1 - (LT_rate / threshold)
        filter_k = min_k + (max_k - min_k) * ratio
    elif mode == "sigmoid":
        steepness=30
        s = torch.sigmoid(steepness * (LT_rate - threshold))
        filter_k = max_k - (max_k - min_k) * s
    elif mode == "exp":
        exp = math.exp(-(LT_rate / threshold) * 5)
        filter_k = min_k + (max_k - min_k) * exp
    else:
        raise ValueError(f"Unsupported mode: {mode}, choose linear/sigmoid/exp")
    return filter_k