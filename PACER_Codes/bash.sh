for DATA in oxford_pets oxford_flowers fgvc_aircraft dtd eurosat stanford_cars food101 sun397 caltech-101 ucf101;
do
    echo "
    Processing $DATA ...
    "
    python main.py -a ViT-B/16 --dataset $DATA
done