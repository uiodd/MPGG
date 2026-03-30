# MPGG

This is the implementation of [Multi-Prototype Guided Gating Framework for Multimodal Emotion Recognition in Conversations]in PyTorch.

## Install
****
```
numpy==1.26.4
pandas==2.3.3
matplotlib==3.10.6
scikit-learn==1.1.1
scipy==1.15.3
torch==1.12.0+cu113
torchvision==0.13.0+cu113
torchaudio==0.12.0+rocm5.1.1
tensorboard==2.20.0
requests==2.32.5
```

This repository contains the source code of our paper, using the following datasets:

- [IEMOCAP](https://drive.google.com/file/d/1y4xLSKicRHZCd7dsG5y75TFecvHHyY3Q/view?usp=drive_link)
Download the preprocessed datasets and put them into `data/`.

- [MELD](https://drive.google.com/file/d/1H8NAwW_ogxsFOQvCuesxiVqBqa8X19J-/view?usp=drive_link)
Download the preprocessed datasets and put them into `data/`.

1 - Install Requirements

```
pip install -r ./requirements.txt
```

2 - Download the preprocessed datasets and put them into `data/`.

 

3 -  Usage:

# Train on IEMOCAP dataset
```
python train.py --dataset iemocap --optimizer adam --batch_size 32 --lr 1e-4 --weight_decay 1e-5 --dropout 0.5 --temperature 1.0 --lambda_proto 0.1 --gamma5 0.1 --epochs 100 --seed 42 --queue_size 4096 --use_memory_queue --use_momentum
```
# Train on MELD dataset
```
python train.py --dataset meld --optimizer adam --batch_size 32 --lr 3e-5 --weight_decay 1e-5 --dropout 0.2 --temperature 8 --lambda_proto 1 --gamma5 0.1 --epochs 10 --seed 42 --queue_size 4096 --use_memory_queue --use_momentum
```
