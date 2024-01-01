# CAILA: Concept-Aware Intra-Layer Adapters for Compositional Zero-Shot Learning [WACV 2024]

> [**CAILA: Concept-Aware Intra-Layer Adapters for Compositional Zero-Shot Learning**](https://arxiv.org/abs/2305.16681)<br>
> [Zhaoheng Zheng](https://zhaohengz.github.io/), [Haidong Zhu](https://haidongz.github.io) and [Ram Nevatia](https://sites.usc.edu/iris-cvlab/professor-ram-nevatia/)

Official implementation of [CAILA: Concept-Aware Intra-Layer Adapters for Compositional Zero-Shot Learning](https://arxiv.org/abs/2305.16681).


## Installation
We build our model based on `Python 3.8` and `PyTorch 1.13`. To prepare the environment, please follow the instructions below.

- Create a conda environment and install the requirements:
	```
	conda create -n caila-release python=3.8.13 pip
	```
- Enter the environment:
	```
	conda activate caila-release
	```
- Install the requirements:
	```
	pip install -r requirements.txt
	```
## Datasets
For `MIT-States`, `C-GQA` and `UT-Zappos`, please run the following script to download the datasets to the directory you desire 1(`DATA_ROOT` in our example):
```
bash ./utils/download_data.sh DATA_ROOT
```
For `VAW-CZSL`, please follow the instruction in the [official repo](https://github.com/nirat1606/OADis).

The `DATA_ROOT` folder should be organized as following:
```
DATA_ROOT/
├── mit-states/
│   ├── images/
│   ├── compositional-split-natural/
├── cgqa/
│   ├── images/
│   ├── compositional-split-natural/
├── ut-zap50k/
│   ├── images/
│   ├── compositional-split-natural/
├── vaw-czsl/
│   ├── images/
│   ├── compositional-split-natural/
```
After preparing the data, set the `DATA_FOLDER` variable in `flags.py` to your data path.
## Model Zoo

| Dataset | AUC (Base/Large) | Download |
| :---: | :---: | :---: |
| MIT-States | 16.1 / 23.4 | [Base](https://drive.google.com/file/d/1RIflA4hrYdJQDvjPvZSNsy_4Mvbv69pX/view?usp=sharing) / [Large](https://drive.google.com/file/d/1k54Ld8P2dEA1iyC6mtU0AUDflE5rEvz4/view?usp=sharing) |
| C-GQA | 10.4 / 14.8 | [Base](https://drive.google.com/file/d/1aV2Pe0CWdx40qGfn7bX3FobPR3M9FZkA/view?usp=sharing) / [Large](https://drive.google.com/file/d/184rCbaymdALvnaU34HfeO5Yaa4ueOOhP/view?usp=sharing) |
| UT-Zappos | 39.0 / 44.1 | [Base](https://drive.google.com/file/d/16xfCBAoKJoBGGyf82ztpJd5riSK1DleJ/view?usp=sharing) / [Large](https://drive.google.com/file/d/1EHp_YVZ1Vl-6xaprfP2uM-1_LLJ5dRp4/view?usp=sharing) |
| VAW-CZSL* |  17.1 / 19.0 | [V](https://drive.google.com/file/d/17ZZvq5AAw6gSk9Dx7GibKwZc0TUSGcn0/view?usp=sharing) / [V+L](https://drive.google.com/file/d/17lrfjcBDVo2xI-gvL2fB6DE89kVzniVK/view?usp=sharing) |

*For VAW-CZSL, we provide two variations of Large model: one has adapters on the vision side (V) and the other has adapters on both the vision and language sides (V+L). The V+L model requires more GPU memory.
## Evaluation
To evaluate the model, put the downloaded checkpoint in a folder. We use `mit-base` as an example:
```
checkpoints/
├── mit-base/
│   ├── ckpt_best_auc.t7
```
Then, run the following command to evaluate the model:
```
python test.py --config configs/caila/mit.yml --logpath checkpoints/mit-base
```

## Training
First, please download CLIP checkpoints from HuggingFace: [VIT-B/32](https://huggingface.co/openai/clip-vit-base-patch32/blob/main/pytorch_model.bin) and [VIT-L/14](https://huggingface.co/openai/clip-vit-large-patch14/blob/main/pytorch_model.bin) and put them under `clip_ckpts` as following:
```
clip_ckpts/
├── clip-vit-base-patch32.pth
├── clip-vit-large-patch14.pth
```
Then, run the following command to train the model:
```
torchrun --nproc_per_node=$N_GPU train.py --config CONFIG_FILE
```
where `CONFIG_FILE` is the path to the config file. We provide the config files for all the experiments in the `configs` folder. For example, to train the base model on MIT-States, run:
```
torchrun --nproc_per_node=$N_GPU train.py --config configs/caila/mit.yml
```

Code will be released soon.

# Citing CAILA
If you find CAILA useful in your research, please consider citing:
```
@InProceedings{Zheng_2024_WACV,
    author    = {Zheng, Zhaoheng and Zhu, Haidong and Nevatia, Ram},
    title     = {CAILA: Concept-Aware Intra-Layer Adapters for Compositional Zero-Shot Learning},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {1721-1731}
}
```