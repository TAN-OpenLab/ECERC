# **ECERC**
This repository contains the official implementation of our paper *ECERC: Evidence-Cause Attention Network for Multi-Modal Emotion Recognition in Conversation*, published at ACL 2025.
### 1. Requirements
The experiments were conducted on a Windows 10 operating system equipped with an NVIDIA A100 GPU (80GB). Further system specifications are provided in the accompanying *OS_info.txt* and *requirement.yml* files.
```
conda env create -f requirement.yml -n ecerc
```

### 2. Datasets
The benchmark datasets used in our paper are IEMOCAP and MELD. Due to copyright restrictions, we provide links to the [preprocessed versions](https://drive.google.com/drive/folders/1yXot2Kcdt7ELbxwAL1lB9yPWokcr7vhq?usp=drive_link) only. The original datasets can be downloaded from their respective official sources.
### 3. Evaluation
You can download our pretrained ECERC_MODEL(" "/"_2"/"_3") for each dataset from our **Huggingface Repository:** [zt-ai/ECERC](https://huggingface.co/zt-ai/ECERC).
After downloading, place the models in the corresponding ```IEMOCAP```/```MELD``` folder.
To reproduce results closely matching those reported in our paper (which presents the average over 3 runs), you can execute the following commands:
```
cd IEMOCAP/MELD
python inference.py
```
You can also run the ```train.py``` script to train the model from scratch. For accurate reproduction of our results, please ensure that your experimental environment matches ours exactly, as specified in [1. Requirements](#1-requirements).

## 4. Citation
```
@inproceedings{zhang-tan-2025-ecerc,
    title = "{ECERC}: Evidence-Cause Attention Network for Multi-Modal Emotion Recognition in Conversation",
    author = "Zhang, Tao  and
      Tan, Zhenhua",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.102/",
    pages = "2064--2077",
    ISBN = "979-8-89176-251-0"
}


```

### 5. License
This code repository is licensed under the MIT License. ECERC supports commercial use.
