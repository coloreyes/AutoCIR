# 

## Environmental Settings

Our experiments are conducted on Ubuntu 22.04, two NVIDIA 3090Ti GPUs, 128GB RAM, and Intel i7-13700KF. AutoCIR is implemented by `Python 3.9`, `Cuda 12.1`.

Create a virtual environment and install GPU-support packages via [Anaconda](https://www.anaconda.com/):
```shell
# create virtual environment
conda create --name AutoCIR python=3.9 cudatoolkit=12.1

# activate virtual environment
conda activate AutoCIR

# install other dependencies
pip install -r requirements.txt
```

## Usage

Here we take the sample of ICIP dataset as an example to demonstrate the usage.

### File Directory Structure

```
AutoCIR/
├── datasets/
│   ├── FASHIONIQ/
│   ├── circo/
│   └── CIRR/
├── model/
├── config/
└── src/
```
### Required Datasets

#### FashionIQ

Download the FashionIQ dataset following the instructions in
the [**official repository**](https://github.com/XiaoxiaoGuo/fashion-iq). 
After downloading the dataset, ensure that the folder structure matches the following:

```
FASHIONIQ/
├── captions/
│   ├── cap.dress.[train | val | test].json
│   ├── cap.toptee.[train | val | test].json
│   └── cap.shirt.[train | val | test].json
├── image_splits/
│   ├── split.dress.[train | val | test].json
│   ├── split.toptee.[train | val | test].json
│   └── split.shirt.[train | val | test].json
└── images/
    ├── B00006M009.jpg
    ├── B00006M00B.jpg
    ├── B00006M6IH.jpg
    └── ...
```


#### CIRR

Download the CIRR dataset following the instructions in the [**official repository**](https://github.com/Cuberick-Orion/CIRR).
After downloading the dataset, ensure that the folder structure matches the following:

```
CIRR/
├── train/
│   ├── 0/
│   │   ├── train-10108-0-img0.png
│   │   ├── train-10108-0-img1.png
│   │   └── ...
│   ├── 1/
│   ├── 2/
│   └── ...
├── dev/
│   ├── dev-0-0-img0.png
│   ├── dev-0-0-img1.png
│   └── ...
├── test1/
│   ├── test1-0-0-img0.png
│   ├── test1-0-0-img1.png
│   └── ...
└── cirr/
    ├── captions/
    │   ├── cap.rc2.train.json
    │   ├── cap.rc2.val.json
    │   └── cap.rc2.test1.json
    └── image_splits/
        ├── split.rc2.train.json
        ├── split.rc2.val.json
        └── split.rc2.test1.json
```

#### CIRCO

Download the CIRCO dataset following the instructions in the [**official repository**](https://github.com/miccunifi/CIRCO).
After downloading the dataset, ensure that the folder structure matches the following:

```
CIRCO/
├── annotations/
│   ├── val.json
│   └── test.json
└── COCO2017_unlabeled/
    ├── annotations/
    │   └── image_info_unlabeled2017.json
    └── unlabeled2017/
        ├── 000000243611.jpg
        ├── 000000535009.jpg
        └── ...
```

### Preprocess

**Step 1: Generate image captions**  
Use `src/generate_captions.py` to generate captions for the dataset:

```bash
python src/generate_captions.py --dataset_name FASHIONIQ --captioner blip2
```

**Step 2: disassemble the dataset:**
Using `src/main.py` with preprocessed captions for image-text retrieval

```bash
python src/main.py  --dataset fashioniq_dress --split val --dataset-path $datapath --preload captions mods --llm_prompt prompts.structural_modifier_prompt_fashion --clip ViT-B-32 --max_correct_num 1
```