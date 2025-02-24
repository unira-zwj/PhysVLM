# RoboThink: Embodied Large Multi-modal Model with Self-Constraints

## 1. Release

## 2. Install
Only support Linux now.
1. Download this repository and `cd robothink`.
2. Install Package
```python
conda create -n robothink python=3.10 -y
conda activate robothink
pip install -e .
```
3. Install additional packages for training cases
```python
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## 3. Train
### 3.1 Stage1
1. data prepare
the pretrain stage annotation file follow the `pain` format like:
```python
[
    {
        "id": "004539375",
        "image": "llava_pretrain/images/00453/004539375.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "Render a clear and concise summary of the photo.\n<image>"
            },
            {
                "from": "gpt",
                "value": "select luxury furniture 3 - inch gel memory foam mattress topper"
            }
        ]
    },
    ...
]
```
2. modify train_stage1.sh
modify the pretrain bash file: `./scripts/train/train_stage1.sh`:
```python
STAGE1_TRAIN_ANNOTATION_PATH=annotation file path of your pretrain data
STAGE1_TRAIN_DATA_IMAGE_FOLDER=image dir path of your pretrain data
```
3. run pretrain.sh on terminal
```python
bash ./scripts/train/train_stage1.sh
```

### 3.2 stage2
1. data prepare
the train stage2 annotation file follow the `shareGPT` format like:
```python
[
    {
        "id": 2327515,
        "image": "gqa/images/2327515.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nAre there scarves or hats that are yellow?\nAnswer the question using a single word or phrase."
            },
            {
                "from": "gpt",
                "value": "Yes"
            },
            {
                "from": "human",
                "value": "Does the man wear shorts?"
            },
            {
                "from": "gpt",
                "value": "No"
            },
            ...
        ]
    },
    ...           
]
```
2. modify train_stage2.sh
modify the train_stage2 bash file: `./scripts/train/train_stage2.sh`:
```python
STAGE2_TRAIN_ANNOTATION_PATH=annotation file path of your train stage2 data
STAGE2_TRAIN_DATA_IMAGE_FOLDER=image dir path of your train stage2 data
```
3. run train_stage2.sh on terminal
```python
bash ./scripts/train/train_stage2.sh
```

## 4. Finetune
### 4.1 Full parameters
follow the train stage2 annotation file format and modify the finetune_full.sh, then run the finetune_full.sh on terminal
```python
bash ./scripts/train/finetune_full.sh
```

### 4.2 LORA finetune
follow the train stage2 annotation file format and modify the finetune_lora.sh, then run the finetune_lora.sh on terminal
```python
bash ./scripts/train/finetune_lora.sh
```

### 4.3 Qlora finetune
follow the train stage2 annotation file format and modify the finetune_qlora.sh, then run the finetune_qlora.sh on terminal
```python
bash ./scripts/train/finetune_qlora.sh
```

## 5. Inference
modify the parser in the inference.py
```python
parser.add_argument("--model-path", type=str, default="your model checkpoint path")
parser.add_argument("--image-path", type=str, default="the image file path")
```
and then run on terminal with: 
```python
python inference.py
```

or 
directly run on terminal with: 
```python
python inference.py -model-path your/model/checkpoint/path --image-path the/image/file/path
```