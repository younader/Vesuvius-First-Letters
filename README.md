# Vesuvius First Letters

This repository contains the second place solution to the Vesuvius First Letters challenge.

The approach uses an I3D architecture to detect ink from within the scrolls


## How to use
### Installation
I provide an image of my environment in the requirements.txt, I believe only the following packages need to be installed
```
pytorch-lightning   
typed-argument-parser   
segmentation_models_pytorch   
albumentations   
warmup_scheduler   
 ```


### Training
Download the data and appropriate segments (instructions [here](https://scrollprize.org/data)).   
Unzip the labels folder and place each {segmentid}_inklabel.png in its appropriate segment folder.

Adjust the CFG class with your compute in 64x64_256stride_i3d.py, these are the typical configs used in the kaggle competition. 
```
python 64x64_256stride_i3d.py
```

### Inference
the inference script runs a trained model , for more info about the arguments check the InferenceArgumentParser class in the inference script. a pretrained checkpoint is available [here](https://drive.google.com/file/d/1fAGZbVPHW6q1hNiI2E2NKzf6TyELzOC4/view?usp=sharing) 
```
e.g: python inference.py --segment_id 123 --model_path 'model.ckpt'
```
