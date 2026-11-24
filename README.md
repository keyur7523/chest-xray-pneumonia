# Pneumonia detection from pediatric chest X-rays

A from-scratch CNN vs a frozen and a fine-tuned ImageNet ResNet, with data-shortcut checks and Grad-CAM.

> **Not for clinical use.** This is a learning project on a public dataset.

## Question

How much does ImageNet pretraining help on a small medical-imaging dataset, and is the model looking at the lungs or at shortcuts in the data?

## Data

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle. Use the top-level `chest_xray/{train,val,test}` folders.

```bash
kaggle datasets download paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip "chest_xray/train/*" "chest_xray/val/*" "chest_xray/test/*" -d data/
```

The images are not included in this repo.

## Data findings

_To be written._

## Models

_To be written._

## Results

_To be written._

## Grad-CAM

_To be written._

## Limitations

_To be written._
