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

- **Counts and imbalance.** Train has 1,341 normal and 3,875 pneumonia images, test 234 and 390, and the official val only 8 and 8. That is about 74% pneumonia, so a model that always says "pneumonia" already scores 74% accuracy. Results are reported as recall, precision and ROC-AUC, not accuracy alone.
- **Image size and shape differ by class, a possible shortcut.** Across all train images, normal scans have a median size of 1640×1328 and a mean aspect ratio of 1.23; pneumonia scans 1168×776 and 1.51. Across the whole dataset, 283 pneumonia images are RGB while every normal image is grayscale. Squashing everything straight to 224×224 would distort the two classes differently, so every image will be converted to one mode and center-cropped to a square before resizing. The difference in sharpness from the original resolution can't be removed and is listed under Limitations.
- **R markers and burned-in text.** Many images carry an "R" side marker in different sizes and positions, and some pneumonia images in the sample grid show printed scanner settings and a study ID. A model could learn these instead of the lungs, which is why Grad-CAM is used to check where it looks. These were seen in samples, not counted across the dataset.
- **The split is by patient, not by image.** The 3,883 train and val pneumonia images come from 1,642 children (one has 30 images). The official val set has only 16 images, so train and val were pooled and re-split 85/15 with whole patients on one side, giving 4,453 train and 779 val images with no shared patients (`splits/`). Pneumonia patients come from the `personNNN` file-name prefix. Normal names have no patient ID, so the `IM-NNNN` prefix is assumed to identify one study.

## Models

_To be written._

## Results

_To be written._

## Grad-CAM

_To be written._

## Limitations

_To be written._
