# chest-xray-pneumonia

Classify pediatric chest X-rays as normal or pneumonia. The question: how much does ImageNet pretraining help on a small medical dataset, and is the model looking at the lungs or at shortcuts?

**Not for clinical use.** A learning project on a public dataset.

## Data

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle. Only the top-level `chest_xray/` folders are used.

```bash
kaggle datasets download paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip "chest_xray/train/*" "chest_xray/val/*" "chest_xray/test/*" -d data/
```

Images are not in this repo.

## What's in the data

- **74% pneumonia.** Always guessing "pneumonia" scores 74% accuracy, so accuracy alone means little. I report recall, specificity, precision, F1 and ROC-AUC.
- **The classes differ in ways that aren't the disease.** Normal scans are bigger (median 1640×1328 vs 1168×776) and squarer (aspect ratio 1.23 vs 1.51). 283 pneumonia images are RGB; every normal one is grayscale. A model could learn these instead of the lungs. So every image is converted to RGB, resized so its short side is 224, then center-cropped to 224×224. Nothing gets stretched. Sharpness still differs, and that can't be removed.
- **Some images have burned-in text:** "R" markers, scanner settings, study IDs. Seen in samples, not counted. Grad-CAM will check where the model looks.
- **One child can have many X-rays,** up to 30. Splitting by image would put the same chest in train and val. So I pooled the official train and val (the official val has only 16 images) and split 85/15 by patient: 4,453 train, 779 val, zero shared patients. Saved in `splits/`. Pneumonia patients come from the `personNNN` prefix. Normal files have no patient ID, so I assume the `IM-NNNN` prefix is one study.

## Pipeline

- Transforms: resize short side to 224, center-crop 224, normalize with ImageNet mean/std. Training adds a random rotation of up to 10°. No horizontal flip: a flipped chest puts the heart on the wrong side.
- Loading takes 7 s per epoch with 4 workers, so it isn't the bottleneck.

## Results (val)

SmallCNN, trained from scratch: 5 blocks of conv, batch norm, ReLU and max-pool, then global average pooling and one linear layer. 980k parameters. SGD (momentum 0.9), batch 32, 15 epochs. lr 0.01, dropped to 0.001 after epoch 10. The weights kept are from the epoch with the lowest val loss.

| Metric | Value | In plain words |
|---|---|---|
| Recall | 0.982 | missed 10 of 564 sick children |
| Specificity | 0.949 | wrongly flagged 11 of 215 healthy children |
| Precision | 0.981 | |
| F1 | 0.981 | |
| ROC-AUC | 0.996 | |

The metrics are written from scratch and match sklearn to 4 decimals.

The lr drop mattered. Before it, val loss jumped between 0.086 and 0.27 from epoch to epoch. After it, it stayed between 0.078 and 0.083.

Two runs with the same seed don't give identical curves: some GPU convolution algorithms add numbers in a slightly different order each time. So small differences between models (around 0.002 AUC) are noise.

Take these with salt. Val is used to choose models, so it's optimistic, and this dataset's test set is known to be harder. The test set gets used once, at the end.

## To do

- ResNet-18: from scratch, frozen, fine-tuned
- Grad-CAM
- Test set, once
- Limitations
