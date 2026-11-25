# VGG19 reproduction for IDAT7215 Project 4

This folder contains one **VGG19 + Adam** reproduction attempt using the authors' public AMdataset images, followed by a separate fine-tuning comparison. It is intended for integration into the group's eight-model project.

## Result and scope

The final model classified all 55 images in the 20% holdout correctly: accuracy 1.000, defective-class precision 1.000, recall 1.000, and F1 1.000. The holdout was also evaluated after every training epoch, so this is a **validation result, not an independent test result**. The model has not been evaluated on new print runs. The authors' Study One/Adam result table also reports 1.000 validation and test accuracy for VGG19, but the exact correspondence between these public images and the paper's Study One dataset and split has not been verified. Do not describe this run as a strict reproduction of the paper's test score.

## Method

Images are resized to 224 × 224, converted to RGB, and divided by 255. ImageNet-pretrained VGG19 is used without its original classifier. All VGG19 convolutional layers are frozen. The replacement head uses 4 × 4 average pooling, flattening, a 64-unit ReLU layer, dropout 0.5, and two-class softmax. Training uses Adam with learning rate 0.001, binary cross-entropy, batch size 5, 50 epochs, and random rotations up to 15 degrees. The images are split 80/20 with random state 42. Details are recorded in `results/config.json`.

The implementation adapts the authors' public `fault_3d_printing_experiment.ipynb` by activating VGG19, sorting source paths, setting a global seed, saving the split and per-image predictions, and processing all batches each epoch to avoid iterator exhaustion under Keras 3. The script, output capture, charts, and checks in this folder were developed for this group reproduction. Source attribution and links follow below.

## Reproduce

Use Python 3.12 on Windows or a compatible TensorFlow environment. The recorded run used TensorFlow 2.16.2 on CPU. From this folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe run.py --download
.\.venv\Scripts\python.exe run.py --epochs 50
```

The download verifies each image against the authors' Git tree hash. The public repository contained 271 image files at the recorded revision, with 136 defective and 135 non-defective images. The author's README describes 136 images in each class, so this one-image discrepancy is recorded here. The code automatically downloads the official ImageNet VGG19 weights on first training run. Outputs are written to `results/baseline_50epochs`.

`split.csv` records the exact images and hashes used for training and holdout. Other group members can use this file to align their model comparisons. `history.csv` supports the learning curves. `predictions_final.csv` and `metrics_final.json` support the confusion matrix and result table. The full trained model is retained locally as `final.keras` and can be shared separately if needed.

## Files for the group report and presentation

- `results/accuracy.png`, `loss.png`, and `confusion_matrix_final.png` are ready to use in the results section and slides.
- `results/config.json`, `environment.json`, `architecture.txt`, and `split.csv` document the experimental setup.
- `results/history.csv`, `predictions_final.csv`, and `metrics_final.json` provide numbers behind the figures.
- The report should discuss the 55-image validation result, the shared validation/holdout limitation, the dataset count discrepancy, and the unverified Study One correspondence.
- The individual contribution section can describe VGG19 implementation, experiment execution, verification, and analysis, with attribution for adapted author code.

## Separate improvement comparison

`compare.py` performs a new stratified 60/20/20 train/validation/test split and saves its exact membership in `comparison/split.csv`. It trains the frozen VGG19 baseline for 50 epochs, then unfreezes only block 5 and fine-tunes for 20 epochs at learning rate 0.00001. Both models use the same split, and their best checkpoints are chosen by validation loss before scoring the test images.

Both models classified **54/55 test images correctly**, giving accuracy 98.18% and defective-class recall 96.43%. Both missed the same defective image, `IMG_9080.JPG`. This trial **did not improve test accuracy, recall, or F1**. It should be reported as an attempted modification without measured performance gain. The random image split may put visually similar frames from a single print run in different subsets, so these results do not establish performance on new print runs.

Run this separate comparison after downloading the data:

```powershell
.\.venv\Scripts\python.exe compare.py
```

The experiment outputs are in `comparison/`. The trained checkpoints remain local because of their size.

## VGG19 preprocessing experiment

`preprocessing_experiment.py` keeps the comparison split, frozen VGG19 backbone, classifier head, optimizer, and 50-epoch schedule. Its single change is to apply Keras VGG19's documented ImageNet preprocessing after training augmentation and to validation/test images. The earlier reproduction divided RGB values by 255; this experiment converts RGB to BGR and subtracts ImageNet channel means as specified by Keras.

The best-validation-loss checkpoint classified **55/55 test images correctly** (100% accuracy, defective-class recall, and defective-class F1), compared with **54/55** for the frozen baseline on the same split. It correctly classified the previously missed defective image `IMG_9080.JPG`. Outputs and the exact configuration are in `comparison/official_preprocessing/`.

This is an **exploratory improvement**. The 55-image test set had already been examined in the earlier baseline and fine-tuning work, and the image-level split may include related frames across subsets. The result should not be presented as an unbiased estimate for new print runs. A fresh print-run-level test would be needed to assess generalization.

Run after downloading the dataset and completing `compare.py` so its saved split exists:

```powershell
.\.venv\Scripts\python.exe preprocessing_experiment.py
```

## Sources

1. M. M. Ahsan, S. Raman, Y. Liu, and Z. Siddique, "Defect analysis of 3D printed object using transfer learning approaches," *Expert Systems with Applications*, vol. 253, art. 124293, 2024, doi: 10.1016/j.eswa.2024.124293.
2. Authors' experiment code: https://github.com/mahsan2/AM3d/blob/main/fault_3d_printing_experiment.ipynb
3. Authors' public dataset: https://github.com/mahsan2/AMdataset
4. Authors' Study One VGG19/Adam result: https://github.com/mahsan2/AM3d/blob/main/Studyone/Adam/vgg19model_results.xlsx
