# Study Three: VGG19 reproduction and exploratory improvements

## Paper and dataset

This work follows Study Three in Ahsan et al., *Defect analysis of 3D printed object using transfer learning approaches*, Expert Systems with Applications 253 (2024), 124293, DOI: 10.1016/j.eswa.2024.124293. Study Three uses the [3D-Printer Defected Dataset](https://www.kaggle.com/datasets/justin900429/3d-printer-defected-dataset), with 759 defective and 798 nondefective images. We downloaded the [public GitHub mirror](https://github.com/pingkung/3D-Printer-Defected-Dataset), which has the same class counts. `author/study3_mirror_tree.json` pins the mirror revision and each source file hash. We could not verify byte-for-byte identity with the Kaggle download.

The paper's Table 1 gives train/validation/test counts of 528/78/153 defective and 561/81/156 nondefective images. `results/study3/split.csv` records our exact image membership and SHA-256 hashes. We matched those counts with a class-wise random split, seed 42. The paper does not provide the exact image membership or random seed, so this is one reproducible approximation of its split.

## Methods

`study3_vgg19.py --mode paper` uses ImageNet VGG19 with a frozen convolutional base, 4×4 average pooling, flattening, 64 ReLU units, dropout 0.5, and two-class softmax. It uses Adam at 0.001, batch size 30, 30 epochs, RGB/255 inputs, and 15-degree random rotations on training images. The architecture, input scaling, batch size, epochs and learning rate follow the paper. The exact rotation setting follows the authors' public notebook; the paper does not fully specify augmentation. We select the checkpoint with lowest validation loss and evaluate the held-out test images.

`--mode vgg_preprocess` changes only input preprocessing to [Keras VGG19 preprocessing](https://keras.io/api/applications/vgg/vgg_preprocessing/): RGB is converted to BGR and ImageNet channel means are subtracted. All other settings and the split remain the same. This is our improvement experiment, not the paper's method.

`study3_finetune.py` starts from the best official-preprocessing checkpoint, unfreezes VGG19 block 5, and trains 10 further epochs with Adam at 0.00001. This is a separate exploratory experiment. The test set was inspected during earlier experiments, so neither improvement result is an unbiased final estimate. A fresh held-out dataset or print-run-level test is needed for that claim.

## Results

| Run | Test correct | Accuracy | Defective recall | Defective F1 |
| --- | ---: | ---: | ---: | ---: |
| Paper-style baseline, our split | 235/309 | 76.05% | 73.86% | 75.33% |
| VGG19 official preprocessing, same split | 267/309 | 86.41% | 86.27% | 86.27% |
| Official preprocessing + block-5 fine-tuning, same split | 308/309 | 99.68% | 99.35% | 99.67% |

The preprocessing change improved test accuracy by **10.36 percentage points** on this fixed split (32 additional correct predictions net). The paper's Study Three VGG19/Adam accuracy is approximately **95% ± 1%**, averaged over its runs. Our result is below that figure. The authors' image membership, exact run protocol, and Kaggle file hashes are unavailable here, so this gap cannot be attributed to one cause. These are single-run results, not estimates of variance.

The block-5 fine-tuning result adds 41 correct predictions relative to the frozen official-preprocessing model. The checkpoint with lowest validation loss was from fine-tuning epoch 9. No exact SHA-256 duplicate files appear across the split, but visually similar frames from one print run may still be shared across subsets. Since this test set had already been evaluated in the preceding experiments, the 99.68% score is exploratory and **must not be presented as a confirmed improvement over the paper's 95% mean or as evidence of generalization to new print runs**.

## Reproduce

Use Python 3.12 and the repository root's `requirements-lock.txt`. From this directory on Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ..\requirements-lock.txt
.\.venv\Scripts\python.exe download_study3.py
.\.venv\Scripts\python.exe study3_vgg19.py --mode paper
.\.venv\Scripts\python.exe study3_vgg19.py --mode vgg_preprocess
.\.venv\Scripts\python.exe study3_finetune.py
```

The pretrained ImageNet weights download automatically. The recorded runs used TensorFlow 2.16.2 on CPU. The scripts save configurations, per-epoch history, per-image test predictions, metrics and figures under `results/study3/`. Large image data and `.keras` checkpoints are excluded from Git. The split CSV and source tree allow the inputs to be reconstructed.

## For the report and slides

Use the result table above, the learning curves, and confusion matrices from `results/study3/`. Describe the dataset source, exact split counts, preprocessing change, and test-set reuse limitation. Keep the paper's mean ± standard deviation separate from our single-run numbers. The earlier 271-image experiment in the repository is a notebook-based attempt and is not this Study Three reproduction.
