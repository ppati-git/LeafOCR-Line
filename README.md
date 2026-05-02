# LeafOCR-Line
Code for evaluation of LeafOCR Line Dataset
Dataset is available at https://doi.org/10.6084/m9.figshare.30158038

Please refer the published paper at:
Sivan, R., Pati, P.B. A benchmark dataset for text line segmentation in palm leaf documents. Sci Data 13, 424 (2026). https://doi.org/10.1038/s41597-026-06718-1





# LeafOCR — Text Line Segmentation

Comparison of 7 semantic segmentation architectures for binary text-line segmentation on 512×512 document image patches. Built with TensorFlow/Keras.

---

## Models

| File | Architecture | Decoder style |
|---|---|---|
| `leafocrline-unet.py` | U-Net | Transposed conv + skip concat |
| `leafocrline-segnet.py` | SegNet | UpSampling2D, no skips |
| `leafocrline-fcn8.py` | FCN-8s | pool3 + pool4 skip fusion |
| `leafocrline-fcn16.py` | FCN-16s | pool4 skip fusion only *(model def only)* |
| `leafocrline-linknet.py` | LinkNet | ResNet encoder, additive skips |
| `leafocrline-PSPNET.py` | PSPNet | Pyramid Pooling Module |
| `leafocrline-deeplabv3.py` | DeepLabV3+ | ASPP + low-level feature refinement |

Each file is fully self-contained: model definition, data generator, training loop, and inference function.

---

## Requirements

```bash
pip install tensorflow numpy opencv-python
```

---

## Data Preparation

Images were resized so that height and width matched the closest multiples of 512 (preserving dimensions as close to the original as possible), then divided into non-overlapping 512×512 patches for training. After prediction, patches are stitched back together to reconstruct the full image. Color masks were converted to binary format before training for computational efficiency.

## Dataset Structure

```
dataset/
├── image_patches/
│   ├── train_patches_512/
│   ├── val_patches_512/
│   └── test_patches_512/
└── mask_patches/
    ├── train_patches_512/
    ├── val_patches_512/
    └── test_patches_512/
```

- Images: RGB, `.jpg` / `.jpeg` / `.png`, auto-resized to 512×512
- Masks: grayscale binary (pixel > 127 → foreground, ≤ 127 → background)
- Mask filename must match image filename (extension can differ)

---

## Training

**Run directly** — edit the paths at the bottom of any script and run:

```bash
python leafocrline-unet.py
```

**Or import:**

```python
from leafocrline_unet import train_unet

model, history = train_unet(
    train_image_dir="path/to/train/images",
    train_mask_dir ="path/to/train/masks",
    val_image_dir  ="path/to/val/images",
    val_mask_dir   ="path/to/val/masks",
    epochs=50,
    batch_size=4,
    use_focal_loss=True
)
model.save("final_unet_model.keras")
```

Same signature across all models — swap import and function name:

| Model | Function |
|---|---|
| U-Net | `train_unet(...)` |
| SegNet | `train_segnet(...)` |
| FCN-8s | `train_fcn8(...)` |
| LinkNet | `train_linknet(...)` |
| PSPNet | `train_pspnet(...)` |
| DeepLabV3+ | `train_deeplabv3(...)` |

### Training config (all models)

| Parameter | Value |
|---|---|
| Optimizer | Adam, lr=1e-4 |
| Loss (default) | Focal + Dice |
| Loss (alt) | BCE + Dice (`use_focal_loss=False`) |
| LR schedule | ReduceLROnPlateau — factor=0.2, patience=5 |
| Early stopping | patience=10, monitors `val_dice_coefficient` |
| Checkpoint | saves on `val_dice_coefficient` improvement |

### Outputs

| File | Description |
|---|---|
| `best_<model>_model.keras` | Best checkpoint by val Dice |
| `final_<model>_model.keras` | End-of-training weights |
| `<model>_training_log.csv` | Per-epoch metrics |

---

## Inference

```python
from leafocrline_unet import predict_image_unet

mask = predict_image_unet(
    model_path="best_unet_model.keras",
    image_path="page.jpg",
    target_size=(512, 512)
)
# returns uint8 numpy array (0/255), resized to original image dimensions
```

Inference functions per model:

| Model | Function |
|---|---|
| U-Net | `predict_image_unet` |
| SegNet | `predict_image_segnet` |
| FCN-8s | `predict_image_fcn8` |
| PSPNet | `predict_image_pspnet` |
| LinkNet | `predict_image` |
| DeepLabV3+ | `predict_image_deeplabv3` |

---

## Metrics

All models track: `dice_coefficient`, `iou_score`, `precision`, `recall`, `f1_score`

Loss functions available in all scripts:
- `combined_focal_dice_loss` — default, handles foreground/background imbalance
- `binary_crossentropy_dice_loss` — alternative
- `focal_loss`, `dice_loss` — available individually

---

