# Data

Images are **not** in git. They live in the shared Google Drive folder.

```
data/
├── raw/asl_raw/      # Mendeley "Type_01 (Raw_Gesture)" photos: 28 folders × 1500 images, 300×300 JPG
├── splits/           # train.csv / val.csv / test.csv  ← tracked in git
├── webcam_train/     # our own images, may be used for training
└── webcam_test/      # held-out trials, NEVER used for training or tuning
```

## Getting the dataset
Source: https://data.mendeley.com/datasets/48dg9vhmyk/2 (`Root.zip`, 3.7 GB,
sha256 `9554bbf9b19c1429f4f9df620a9a82d4b5f362276c5d6ff26563808d46be1dd6`).

`Root.zip` contains 5 *versions of the same photos* (`Type_01` raw, `Type_02`–`Type_05`
hand-skeleton renderings). We only use **`Type_01_(Raw_Gesture)`**, because the webcam
app sees raw photos. Extract it to `data/raw/asl_raw/`:

```bash
unzip -q Root.zip 'Root/Type_01_(Raw_Gesture)/*' -d data/raw
mv "data/raw/Root/Type_01_(Raw_Gesture)" data/raw/asl_raw && rmdir data/raw/Root
```

On Colab, use `asl_raw.zip` from the shared Drive folder instead (see notebook setup cell).
