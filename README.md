# ASL Alphabet Recognition: HOG + SVM vs. ResNet18
CP467 Image Processing & Recognition · Group 18

We classify the 26 ASL letters from single hand images. The project compares a classical pipeline
(HOG features + linear SVM) with a fine-tuned ResNet18 CNN on accuracy, errors and speed, using the
public dataset and our own webcam photos taken under different lighting and backgrounds.
J and Z use the held end pose of each letter.

**Train on Google Colab → export weights → run the webcam app locally.**

## Team
| Member | Role |
|---|---|
| Ayla Topuz | Setup & Webcam App |
| Ryan Tran | Dataset Prep & HOG + SVM |
| Amrit Arora | ResNet18 (CNN) |
| Sushant Sah | Webcam Data Collection |
| Roneet Topiwala | Evaluation & Report |

## Structure
```
├── common.py              # class list, settings, preprocessing. Used by EVERYTHING
├── requirements.txt
├── notebooks/             # run on Colab
│   ├── 1_prepare_data.ipynb   # inspect dataset, make subset + splits
│   ├── 2_hog_svm.ipynb        # HOG + linear SVM
│   ├── 3_resnet18.ipynb       # ResNet18 fine-tuning
│   └── 4_compare.ipynb        # metrics, confusion matrices, timing
├── app/                   # run locally
│   ├── capture.py             # collect + label our own webcam images
│   └── webcam_app.py          # live demo
├── models/                # hog_svm.joblib, resnet18.pt (from Drive, not in git)
├── results/               # figures and tables for the reports
└── data/                  # images on Google Drive, not in git (see data/README.md)
```

## How to run

### What to run right now (Phase 1)
| Notebook | What it does | Time |
|---|---|---|
| `notebooks/1_prepare_data.ipynb` | Inspects the dataset: counts, sample images, J/Z vs I/D, near-duplicate check, brightness/blur, what each model's input looks like. Findings are at the bottom. | ~1 min |
| `notebooks/3_resnet18.ipynb` | GPU smoke test: loads pretrained ResNet18 with a 26-letter head, trains 10 steps (loss should fall from ~3.4 to ~0.1), saves and reloads the weights. | ~30 s CPU, seconds on GPU |

`2_hog_svm.ipynb` and `4_compare.ipynb` only have section headings for now (Phase 3).

### Option A: Google Colab (training)
One-time setup:
1. Get access to the shared Google Drive folder **`image-processing-group18`**. It contains `asl_raw.zip`.
2. In Google Drive, right-click that folder → **Organize → Add shortcut → My Drive**.
   (The notebooks look for it at `/content/drive/MyDrive/image-processing-group18`.)

Every time:
1. Go to [colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → GitHub** → enter `aylat7/image-processing-group18` → pick a notebook.
2. For `3_resnet18.ipynb`: **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all**, and allow Google Drive access when asked.

The first cell does everything else: mounts Drive, clones/pulls this repo, installs the pinned packages
and unzips the dataset to Colab's local disk.

> **Private repo?** Colab can't clone it without a token. Create a GitHub personal access token
> (Settings → Developer settings → Tokens, read access to this repo) and in the setup cell change the
> URL to `https://<TOKEN>@github.com/aylat7/image-processing-group18.git`. **Don't commit your token.**

> **Saving work from Colab:** changes made in Colab are lost when the session ends. Use
> **File → Save a copy in GitHub**, or download the notebook and commit it locally.

### Option B: Local (webcam app, quick checks)
One-time setup:
```bash
git clone git@github.com:aylat7/image-processing-group18.git
cd image-processing-group18
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
```
Then get the dataset into `data/raw/asl_raw/`. Either download `asl_raw.zip` from the shared Drive and run
`unzip asl_raw.zip -d data/raw`, or follow `data/README.md` to extract it from the original Mendeley download.
You should end up with `data/raw/asl_raw/A/A_1.jpg`, and so on.

Every time:
```bash
cd image-processing-group18
source .venv/bin/activate
jupyter notebook                   # opens in your browser → notebooks/ → open one → Run → Run All Cells
```
Or in **VS Code**: open the folder, open a notebook, **Select Kernel → `.venv`**, then **Run All**.

### Later phases
| Phase | Run | Where |
|---|---|---|
| 2 | `1_prepare_data.ipynb` (subset + splits) · `python app/capture.py` | Colab · local |
| 3 | `2_hog_svm.ipynb` · `3_resnet18.ipynb` → download models into `models/` | Colab |
| 4 | `python app/webcam_app.py` · `4_compare.ipynb` | local |

## Rules
1. All image preprocessing goes through `common.py`, in training **and** in the app.
2. Both models use the **same** split CSVs in `data/splits/`.
3. `data/webcam_test/` is **never** used for training or tuning.
4. Colab: save checkpoints to Drive every epoch; sessions time out.
5. Don't change settings in `common.py` without telling the group, because both models must be retrained after a change.
