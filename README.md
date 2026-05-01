# Robust Partial Face Recognition System

![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-3fb950?style=flat-square)

A deep learning-powered, closed-set facial recognition application engineered to accurately identify individuals even under challenging real-world occlusions (such as surgical masks, heavy blur, cropping, and artificial noise).

Powered by **PyTorch** and an **EfficientNet-B0** backbone via transfer learning, featuring an interactive **Streamlit** dashboard for testing, dataset management, and real-time evaluation.

> *Developed as part of an advanced university AI/Deep Learning project.*

---

## Table of Contents

- [Project Overview](#project-overview)
- [Performance Metrics](#performance-metrics)
- [Repository Structure](#repository-structure)
- [Dataset Details](#dataset-details)
- [Installation & Setup](#installation--setup)
- [Training Workflow](#training-workflow)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Model Architecture](#model-architecture)
- [Important Notes](#important-notes)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)
- [License](#license)

---

## Project Overview

Standard facial recognition systems often fail when features are obscured by PPE, sunglasses, or poor camera angles. This project tackles that exact problem. By utilizing **transfer learning** and a carefully structured multi-phase training pipeline, this model maintains high prediction accuracy across ten distinct types of visual obstruction.

---

## Performance Metrics

### Overall System Accuracy
Tested against a 100-class dataset containing over 21,260 images.

| Metric | Score |
|---|---|
| Top-1 Accuracy | **~88–89%** |
| Top-3 Accuracy | **~94–95%** |

### Accuracy Breakdown by Occlusion

| Condition | Accuracy |
|---|---|
| Original (Unmodified) | ~99% |
| Sunglasses | ~98% |
| Random Block | ~97% |
| Noise Patch | ~96% |
| Crop (Left / Right) | ~91% |
| Surgical Mask | ~87% |
| Top Crop | ~78% |
| Heavily Blurred | ~43% |

---

## Repository Structure

```text
partial-face-recognition/
├── App/
│   └── steamlit_app.py           # Interactive Streamlit web interface
├── Model/
│   ├── pretrained_model.py       # Phase 1: Feature extraction & head initialization
│   ├── Main.py                   # Phase 2: Classification head fine-tuning
│   ├── deep_train.py             # Full architecture training (staged unfreezing)
│   ├── night_train.py            # Automated hyperparameter search for overnight runs
│   └── predict.py                # Command-line inference utility
├── requirements.txt
└── README.md
```

---

### Supported Occlusion Types

| Type | Description |
|---|---|
| `original` | Clean, unobstructed face |
| `blurred` | Heavy Gaussian blur applied |
| `sunglasses` | Eyes covered by dark rectangle |
| `surgical_mask` | Lower face covered |
| `top_crop` | Upper half of face removed |
| `bottom_crop` | Lower half of face removed |
| `left_crop` | Left side removed |
| `right_crop` | Right side removed |
| `noise_patch` | Centre replaced with random noise |
| `random_block` | Solid black rectangle over face |

> The full dataset (~900 MB, 166 identities) is excluded due to size and CelebA licensing.
> The `metadata.csv` must contain columns: `identity`, `filename`, `transformation`, `split`.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download required files

Place the following in the `assets/` folder:

```
assets/res10_300x300_ssd_iter_140000.caffemodel
```

This is the OpenCV ResNet-SSD face detector (~11 MB, available from the OpenCV repository)

### 3. Prepare the Data

If using the full dataset, place it at the root of the project following this hierarchy:

```
partial_face_dataset/
└── <identity_name>/
    ├── <identity_name>_000_original.jpg
    ├── <identity_name>_000_blurred.jpg
    └── ...
```

---

## Training Workflow

For optimal results, execute the training scripts in the following order:

### Phase 1 — Feature Extraction & Head Training

Extracts and caches EfficientNet-B0 features for the entire dataset, then trains the custom classifier head. Run this once (~20–30 mins on CPU).

```bash
python Model/pretrained_model.py
```

### Phase 2 — Head Fine-Tuning

Optimizes the newly trained classifier head using a reduced learning rate. Fast and highly stable (~5–15 mins).

```bash
python Model/Main.py
```

### Full Backbone Training *(optional)*

Gradually unfreezes the EfficientNet backbone, applying differential learning rates to push accuracy above the 90% mark. Computationally intensive (~4–8 hrs on CPU).

```bash
python Model/deep_train.py
```

### Overnight Hyperparameter Search *(optional)*

Automates a training loop across 13 distinct hyperparameter configurations, designed to terminate at 08:00.

```bash
python Model/night_train.py
```

---

## Streamlit Dashboard

To interact with the model locally, launch the web application:

```bash
streamlit run App/steamlit_app.py
```

Open in your browser at `http://localhost:8501`

### Dashboard Features:

| Page | Description |
|---|---|
| **Recognize** | Upload a face image — get top-K predictions alongside confidence scores |
| **Dataset** | Browse all 166 identities and their occlusion variants |
| **Add Data** | Incorporate new faces and map them to specific identities within the database. |
| **Evaluations** | View detailed analytics, including F1-scores, confusion matrices, and occlusion-specific performance breakdowns. |
| **Train & History** | Trigger training routines directly from the UI and track historical model metrics. |

---

## Model Architecture

```
Input Image (224x224)
       |
EfficientNet-B0 Backbone  <-- ImageNet pretrained, frozen during head training
       |
  [1280-dim features]
       |
  Dropout (p=0.4)
  Linear (1280 -> 512)
  BatchNorm1D
  ReLU
  Dropout (p=0.3)
  Linear (512 -> 100)
       |
  Class Logits
       |
Temperature Scaling (T=0.5)  <-- sharpens confidence scores at inference
       |
    Softmax
```

- **Loss Fuction:** CrossEntropyLoss augmented with label smoothing (0.05–0.08) and weighted classes to handle imbalance.
- **Detection Pipeline:** OpenCV ResNet-SSD acts as the primary facial bounds detector, falling back to a Haar cascade if the primary fails.
- **Optimizer:** Adam optimizer paired with cosine annealing learning rate schedules.

---

## Important Notes

* **Closed-Set Limitation:** This system utilizes closed-set logic, meaning it can only correctly identify the specific 100 individuals it was trained on.
* **Unknown Identities:** Submitting a photo of someone outside the training set will yield a low-confidence prediction (the UI automatically flags results below 40% confidence).
* **Missing Files:** Pre-trained model weights (`.pt`) and the extensive full dataset are excluded from this repository. You must execute the training scripts locally to generate the required weights and feature caches.

---

## Future Enhancements

- Implement open-set recognition logic with explicit rejection thresholds for unknown faces.
- Enhance stabilization and accuracy on heavily blurred input data.
- Incorporate real-time webcam video processing for live inference.
- Deploy the application via cloud services (e.g., Streamlit Community Cloud, AWS, Azure).
- Broaden the dataset to improve demographic fairness and diverse representation.

---

## Contributors

**Aatma Ram**

---

## License

This project is licensed under the **MIT License** — see [LICENSE](./LICENSE) for details.

