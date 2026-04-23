# WikiArt Painter Classification

Deep learning project for classifying paintings by artist using **Transfer Learning** and a structured experimental pipeline.

The project evolves from **Exploratory Data Analysis → Controlled Experiments → Final Modeling & Tuning**, using multiple CNN backbones and systematic evaluation.

---

## 📁 Project Structure

> Large data folders (`wikiart/`, `wikiart_split/`), checkpoints, and outputs are excluded via `.gitignore`.

---

## 🎯 Problem

Given an image of a painting, predict **which of 23 artists** created it.

- **Classes:** 23 artists  
- **Dataset size:** 13,340 images  
- **Challenges:**
  - Class imbalance
  - High intra-class variability (same artist, multiple styles)
  - Visual similarity across artists

---

## 🧠 Project Pipeline

### 1. Exploratory Data Analysis (`Data_Exploration.ipynb`)
- Class imbalance analysis
- Image statistics (size, ratio, brightness)
- RGB distribution per artist
- Edge detection (brushstroke patterns)
- Data quality checks (duplicates, corrupted files)

---

### 2. Experiments (`Experiments.ipynb`)
Used to **design the pipeline**, not final results:

- Data augmentation strategies
- Backbone comparison (e.g. EfficientNet, ConvNeXt)
- Number of unfrozen layers
- Classification head depth
- Overfitting vs generalisation analysis

---

### 3. Final Modeling (`Modeling.ipynb`)
- Uses best configurations found in experiments
- Full training pipeline:
  - Augmentation + preprocessing
  - Two-phase training
  - Hyperparameter tuning (Hyperband)
  - Final evaluation

---

## 🏗️ Architecture

Transfer Learning with ImageNet-pretrained backbones:

The classification head is **fully configurable** (number of layers, units, dropout) via the pipeline.

---

## ⚙️ Training Strategy

### Two-phase training

| Phase | Description |
|------|------------|
| Phase 1 | Backbone frozen → train classification head |
| Phase 2 | Partial unfreezing → fine-tuning |

Implemented in:
- `run_phase1(...)`
- `run_phase2(...)` :contentReference[oaicite:1]{index=1}  

---

## 🔬 Techniques

- **Data augmentation**
  - Flip, rotation, zoom
  - Brightness, contrast, saturation
  - Random Erasing

- **Class weights** → handle imbalance  
- **EarlyStopping + ReduceLROnPlateau**  
- **Transfer Learning** (ImageNet)  
- **Fine-tuning (partial unfreeze)**  
- **Hyperparameter tuning (Keras Tuner - Hyperband)**  
- **Test-Time Augmentation (TTA)**  

---

## 🧪 Hyperparameter Tuning

Hyperband is used to optimize:

- Learning rate
- Dropout
- Number of Dense layers
- Units per layer

Implemented via:
- `build_tuner_model(...)`
- `run_hyperband(...)` :contentReference[oaicite:2]{index=2}  

---

## 📊 Evaluation Metrics

- **Macro F1-score (primary)** → handles class imbalance
- Top-1 Accuracy
- Top-3 Accuracy
- Confusion Matrix (row-normalised)
- Overfitting gap (train vs validation)

---

## 🧩 Core Pipeline (functions.py)

Main components:

- Dataset processing:
  - `build_resized_ds`
  - `apply_preprocess_ds`
  - `apply_augmented_preprocess_ds`

- Model:
  - `build_base_model`

- Training:
  - `run_phase1`
  - `run_phase2`

- Experiments:
  - `run_augmentation_experiment`

- Tuning:
  - `run_hyperband`

This modular design allows **reproducible and controlled experiments**.

---

## 🛠️ Stack

| Library | Version |
|--------|--------|
| Python | 3.x |
| TensorFlow / Keras | 2.x / 3.x |
| NumPy | — |
| scikit-learn | — |
| OpenCV | — |
| Matplotlib / Seaborn | — |

---

## 🚀 Getting Started

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place dataset
wikiart/<artist>/*.jpg