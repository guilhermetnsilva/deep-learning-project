# WikiArt Painter Classification

Deep learning project for classifying paintings by artist using Transfer Learning with EfficientNetB0.

## Structure

```
deep-learning-project/
├── wikiart_classification.ipynb   # Main notebook (EDA + training + evaluation)
├── requirements.txt               # Python dependencies
├── figures/                       # Figures generated during EDA
│   ├── class_distribution.png
│   ├── rgb_per_artist.png
│   ├── brightness_per_artist.png
│   ├── sample_paintings.png
│   └── ...
├── .gitignore
└── README.md
```

> Dataset (`wikiart/`), split folders (`wikiart_split/`), checkpoints, and outputs are excluded from the repository via `.gitignore`.

## Problem

Given a painting image, predict **which of 23 artists** created it — a 23-class image classification problem with 13,340 images.

**Artists:** Albrecht Dürer, Boris Kustodiev, Camille Pissarro, Childe Hassam, Claude Monet, Edgar Degas, Eugene Boudin, Gustave Doré, Ilya Repin, Ivan Aivazovsky, Ivan Shishkin, John Singer Sargent, Marc Chagall, Martiros Saryan, Nicholas Roerich, Pablo Picasso, Paul Cézanne, Pierre-Auguste Renoir, Pyotr Konchalovsky, Raphael Kirchner, Rembrandt, Salvador Dalí, Vincent van Gogh.

## Approach

### Exploratory Data Analysis (EDA)
- Class distribution and artist imbalance
- Image dimension, file size, and aspect ratio statistics
- Mean RGB channels and brightness per artist
- Average painting per artist (dominant palette)
- Canny edge detection (brushstroke density)
- Data quality checks: corrupted files, wrong channels, duplicates, outliers

### Architecture
Transfer Learning with **EfficientNetB0** pretrained on ImageNet:

```
Input (224×224×3)
  └─ EfficientNetB0 backbone (ImageNet weights)
       └─ GlobalAveragePooling2D
            └─ BatchNormalization
                 └─ Dense(512, relu) + Dropout(0.4)
                      └─ Dense(256, relu) + Dropout(0.3)
                           └─ Dense(23, softmax)
```

### Training — two phases
| Phase | Backbone | LR | Max epochs |
|-------|----------|----|------------|
| 1 — Feature Extraction | Frozen | 1e-3 | 30 |
| 2 — Fine-tuning (top 30 layers) | Partially unfrozen | 1e-4 | 50 |

### Techniques
- **Data augmentation**: horizontal flip, rotation, zoom, translation, brightness, contrast, hue, saturation, Random Erasing
- **Class weights** to handle class imbalance
- **EarlyStopping** + **ReduceLROnPlateau**
- **Test-Time Augmentation (TTA)** with 10 passes (typical gain of +1–2%)

## Evaluation Metrics
- Top-1 Accuracy (primary metric)
- Top-3 Accuracy
- Macro F1-Score
- Row-normalised Confusion Matrix

## Stack

| Library | Version |
|---------|---------|
| Python | 3.x |
| TensorFlow | 2.20.0 |
| Keras | 3.13.2 |
| NumPy | 2.4.2 |
| scikit-learn | 1.8.0 |
| OpenCV | 4.13.0 |
| Pillow | 12.1.1 |
| Matplotlib | 3.10.8 |
| Seaborn | 0.13.2 |

## Getting Started

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset under wikiart/<artist>/*.jpg

# 4. Open the notebook
jupyter notebook wikiart_classification.ipynb
```

> On the first run, the notebook splits the dataset into `wikiart_split/train`, `wikiart_split/val`, and `wikiart_split/test` and copies the images there. Subsequent runs load directly from those folders.
