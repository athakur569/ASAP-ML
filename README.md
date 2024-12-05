# ASAP-ML

A modular, extensible machine learning framework for digital pathology — designed for stain-robust, multi-output architectures (e.g., SAMA-Net) and efficient dataset preprocessing.

---

## Features
- Modular data pipeline with dynamic transforms  
- Patch-based histopathology loaders (CSV, WSI, HDF5 support)  
- Metadata tracking for reproducible experiments  
- PyTorch-based model training (mixed precision, grad accumulation in upcoming commits)

---

## Structure
ASAP-ML/
├── data/              # Dataset loaders, augmentations, metadata
├── models/            # Architectures (U-Net, SAMA-Net, etc.)
├── utils/             # Metrics, config parsers, visualization helpers
├── scripts/           # Entry points for training, preprocessing
└── notebooks/         # Exploratory Jupyter notebooks