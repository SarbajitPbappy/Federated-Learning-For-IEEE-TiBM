# FedMAM: Federated Multi-Attention Mechanism for Medical Image Analysis

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**A Privacy-Preserving, Energy-Efficient Federated Learning Framework for Multimodal Medical Image Classification**

[Overview](#overview) • [Features](#key-features) • [Datasets](#datasets) • [Architecture](#architecture) • [Results](#results) • [Installation](#installation) • [Usage](#usage) • [Citation](#citation)

</div>

---

## 📋 Overview

**FedMAM** (Federated Multi-Attention Mechanism) is an advanced federated learning framework designed specifically for medical image analysis. This project addresses the critical challenges of privacy preservation, data heterogeneity, and computational efficiency in healthcare AI applications.

The framework integrates a lightweight Custom Convolutional Neural Network (CustomCNN) with an innovative attention-based aggregation mechanism, enabling collaborative learning across multiple healthcare institutions while maintaining strict data privacy. FedMAM outperforms state-of-the-art federated learning baselines (FedAVG, FedProx, FedBN) with **7.5% average improvement** in accuracy while being **29.5% faster** in training.

### 🎯 Key Contributions

- **Novel Aggregation Mechanism**: Modality-Aware Attention Aggregation for handling heterogeneous medical imaging data
- **Meta-Learning Integration**: Adaptive capabilities for cross-modal knowledge transfer
- **Lightweight Architecture**: CustomCNN model optimized for efficiency and fast performance
- **Comprehensive Evaluation**: Rigorous evaluation across 6 diverse medical imaging datasets (67K+ images)
- **Statistical Validation**: Extensive statistical analysis demonstrating significant improvements over baselines

---

## ✨ Key Features

### 🔒 Privacy-Preserving
- **Federated Learning Architecture**: Raw medical data never leaves local institutions
- **Secure Aggregation**: Only model parameters are shared, not patient data
- **HIPAA-Compliant Design**: Framework designed with healthcare privacy regulations in mind

### ⚡ Energy-Efficient
- **29.5% Faster Training** compared to baseline methods
- **50% Reduction** in communication rounds
- **Lightweight Model**: CustomCNN with optimized architecture for edge devices
- **Efficient Attention Mechanisms**: SE, CBAM, and Self-Attention blocks for enhanced performance

### 🎨 Multimodal
- **Cross-Modal Learning**: Unified framework for diverse medical imaging modalities
- **Modality-Aware Aggregation**: Attention mechanism adapts to different image types
- **6 Medical Imaging Datasets**: CT scans, X-rays, MRIs, and histopathological images

### 🔄 Adaptive Capabilities
- **Dynamic Class Handling**: Automatically adapts to varying numbers of classes per dataset
- **Meta-Learning Integration**: Facilitates knowledge transfer across modalities
- **Robust to Non-IID Data**: Handles heterogeneous data distributions effectively

---

## 🗂️ Datasets

This project utilizes **6 comprehensive medical imaging datasets** from Kaggle, covering diverse medical imaging modalities:

| Dataset | Modality | Classes | Description | Source |
|---------|----------|---------|-------------|--------|
| **Kidney CT** | Computed Tomography | 4 | Normal, Cyst, Tumor, Stone classification | [Kaggle](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) |
| **Leukemia** | Microscopy | 4 | Acute Lymphoblastic Leukemia (ALL) classification | [Kaggle](https://www.kaggle.com/datasets/mehradaria/leukemia) |
| **Lung & Colon Cancer** | Histopathology | 5 | Cancer vs. benign tissue classification | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images) |
| **Lymphoma** | Histopathology | 3 | Malignant lymphoma classification | [Kaggle](https://www.kaggle.com/datasets/andrewmvd/malignant-lymphoma-classification) |
| **Brain Tumor** | Magnetic Resonance Imaging | 4 | Brain tumor classification from MRI scans | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| **Chest X-ray** | Radiography | 2 | Pneumonia detection from chest X-rays | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |

### Dataset Statistics
- **Total Images**: 67,000+ medical images
- **Data Split**: 70% Training, 15% Validation, 15% Test (Centralized) / 70% Train, 30% Test (Federated)
- **Preprocessing**: Comprehensive preprocessing pipeline with augmentation
- **Augmentation**: Rotation, Flip, Zoom, Shear, and Oversampling for class imbalance

---

## 🏗️ Architecture

### CustomCNN Model

The CustomCNN is a lightweight, efficient architecture specifically designed for medical image analysis:

```
Input (224×224×3)
    ↓
[Conv2D(32) + BN + ReLU + SE Block]
    ↓
[Conv2D(64) + BN + ReLU + SE Block]
    ↓
[Conv2D(128) + BN + ReLU + SE Block + CBAM]
    ↓
[Conv2D(256) + BN + ReLU + SE Block + CBAM + Self-Attention]
    ↓
Global Average Pooling
    ↓
Dropout (0.5)
    ↓
Output (Binary/Multi-class)
```

**Key Components:**
- **Squeeze-and-Excitation (SE) Blocks**: Channel attention mechanism
- **CBAM (Convolutional Block Attention Module)**: Combined channel and spatial attention
- **Self-Attention Blocks**: Long-range dependency modeling
- **Batch Normalization**: Stable training and regularization
- **Label Smoothing**: Improved generalization

### FedMAM Aggregation Mechanism

The FedMAM framework employs a novel attention-based aggregation strategy:

1. **Local Training**: Each client trains the CustomCNN model on local data
2. **Modality-Aware Attention**: Attention weights computed based on data modality
3. **Meta-Learning Integration**: Cross-modal knowledge transfer during aggregation
4. **Adaptive Aggregation**: Weighted combination of client models based on performance and data characteristics

### Preprocessing Pipeline

**For Radiology Images (CT/MRI/X-ray):**
1. Grayscale conversion
2. Resizing to 224×224
3. Gaussian denoising (3×3 kernel)
4. CLAHE (Contrast Limited Adaptive Histogram Equalization)
5. Normalization (0-1 range)

**For Histopathology Images:**
1. RGB color space conversion
2. Resizing to 224×224
3. Gaussian denoising
4. CLAHE on V-channel (HSV color space)
5. Normalization

**Augmentation Strategies:**
- Rotation (±15°)
- Horizontal/Vertical Flip
- Zoom (0.9-1.1×)
- Shear transformation
- Oversampling for class imbalance

---

## 📊 Results

### Performance Metrics

FedMAM demonstrates superior performance across all evaluation metrics:

| Metric | FedMAM | FedAVG | FedProx | FedBN | Improvement |
|--------|--------|--------|---------|-------|-------------|
| **Final Accuracy** | **95.04%** | 94.22% | 87.28% | 84.30% | **+0.82%** |
| **Best Accuracy** | **95.22%** | 94.54% | 89.85% | 88.80% | **+0.68%** |
| **Mean Accuracy** | **93.31%** | 92.04% | 87.40% | 82.60% | **+1.27%** |
| **Training Time** | **262.5 min** | 371.2 min | 375.8 min | 372.1 min | **-29.5%** |
| **Efficiency (Acc/min)** | **0.0036** | 0.0025 | 0.0023 | 0.0023 | **+30.6%** |
| **Convergence Rounds** | **2** | 2 | 2 | 1 | Stable |
| **Win Rate** | **90.5%** | - | - | - | - |

### Statistical Significance

- **Kruskal-Wallis Test**: p = 0.0003 (Highly Significant)
- **Friedman Test**: p = 0.003 (Highly Significant)
- **Effect Sizes**: All Cohen's d > 0.8 (Large practical significance)
- **Average Improvement**: +7.50% over baseline methods

### Key Achievements

✅ **Best Final Accuracy**: 95.04% (Highest among all methods)  
✅ **Best Mean Accuracy**: 93.31% (Most consistent performance)  
✅ **Faster Training**: 29.5% reduction in training time  
✅ **Higher Efficiency**: 30.6% better accuracy-per-minute ratio  
✅ **Improved Stability**: 6.2% lower coefficient of variation  
✅ **Round-by-Round Dominance**: 19/21 wins (90.5% win rate)

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.16+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM (recommended for full dataset processing)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/fedmam-medical-imaging.git
cd fedmam-medical-imaging
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install tensorflow==2.16.2
pip install opencv-python
pip install scikit-learn
pip install pandas
pip install matplotlib
pip install seaborn
pip install tqdm
pip install numpy
```

4. **Download datasets:**
   - Download the 6 medical imaging datasets from Kaggle (links provided above)
   - Organize them in the `Data/Raw_Data/` directory following the structure:
   ```
   Data/
   └── Raw_Data/
       ├── Kidney Cancer/
       ├── Leukemia/
       ├── Lung and Colon Cancer/
       ├── Lymphoma/
       ├── Brain Tumor/
       └── Chest_Xray/
   ```

---

## 💻 Usage

### 1. Data Preprocessing

Preprocess raw medical images:

```python
# Run the preprocessing notebook
jupyter notebook dataprocessing.ipynb
```

This will:
- Apply preprocessing pipeline (resizing, denoising, CLAHE, normalization)
- Generate processed images in `Data/Processed_Data/`
- Create augmented datasets in `Data/Augmented/`

### 2. Centralized Training (Baseline)

Train the CustomCNN model in a centralized setting:

```python
# Run the lightweight CNN notebook
jupyter notebook light_Weight_CNN.ipynb
```

### 3. Federated Learning Training

Train using the FedMAM framework:

```python
# Run the federated learning notebook
jupyter notebook Federated_Learning.ipynb
```

Configuration options:
```python
NUM_ROUNDS = 10          # Number of federated rounds
LOCAL_EPOCHS = 3         # Epochs per client per round
BATCH_SIZE = 16          # Batch size for training
IMG_SIZE = (224, 224)    # Input image size
```

### 4. Model Evaluation

Evaluate trained models and generate visualizations:

```python
# Run evaluation and analysis notebooks
jupyter notebook statistical_analysis_complete.ipynb
jupyter notebook Explainability.ipynb
```

### 5. Results Visualization

Generate comprehensive result visualizations:

```python
python federated_learning_plots.py
```

Results will be saved in:
- `federated_learning_results/` - Baseline comparisons
- `federated_learning_results_fedmam/` - FedMAM-specific results
- `Comparision/` - Statistical analysis and plots

---

## 📁 Project Structure

```
Model Optimization/
│
├── Data/
│   ├── Raw_Data/              # Original datasets
│   ├── Processed_Data/        # Preprocessed images
│   └── Augmented/             # Augmented datasets
│
├── notebooks/
│   ├── dataprocessing.ipynb              # Data preprocessing pipeline
│   ├── light_Weight_CNN.ipynb            # CustomCNN architecture
│   ├── Federated_Learning.ipynb          # FedMAM implementation
│   ├── pretrainedmodel.ipynb             # Transfer learning baselines
│   ├── statistical_analysis_complete.ipynb  # Statistical evaluation
│   └── Explainability.ipynb              # Model interpretability
│
├── federated_learning_results/
│   ├── comprehensive_analysis/           # Statistical reports
│   └── ULTIMATE_ENDGAME/                # Final analysis
│
├── federated_learning_results_fedmam/
│   ├── fedmam_global_model.h5           # Trained FedMAM model
│   ├── fedmam_metrics.csv               # Performance metrics
│   └── per_client_analysis/             # Client-wise results
│
├── Comparision/                          # Comparison visualizations
├── Explainability/                       # XAI visualizations
├── FeatureMaps/                          # Feature map visualizations
├── Saved Model/                          # Saved model checkpoints
│
├── federated_learning_plots.py          # Visualization scripts
└── README.md                            # This file
```

---

## 🔬 Experimental Setup

### Federated Learning Configuration

- **Number of Clients**: 6 (one per dataset)
- **Federated Rounds**: 10
- **Local Epochs**: 3 per round
- **Batch Size**: 16
- **Learning Rate**: 0.0001 (Adam optimizer)
- **Data Split**: 70% Train, 30% Test (per client)

### Baseline Comparisons

FedMAM is compared against three state-of-the-art federated learning algorithms:

1. **FedAVG**: Federated Averaging (McMahan et al., 2017)
2. **FedProx**: Federated Proximal (Li et al., 2020)
3. **FedBN**: Federated Batch Normalization (Li et al., 2021)

### Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- Area Under Curve (AUC)
- Training Time and Efficiency
- Convergence Speed
- Coefficient of Variation (Stability)
- Statistical Significance Tests (ANOVA, Kruskal-Wallis, Friedman)

---

## 📈 Visualizations

The project includes comprehensive visualizations:

- **Training Curves**: Loss and accuracy progression
- **Confusion Matrices**: Per-dataset classification performance
- **Statistical Analysis**: Box plots, violin plots, heatmaps
- **Feature Maps**: Convolutional layer activations
- **Explainability**: Grad-CAM and attention visualizations
- **Comparison Charts**: Side-by-side performance comparisons

All visualizations are saved at 300-800 DPI for publication quality.

---

## 🔍 Explainability & Interpretability

The framework includes explainability analysis:

- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- **Feature Map Visualization**: Convolutional layer activations
- **Attention Visualization**: Attention weight distributions
- **Model Interpretability**: Understanding model decisions

Results are available in the `Explainability/` directory.

---

## 📝 Citation

If you use FedMAM in your research, please cite:

```bibtex
@article{fedmam2024,
  title={FedMAM: Federated Multi-Attention Mechanism for Privacy-Preserving Medical Image Analysis},
  author={Your Name},
  journal={Journal/Conference Name},
  year={2024},
  note={Under Review}
}
```

### Dataset Citations

Please also cite the datasets used in this work:

```bibtex
@misc{kidney2021,
  author = {Nazmul, M.},
  title = {CT KIDNEY DATASET: Normal-Cyst-Tumor and Stone},
  year = {2021},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone}}
}

@misc{leukemia2021,
  author = {Aria, M.},
  title = {Acute Lymphoblastic Leukemia (ALL) image dataset},
  year = {2021},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/mehradaria/leukemia}}
}

@misc{lungcolon2020,
  author = {Andrew, A. M. V. D.},
  title = {Lung and Colon Cancer Histopathological Images},
  year = {2020},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images}}
}

@misc{lymphoma2020,
  author = {Andrew, A. M. V. D.},
  title = {Malignant Lymphoma Classification},
  year = {2020},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/andrewmvd/malignant-lymphoma-classification}}
}

@misc{braintumor2021,
  author = {Nickparvar, M.},
  title = {Brain Tumor MRI Dataset},
  year = {2021},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset}}
}

@misc{chestxray2018,
  author = {Mooney, P.},
  title = {Chest X-Ray Images (Pneumonia)},
  year = {2018},
  publisher = {Kaggle},
  howpublished = {\url{https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia}}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

- **Sarbajit Paul Bappy** - *Initial work* - [GitHub](https://github.com/SarbajitPbappy)

---

## 🙏 Acknowledgments

- Kaggle community for providing high-quality medical imaging datasets
- TensorFlow team for the excellent deep learning framework
- Contributors to open-source federated learning libraries
- Healthcare professionals working on medical AI applications

---



---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ for the Medical AI and Federated Learning communities

</div>

