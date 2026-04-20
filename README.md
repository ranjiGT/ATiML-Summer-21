# Object Detection using Semi-Supervised Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.6+](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://www.python.org/)

## Overview

This project explores **Semi-Supervised Learning (SSL)** approaches for image classification on the [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html) dataset (5,011 RGB images across 20 object classes). The central hypothesis is that SSL methods leveraging both labeled and unlabeled data can achieve competitive or superior results compared to purely supervised techniques — while significantly reducing the need for annotated data.

Developed as part of the **Advanced Topics in Machine Learning (ATiML) — Summer 2021** course at Otto von Guericke University Magdeburg.

## Methods

### Semi-Supervised Learning Algorithms

| Algorithm | Category | Implementation |
|---|---|---|
| **Label Propagation (LPA)** | Graph-based | `codebase/LabelPropagation.py` |
| **Label Spreading** | Graph-based | `codebase/LabelSpreading.py` |
| **Semi-Supervised GMM (SSGMM)** | Inductive (mixture model) | `codebase/SSGMM.py` |
| **Semi-Supervised SVM (S3VM/TSVM)** | Margin-based | `files/SSL-S3VM.py` |

### Feature Extraction

**Primary features:**
- **MPEG-7 Color Layout Descriptor** — image partitioning, DCT transformation, zigzag scanning
- **Visual Bag-of-Words (VBOW)** with **SURF** keypoints — codebook construction via k-means, histogram encoding
- **Speeded Up Robust Features (SURF)** — keypoint detection via OpenCV

![](https://github.com/ranjiGT/ATiML-Summer-21/blob/main/files/surf-1.jpg) ![](https://github.com/ranjiGT/ATiML-Summer-21/blob/main/files/surf-2.jpg)

**Supplementary features:**
- **Local Binary Patterns (LBP)** — texture descriptor
- **Color Histogram** — color distribution descriptor

### Data Pipeline

- PASCAL VOC XML annotation parsing with bounding box extraction
- Feature normalization and combination via PCA dimensionality reduction
- Stratified train/test split with configurable labeled/unlabeled ratios (0.1–0.9)
- Class balancing via `RandomUnderSampler` (imbalanced-learn)
- Feature selection via ANOVA (`SelectKBest` with `f_classif`)

## Project Structure

```
.
├── codebase/               # Final implementations (.py, .ipynb)
│   ├── LabelPropagation.py
│   ├── LabelSpreading.py
│   ├── SSGMM.py
│   └── SSGMM.ipynb
├── files/                  # Legacy pipeline and utility scripts
│   ├── data_loader.py
│   ├── pipeline.py
│   ├── pipeline_v2.py
│   ├── SSL-S3VM.py
│   └── VBOW.py
├── fig/plots/              # Result plots (box plots, learning curves)
├── HLD_LLD/                # High/Low Level Design documents (.pdf)
├── posterbase/             # LaTeX poster source (Gemini/beamer theme)
├── requirements.txt
└── LICENSE
```

## Getting Started

### Prerequisites

Python 3.6 or higher is required.

### Installation

```bash
pip install -r requirements.txt
```

### Running

```bash
python codebase/<model>.py --data path/to/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/
```

Replace `<model>` with one of: `LabelPropagation`, `LabelSpreading`, `SSGMM`.

## Results

### Scientific Poster

![Poster](https://github.com/ranjiGT/ATiML-Summer-21/blob/main/posterbase/Poster.png)

### Download PDF

![QR Code](https://github.com/ranjiGT/ATiML-Summer-21/blob/main/posterbase/flowcode.png)

## References

1. F. Perronnin, J. Sánchez and Yan Liu, "Large-scale image categorization with explicit data embedding," *2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, 2010, pp. 2297-2304, doi: [10.1109/CVPR.2010.5539914](https://doi.org/10.1109/CVPR.2010.5539914).

2. M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. *The PASCAL Visual Object Classes Challenge 2007 (VOC2007)*. [http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html)

## Contributors

- **Shubham Kumar Agrawal** — Label Propagation
- **Pavan Tummala** — Semi-Supervised GMM
- **Usama Ashfaq** ([@aaashfaq](https://github.com/aaashfaq), usama.ashfaq@volkswagen.de)
- **Syed Muhammad Laique Abbas** — Label Spreading
- **Ranji Raj** — Visual Bag-of-Words, project infrastructure

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
