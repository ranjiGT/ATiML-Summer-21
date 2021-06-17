# `Object detection using SSL Classification`

## `Preamble` :scroll:
The Proceedings of the European Conference on Computer Vision has conducted a benchmark PASCAL Visual Object Challenge (VOC) evaluating performance on object class recognition (from 2005-2012, now finished). For our task, we examine the 2007 dataset which consists of several types of random images collected in January 2007 from _Flickr_. There are five challenges: classification, detection, segmentation, action classification, and person layout.

Our goal from this challenge is to perform __image classification__ from several visual object classes in realistic scenes (i.e. not pre-segmented objects). And so, we will be using certain Semi-Supervised Learning approaches where we have both labeled and unlabeled sets to check if we get superior results as opposed to supervised techniques.

Mandatory features

- MPEG-7 Color Layout Descriptor
- Visual Bag-of-Words
- Speeded up robust features (SURF)

![](https://github.com/ranjiGT/ATiML-Summer-21/blob/main/files/surf-1.jpg) ![](https://github.com/ranjiGT/ATiML-Summer-21/blob/main/files/surf-2.jpg)


Supplementary features

- Local Binary Patterns
- Color Histogram

Folder Structure :open_file_folder:
============================

> Folder structure and naming conventions for this project

### A top-level directory layout

    .
    ├── HLD_LLD                  # Documentation files (.pdf)
    ├── codebase                  # workable codebase (.py, .ipynb)
    ├── fig
          └── plots                 # Compiled image files (.png, .jpg)
    ├── files                       # Legacy files                  
    ├── posterbase                  # layout and template files (.sty, .tex)                                      
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    └── requirements.txt          # Python libraries for this project (versioned)

### Usage

Code requires Python 3.6 to run.
Install the dependencies required for the code to run.

```
pip install -r requirements.txt
```

### For running the code
```
python {change_the_model_here}.py --data ..\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\
```
## Datasets and References🌐

- F. Perronnin, J. Sánchez and Yan Liu, "Large-scale image categorization with explicit data embedding," _2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition_, 2010, pp. 2297-2304, doi: 10.1109/CVPR.2010.5539914.

- Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. _The PASCAL Visual Object Classes Challenge 2007 (VOC2007)_ http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html


## Contributors :man_technologist:

- Shubham Kumar Agrawal
- Pavan Tummala
- Usama Ashfaq
- Syed Muhammad Laique Abbas
- Ranji Raj
