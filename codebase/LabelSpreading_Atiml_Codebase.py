#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Thu May 27 2021

@author: Laique Abbas
"""
import os, numpy as np
import cv2
import random
import torch
import torch.utils.data as data
import xml.etree.ElementTree as ET
from abc import ABCMeta, abstractmethod
import scipy.cluster.vq as vq
import pickle
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from cv2 import imread, resize
from numpy import concatenate
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelSpreading
from sklearn.model_selection import train_test_split
import argparse
from imblearn.under_sampling import RandomUnderSampler
from skimage import feature
import warnings
warnings.filterwarnings("ignore")


"""
Data Loader reading the files, extracting individual objects from each image
"""

class DataLoader(data.Dataset):
    def __init__(self,data_path="", trainval='trainval',transform=None):
        self.data_path = data_path
        self.transform = transform
        self.trainval = trainval
        self.__init_classes()
        self.names, self.labels, self.lable_set, self.bounding_box = self.__dataset_info()
    
    def __getitem__(self, index):
        self.data = []
        self.lables = []
        x = imread(self.data_path+'JPEGImages/'+self.names[index]+'.jpg')
        #x = resize(x, (256,256))
        #print(self.bounding_box[index])
        x_min, y_min, x_max, y_max  = self.bounding_box[index]
        for i in range(len(x_min)):
            #print(i)
            sub_img = x[y_min[i]:y_max[i],x_min[i]:x_max[i]]
            #print(sub_img.shape)
            #sub_img = resize(sub_img, (64,64))
            
            sub_img = cv2.resize(sub_img, (64, 64), 
                           interpolation=cv2.INTER_NEAREST)
            self.data.append(sub_img)
            self.lables.append(self.lable_set[index][i])
            
        #print(self.lable_set[index])
        #print(len(self.lable_set[index]))
        #print(len(self.bounding_box[index]))
        #x = Image.fromarray(x)
        if self.transform !=None:
            x = self.transform(x)
        y = self.labels[index]
        #return x, y
    
    def __fetchdata__(self):
        return self.data, self.lables
        
    
    def __len__(self):
        return len(self.names)
    
    def __dataset_info(self):
        #annotation_files = os.listdir(self.data_path+'/Annotations')
        with open(self.data_path+'ImageSets/Main/'+self.trainval+'.txt') as f:
            annotations = f.readlines()
        annotations = [n[:-1] for n in annotations]
        names = []
        labels = []
        lable_set = []
        bounding_box = []
        for af in annotations:
            filename = os.path.join(self.data_path,'Annotations',af)
            tree = ET.parse(filename+'.xml')
            objs = tree.findall('object')
            num_objs = len(objs)
            
            bdg_box = [obj.find('bndbox') for obj in objs]
            x_min = [int(box.find('xmin').text.lower().strip()) for box in bdg_box]
            y_min = [int(box.find('ymin').text.lower().strip()) for box in bdg_box]
            x_max = [int(box.find('xmax').text.lower().strip()) for box in bdg_box]
            y_max = [int(box.find('ymax').text.lower().strip()) for box in bdg_box]
            coords = (x_min, y_min, x_max, y_max)
                        
            boxes_cl = np.zeros((num_objs), dtype=np.int32)
            temp_lbls = []
            for ix, obj in enumerate(objs):
                cls = self.class_to_ind[obj.find('name').text.lower().strip()]
                boxes_cl[ix] = cls
                temp_lbls.append(cls)
            
            lbl = np.zeros(self.num_classes)
            lbl[boxes_cl] = 1
            labels.append(lbl)
            names.append(af)
            lable_set.append(temp_lbls)
            bounding_box.append(coords)
        
        return np.array(names), np.array(labels).astype(np.float32), lable_set, bounding_box
    
    def __init_classes(self):
        self.classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self.num_classes  = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        

"""
Local Binary Pattern
"""
class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist
        
        
"""
color layout descriptor
"""
        
class DescriptorComputer:
	__metaclass__ = ABCMeta
	
	@abstractmethod
	def compute(self, frame):
		pass



class ColorLayoutComputer(DescriptorComputer):
    def __init__(self):
        self.rows = 8
        self.cols = 8
        self.prefix = "CLD"
    
    def compute(self, img):
        averages = np.zeros((self.rows,self.cols,3))
        imgH, imgW, _ = img.shape
        for row in range(self.rows):
            for col in range(self.cols):
                row_start = int(imgH/self.rows * row)
                row_end = int(imgH/self.rows * (row+1))
                col_start = int(imgW/self.cols*col)
                col_end = int(imgW/self.cols*(col+1))
                slice1 = img[row_start:row_end, col_start:col_end]
                #slice1 = img[imgH/self.rows * row: imgH/self.rows * (row+1), imgW/self.cols*col : imgW/self.cols*(col+1)]
                #print(slice)
                average_color_per_row = np.mean(slice1, axis=0)
                average_color = np.mean(average_color_per_row, axis=0)
                average_color = np.uint8(average_color)
                averages[row][col][0] = average_color[0]
                averages[row][col][1] = average_color[1]
                averages[row][col][2] = average_color[2]
        icon = cv2.cvtColor(np.array(averages, dtype=np.uint8), cv2.COLOR_BGR2YCR_CB)
        y, cr, cb = cv2.split(icon)
        dct_y = cv2.dct(np.float32(y))
        dct_cb = cv2.dct(np.float32(cb))
        dct_cr = cv2.dct(np.float32(cr))
        dct_y_zigzag = []
        dct_cb_zigzag = []
        dct_cr_zigzag = []
        flip = True
        flipped_dct_y = np.fliplr(dct_y)
        flipped_dct_cb = np.fliplr(dct_cb)
        flipped_dct_cr = np.fliplr(dct_cr)
        for i in range(self.rows + self.cols -1):
            k_diag = self.rows - 1 - i
            diag_y = np.diag(flipped_dct_y, k=k_diag)
            diag_cb = np.diag(flipped_dct_cb, k=k_diag)
            diag_cr = np.diag(flipped_dct_cr, k=k_diag)
            if flip:
                diag_y = diag_y[::-1]
                diag_cb = diag_cb[::-1]
                diag_cr = diag_cr[::-1]
            dct_y_zigzag.append(diag_y)
            dct_cb_zigzag.append(diag_cb)
            dct_cr_zigzag.append(diag_cr)
            flip = not flip
        return np.concatenate([np.concatenate(dct_y_zigzag), np.concatenate(dct_cb_zigzag), np.concatenate(dct_cr_zigzag)])
    
    

"""
Bag of Visual word
"""

device = torch.device('cpu')

def random_init(dataset, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    used = torch.zeros(num_points, dtype=torch.long)
    indices = torch.zeros(num_centers, dtype=torch.long)
    for i in range(num_centers):
        while True:
            cur_id = random.randint(0, num_points - 1)
            if used[cur_id] > 0:
                continue
            used[cur_id] = 1
            indices[i] = cur_id
            break
    indices = indices.to(device)
    centers = torch.gather(dataset, 0, indices.view(-1, 1).expand(-1, dimension))
    return centers


def compute_codes(dataset, centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)
    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(5e8 / num_centers)
    codes = torch.zeros(num_points, dtype=torch.long, device=device)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers ** 2, dim=1).view(1, -1)
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
    return codes


def update_centers(dataset, codes, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    centers = torch.zeros(num_centers, dimension, dtype=torch.float, device=device)
    cnt = torch.zeros(num_centers, dtype=torch.float, device=device)
    centers.scatter_add_(0, codes.view(-1, 1).expand(-1, dimension), dataset)
    cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float, device=device))
    # Avoiding division by zero
    # Not necessary if there are no duplicates among the data points
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(num_centers, dtype=torch.float, device=device))
    centers /= cnt.view(-1, 1)
    return centers

def cluster(dataset, num_centers):
    centers = random_init(dataset, num_centers)
    codes = compute_codes(dataset, centers)
    num_iterations = 0
    while True:
        num_iterations += 1
        centers = update_centers(dataset, codes, num_centers)
        new_codes = compute_codes(dataset, centers)
        # Waiting until the clustering stops updating altogether
        # This is too strict in practice
        if torch.equal(codes, new_codes):
            print('Converged in %d iterations' % num_iterations)
            break
        codes = new_codes
    return centers, codes

def extract_sift_descriptors(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors


def build_codebook(X, voc_size):
    """
    Inupt a list of feature descriptors
    voc_size is the "K" in K-means, k is also called vocabulary size
    Return the codebook/dictionary
    """
    features = np.vstack((descriptor for descriptor in X)).astype(np.float32)
    dataset = torch.from_numpy(features)
    print('Starting clustering')
    centers, codes = cluster(dataset, voc_size)
    return centers


def input_vector_encoder(feature, codebook):
    """
    Input all the local feature of the image
    Pooling (encoding) by codebook and return
    """
    code, _ = vq.vq(feature, codebook)
    word_hist, bin_edges = np.histogram(code, bins=range(codebook.shape[0] + 1), normed=True)
    return word_hist

    
def extract_surf_descriptors(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()
    keypoints, descriptors = surf.detectAndCompute(gray, None)
    #descriptors = descriptors.flatten()
    #fi = surf.integral(gray.copy())
    #points = surf.interest_points(fi, 6, 24, 1, max_points=1024, is_integral=True)
    #descs = surf.descriptors(fi, points, is_integral=True, descriptor_only=True)
    return descriptors

"""
Histogram Features
"""

def fd_histogram(image, mask=None):
    bins=8
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
    
"""
Feature Normalization
"""
    
def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                       help='path for voc2007')
    
    args = parser.parse_args()
    path = args.data
    #path = r'C:\backup(laique)\Desktop\atiml\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007/'
    data_load = DataLoader(data_path = path)
        
    lst_data = []
    lst_lbl = []
    
    for i in range(0, 5000):
        data_load.__getitem__(i)
        test_data, test_label = data_load.__fetchdata__()
    
        lst_data.append(test_data)
        lst_lbl.append(test_label)
    
    labels = np.hstack(lst_lbl)
    data = np.concatenate(lst_data, axis=0)
    print(len(data))
    print("################### Data load completed #######################")
    
    
    """
    color layour features
    """
    computer = ColorLayoutComputer()
    color_layout_features = [computer.compute(data[i]) for i in range(len(data))]
    
    print("################### color layout feature generated #######################")
    
    VOC_SIZE = 128
# =============================================================================
#     """
#     visual bag of words using sift
#     """
#     bow_sift = [extract_sift_descriptors(data[i].astype('uint8')) for i in range(len(data))]
#     bow_sift = [each for each in zip(bow_sift, labels) if not each[0] is None]
#     bow_sift, y_train = zip(*bow_sift)
#     
#     codebook = build_codebook(bow_sift, voc_size=VOC_SIZE)
#     bow_sift = [input_vector_encoder(x, codebook) for x in bow_sift]
# =============================================================================
    
    """
    visual bag of words using surf
    """
    bow_surf = [extract_surf_descriptors(data[i].astype('uint8')) for i in range(len(data))]
    bow_surf = [each for each in zip(bow_surf, labels) if not each[0] is None]
    bow_surf, y_train = zip(*bow_surf)
    
    codebook = build_codebook(bow_surf, voc_size=VOC_SIZE)
    bow_surf = [input_vector_encoder(x, codebook) for x in bow_surf]
    
    print("################### viusal bag of word and surf generated #######################")

    """
    color histogram
    """
    color_hist_features = [fd_histogram(data[i].astype('uint8')) for i in range(len(data))]
    
    print("################### color histogram generated #######################")
    
    """
    local binary pattern
    """
    
    desc = LocalBinaryPatterns(24, 8)
    lbp = [desc.describe(data[i]) for i in range(len(data))]

    print("################### local binary pattern generated #######################")
    
    bow_surf = np.array(bow_surf)
    color_layout_features = np.array(color_layout_features)
    color_hist_features = np.array(color_hist_features)
    lbp = np.array(lbp)

    # with open('color_layout_descriptor_64.pkl','wb') as f:
    #     pickle.dump(color_layout_features, f)
        
    # with open('bow_surf_64.pkl','wb') as f:
    #     pickle.dump(bow_surf, f)
        
    # with open('hist_64.pkl','wb') as f:
    #     pickle.dump(color_hist_features, f)
        
    # with open('labels_64.pkl','wb') as f:
    #     pickle.dump(labels, f)
        
    # with open('data_64.pkl','wb') as f:
    #     pickle.dump(data, f)



    """
    pickle read
    """
    
    # color_layout_features  = pd.read_pickle(path + "/color_layout_descriptor_64.pkl")
    # bow_surf  = pd.read_pickle(path + "/bow_surf_64.pkl")
    # color_hist_features  = pd.read_pickle(path + "/hist_64.pkl")
    # labels  = pd.read_pickle(path +"/labels_64.pkl")
    # data  = pd.read_pickle(path +"/data_64.pkl")


    """
    Normalizing color layour feature only
    since other features have been normalized while feature extraction above
    """
    color_layout_features_scaled = scale(color_layout_features, 0, 1)
    
    """
    stacking all the features into one array
    """
    features = np.hstack([color_layout_features_scaled,color_hist_features])
    features = features.astype('float64')

    """
    Feature Selection using Anova, 
    K is the hyper parameter that needs to be varied and tested
    """

    fs = SelectKBest(score_func=f_classif, k=150)
    fs.fit(features, labels)
    features = fs.transform(features)
    
    print("################### Feature Selection Completed #######################")
    
    undersample = RandomUnderSampler(random_state=123)
    X_over, y_over = undersample.fit_resample(features, labels)
    X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_over, y_over, test_size=0.1, random_state=1, stratify=y_over)
    
    print("################### Class Balancing Completed #######################")
    
    print("Labelled features set Size: %d, %d"%X_train_lab.shape)
    print("Labelled label set Size: %d"%y_train_lab.shape)
    print("UnLabelled features set Size: %d, %d"%X_test_unlab.shape)
    print("UnLabelled label set Size: %d"%y_test_unlab.shape)
    
    X_train_mixed = concatenate((X_train_lab, X_test_unlab))
    nolabel = [-1 for _ in range(len(y_test_unlab))]
    y_train_mixed = concatenate((y_train_lab, nolabel))

    model = LabelSpreading(max_iter=5000,gamma=15)
    model.fit(X_train_mixed, y_train_mixed)
    
    print("################### Label Spreading model built #######################")
    
    yhat = model.predict(X_train_mixed[y_train_mixed==-1])
    print("Unlabelled Dataset Accuracy: %f%% for %s%% of Unlabelled Dataset"% ((accuracy_score(y_test_unlab, yhat) * 100), 0.1 *100))
    
    


# In[ ]:





# In[ ]:




