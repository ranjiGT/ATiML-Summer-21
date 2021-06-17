# -*- coding: utf-8 -*-
"""
Created on Thu June 17 11:53:42 2021

@author: Pavan Tummala
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
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split
import argparse
from imblearn.under_sampling import RandomUnderSampler
from skimage import feature
import warnings

from scipy.sparse import issparse

from sklearn.naive_bayes import _BaseDiscreteNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
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
        x_min, y_min, x_max, y_max  = self.bounding_box[index]
        for i in range(len(x_min)):
            sub_img = x[y_min[i]:y_max[i],x_min[i]:x_max[i]]
            
            sub_img = cv2.resize(sub_img, (64, 64), 
                           interpolation=cv2.INTER_NEAREST)
            self.data.append(sub_img)
            self.lables.append(self.lable_set[index][i])
            
        if self.transform !=None:
            x = self.transform(x)
        y = self.labels[index]
    
    def __fetchdata__(self):
        return self.data, self.lables
        
    
    def __len__(self):
        return len(self.names)
    
    def __dataset_info(self):
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
local binary pattern
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
    return descriptors

"""
Histogram features
"""

def fd_histogram(image, mask=None):
    bins=8
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
    
"""
feature normalization
"""
    
def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

class MultinomialNBSS(_BaseDiscreteNB):
    """
    Semi-supervised Naive Bayes classifier for multinomial models.  Unlabeled
    data must be marked with -1.  In comparison to the standard scikit-learn
    MultinomialNB classifier, the main differences are in the _count and fit
    methods.

    Parameters
    ----------
    alpha : float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).

    beta : float, optional (default=1.0)
        Weight applied to the contribution of the unlabeled data
        (0 for no contribution).

    fit_prior : boolean, optional (default=True)
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    class_prior : array-like, size (n_classes,), optional (default=None)
        Prior probabilities of the classes. If specified the priors are not
        adjusted according to the data.

    tol : float, optional (default=1e-3)
        Tolerance for convergence of EM algorithm.

    max_iter : int, optional (default=1500)
        Maximum number of iterations for EM algorithm.

    verbose : boolean, optional (default=True)
        Whether to output updates during the running of the EM algorithm.

    Attributes
    ----------
    class_log_prior_ : array, shape (n_classes, )
        Smoothed empirical log probability for each class.

    intercept_ : array, shape (n_classes, )
        Mirrors ``class_log_prior_`` for interpreting MultinomialNBSS
        as a linear model.

    feature_log_prob_ : array, shape (n_classes, n_features)
        Empirical log probability of features
        given a class, ``P(x_i|y)``.

    coef_ : array, shape (n_classes, n_features)
        Mirrors ``feature_log_prob_`` for interpreting MultinomialNBSS
        as a linear model.

    class_count_ : array, shape (n_classes,)
        Number of samples encountered for each class during fitting. This
        value is weighted by the sample weight when provided.

    feature_count_ : array, shape (n_classes, n_features)
        Number of samples encountered for each (class, feature)
        during fitting. This value is weighted by the sample weight when
        provided.
    """

    def __init__(self, alpha=1.0, beta=1.0, fit_prior=True, class_prior=None,
                 tol=1e-3, max_iter=1500, verbose=True):
        self.alpha = alpha
        self.beta = beta
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def _count(self, X, Y, U_X=np.array([]), U_prob=np.array([])):
        """Count and smooth feature occurrences."""
        if np.any((X.data if issparse(X) else X) < 0):
            raise ValueError("Input X must be non-negative")

        self.feature_count_ = safe_sparse_dot(Y.T, X)
        self.class_count_ = Y.sum(axis=0)

        if U_X.shape[0] > 0:
            self.feature_count_ += self.beta*safe_sparse_dot(U_prob.T, U_X)
            self.class_count_ += self.beta*U_prob.sum(axis=0)
        else:
            self.feature_count_ = safe_sparse_dot(Y.T, X)
            self.class_count_ = Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.feature_log_prob_ = (np.log(smoothed_fc) -
                                  np.log(smoothed_cc.reshape(-1, 1)))

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        check_is_fitted(self, "classes_")

        X = check_array(X, accept_sparse='csr')
        return (safe_sparse_dot(X, self.feature_log_prob_.T) +
                self.class_log_prior_)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """A semi-supervised version of this method has not been implemented.
        """

    def fit(self, X, y, sample_weight=None):
        """Fit semi-supervised Naive Bayes classifier according to X, y

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.  Unlabeled data must be marked with -1.

        sample_weight : array-like, shape = [n_samples], (default=None)
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, 'csr')
        _, n_features = X.shape
        # Unlabeled data are marked with -1
        unlabeled = np.flatnonzero(y == -1)
        labeled = np.setdiff1d(np.arange(len(y)), unlabeled)

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y[labeled])
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently;
        # this means we also don't have to cast X to floating point
        Y = Y.astype(np.float64, copy=False)
        if sample_weight is not None:
            sample_weight = np.atleast_2d(sample_weight)
            Y *= check_array(sample_weight).T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        n_effective_classes = Y.shape[1]

        alpha = self._check_alpha()
        self._count(X[labeled], Y)


        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        jll = self._joint_log_likelihood(X)
        sum_jll = jll.sum()

        # Run EM algorithm
        if len(unlabeled) > 0:
            self.num_iter = 0
            pred = self.predict(X)
            while self.num_iter < self.max_iter:
                self.num_iter += 1
                prev_sum_jll = sum_jll

                # First, the E-step:
                prob = self.predict_proba(X[unlabeled])

                # Then, the M-step:
                self._count(X[labeled], Y, X[unlabeled], prob)
                self._update_feature_log_prob(self.beta)
                self._update_class_log_prior(class_prior=class_prior)

                jll = self._joint_log_likelihood(X)
                sum_jll = jll.sum()
                if self.verbose:
                    print(
                        'Step {}: jll = {:f}'.format(self.num_iter, sum_jll)
                    )

                if self.num_iter > 1 and prev_sum_jll - sum_jll < self.tol:
                    break

            if self.verbose:
                end_text = 's.' if self.num_iter > 1 else '.'
                print(
                    'Optimization converged after {} '
                    'iteration'.format(self.num_iter)
                    + end_text
                )

        return self
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                       help='path for voc2007')
    
    args = parser.parse_args()
    path = args.data
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
    
    print("################### Color layout feature generated #######################")
    
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
    
    print("################### Visual bag of words and surf generated #######################")

    """
    color histogram
    """
    color_hist_features = [fd_histogram(data[i].astype('uint8')) for i in range(len(data))]
    
    print("################### Color Histogram generated #######################")
    
    """
    local binary pattern
    """
    
    desc = LocalBinaryPatterns(24, 8)
    lbp = [desc.describe(data[i]) for i in range(len(data))]

    print("################### Local Binary Pattern generated #######################")
    
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
    features = np.hstack([color_layout_features_scaled, color_hist_features, lbp])
    features = features.astype('float64')

    """
    feature selection using Anova, 
    K is the hyper param that needs to be varied and tested
    """
    
    fs = SelectKBest(score_func=f_classif, k=200)
    fs.fit(features, labels)
    selected_features = fs.transform(features)
    
    print("################### Feature Selection completed #######################")
    
    undersample = RandomUnderSampler(random_state=123)
    X_over, y_over = undersample.fit_resample(selected_features, labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.1, random_state=42)
    X_train_lab, X_test_unlab, y_train_lab, y_test_unlab = train_test_split(X_train, y_train, test_size=0.1, random_state=1, stratify=y_train)
    
    print("################### Class Balancing completed #######################")
    
    print("Labelled features set size: %d, %d"%X_train_lab.shape)
    print("Labelled lable set size: %d"%y_train_lab.shape)
    print("Unlabelled features set size: %d, %d"%X_test_unlab.shape)
    print("Unlabelled lable set size: %d"%y_test_unlab.shape)
    
    X_train_mixed = concatenate((X_train_lab, X_test_unlab))
    nolabel = [-1 for _ in range(len(y_test_unlab))]
    y_train_mixed = concatenate((y_train_lab, nolabel))
    
    model = MultinomialNBSS(verbose=False)
    model.fit(X_train_mixed, y_train_mixed)
    
    print("################### SSGMM model built #######################")
    
    yhat = model.predict(X_test)
    print("Test data accuracy: %.2f%%"% (accuracy_score(y_test, yhat)*100))
