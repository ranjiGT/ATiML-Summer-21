#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import decomposition
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


path = r'C:\Users\User\Downloads\ATMLVOC_PCKL/'

color_layout_features  = pd.read_pickle(path + "/color_layout_descriptor.pkl")
bow_surf  = pd.read_pickle(path + "/bow_surf.pkl")
color_hist_features  = pd.read_pickle(path + "/hist.pkl")
labels  = pd.read_pickle(path +"/labels.pkl")


# # Feat. Scaling

# In[3]:


def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom 


# In[4]:


#Scaled features
color_layout_features_scaled = scale(color_layout_features, 0, 1)
color_hist_features_scaled = scale(color_hist_features, 0, 1)
bow_surf_scaled = scale(bow_surf, 0, 1)


# In[5]:


features = np.hstack([color_layout_features_scaled, color_hist_features_scaled, bow_surf_scaled])
X, Y = features,labels
np.unique(labels)


# # PCA

# In[6]:


features=pd.DataFrame(features)
lb=pd.DataFrame(labels)
features['target']=lb
X1, Y1 = features,labels


# In[15]:


pca = decomposition.PCA(n_components=4)
pc = pca.fit_transform(X1)

pc_df = pd.DataFrame(data = pc , 
        columns = ['PC1', 'PC2','PC3','PC4'])
pc_df['Cluster'] = Y1
pc_df.head()
#pc_df.names()


# In[12]:


sns.lmplot( x="PC1", y="PC2",
  data=pc_df, 
  fit_reg=False, 
  hue='Cluster', # color by cluster
  legend=True,
  scatter_kws={"s": 80})


# In[16]:


pc_df.columns


# # tSNE

# In[9]:


fashion_tsne = TSNE(n_components=2).fit_transform(features)


# In[10]:


def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    #ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
        
fashion_scatter(fashion_tsne, labels)

