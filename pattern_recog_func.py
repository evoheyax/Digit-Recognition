from __future__ import print_function, division
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2

from sklearn.decomposition import PCA
from sklearn import svm

from scipy.interpolate import interp2d, RectBivariateSpline

import os

def interpol_im(im, dim1 = 8, dim2 = 8, plot_new_im = False, cmap = 'binary', grid_off = False):
    
    if((len(im.shape)) == 3):
        im = im[:, :, 0]
    
    x = np.arange(im.shape[1])
    y = np.arange(im.shape[0])
    
    f2d = interp2d(x, y, im)
    
    x_new = np.linspace(0, im.shape[1], dim1)
    y_new = np.linspace(0, im.shape[0], dim2)
    
    new_im = f2d(x_new, y_new)
    if(plot_new_im):
        plt.grid(True)
        if(grid_off):
            plt.grid(False)
        plt.imshow(new_im, cmap = cmap)
        plt.show()
    
    new_im_flat = new_im.flatten()
    return new_im, new_im_flat

def pca_svm_pred(imfile, md_pca, md_clf, dim1 = 45, dim2 = 60):
    
    intp_img, intp_img_flat = interpol_im(imfile, dim1 = dim1, dim2 = dim2, plot_new_im = True)
    
    intp_img_flat = intp_img_flat.reshape(1, -1)
    
    X_proj = md_pca.transform(intp_img_flat)

    return md_clf.predict(X_proj)

def pca_X(X, n_comp = 10):
    md_pca = PCA(n_comp, whiten = True)
    
    # finding pca axes
    md_pca.fit(X)
    
    # projecting training data onto pca axes
    X_proj = md_pca.transform(X)
    
    return md_pca, X_proj

def rescale_pixel(unseen, ind = 0):
    unseen_rescaled = np.array((unseen * 15), dtype=np.int)
    for i in range(8):
        for j in range(8):
            if(unseen_rescaled[i, j] == 0):
                unseen_rescaled[i, j] = 15
            else:
                unseen_rescaled[i, j] = 0

    return unseen_rescaled

def svm_train(X, y, gamma = 0.001, C = 100):
    md_clf = svm.SVC(kernel='rbf', class_weight='balanced', gamma=gamma, C=C)
    
    # apply SVM to training data and draw boundaries.
    md_clf.fit(X, y)
    
    return md_clf