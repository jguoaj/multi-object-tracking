'''
  File name: getFeatures.py
  Author: 
  Date created: 2018-11-04
'''

import cv2
import numpy as np
from skimage.feature import corner_harris, corner_shi_tomasi, peak_local_max

def getFeatures(img,bbox,use_shi=False):
    n_object = np.shape(bbox)[0]
    N = 0
    temp = np.empty((n_object,),dtype=np.ndarray)   # temporary storage of x,y coordinates
    for i in range(n_object):
        (xmin, ymin, boxw, boxh) = cv2.boundingRect(bbox[i,:,:].astype(int))
        roi = img[ymin:ymin+boxh,xmin:xmin+boxw]
        # cv2.imshow('roi',roi)
        if use_shi:
            corner_response = corner_shi_tomasi(roi)
        else:
            corner_response = corner_harris(roi)
        coordinates = peak_local_max(corner_response,num_peaks=20,exclude_border=2)
        coordinates[:,1] += xmin
        coordinates[:,0] += ymin
        temp[i] = coordinates
        if coordinates.shape[0] > N:
            N = coordinates.shape[0]
    x = np.full((N,n_object),-1)
    y = np.full((N,n_object),-1)
    for i in range(n_object):
        n_feature = temp[i].shape[0]
        x[0:n_feature,i] = temp[i][:,1]
        y[0:n_feature,i] = temp[i][:,0]
    return x,y

