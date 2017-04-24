import os
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt   
import h5py # for reading our dataset 

with h5py.File(''.join(['faces_dataset_new.h5']), 'r') as hf:
    faces = hf['images'].value
    headers = hf['headers'].value
    labels = hf['label_input'].value

# extract Young
labels = labels[:,-1]
faces = (faces/255.)
trainsize = 1000
testsize = 100
import scipy.misc
ind_young = 0
ind_old = 0
for i,face in enumerate(faces):
    label = labels[i]
    if label == 1:
        if ind_young < trainsize:
            ind_young += 1
            scipy.misc.imsave('celebA/trainA/image-{}.jpg'.format(i),face.reshape((64,64,3)))
        elif ind_young < trainsize+testsize:
            ind_young += 1
            scipy.misc.imsave('celebA/testA/image-{}.jpg'.format(i),face.reshape((64,64,3)))
    else:
        if ind_old < trainsize:
            ind_old += 1
            scipy.misc.imsave('celebA/trainB/image-{}.jpg'.format(i),face.reshape((64,64,3)))
        elif ind_old < trainsize+testsize:
            ind_old += 1
            scipy.misc.imsave('celebA/testB/image-{}.jpg'.format(i),face.reshape((64,64,3)))
