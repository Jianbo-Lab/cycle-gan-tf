import numpy as np 
from os import listdir
from os.path import isfile, join
import matplotlib.image as mpimg
# def extract_files(path):
# 	filenames = [f for f in listdir(path)]
# 	for filename in filenames:
# 		header = filename.split('.')[0]
# 		birthyear = header.split('-')[1]
# 		photoyear = header.split('-')[-1]
# 		age = int(photoyear) - int(birthyear)

# 		image = mpimg.imread(filename)
# 		print image.shape
from glob import glob
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt 
import os
import pandas as pd
import h5py
import tqdm
data = glob(os.path.join("img_align_celeba", "*.jpg"))
data = np.sort(data)
def imread(path):
	return scipy.misc.imread(path).astype(np.float)

def resize_width(image, width=64.):
	h, w = np.shape(image)[:2]
	return scipy.misc.imresize(image,[int((float(h)/w)*width),width])
		
def center_crop(x, height=64):
	h= np.shape(x)[0]
	j = int(round((h - height)/2.))
	return x[j:j+height,:,:]

def get_image(image_path, width=64, height=64):
	return center_crop(resize_width(imread(image_path), width = width),height=height)

images = np.zeros((len(data),dim*dim*3), dtype = np.uint8)

# make a dataset
for i in tqdm.tqdm(range(len(data))):
    #for i in tqdm.tqdm(range(10)):
    image = get_image(data[i], dim,dim)
    images[i] = image.flatten()

attribute_file = 'list_attr_celeba.txt'

with open(attribute_file, 'r') as f:
    num_examples = f.readline()
    headers = f.readline()
headers = headers.split()

label_input = pd.read_fwf(attribute_file,skiprows=2,
                       widths = [10,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3, 
                                 3,3,3,3,3,3,3,3,3,3,3],
                   index_col=0,
                   header=None
                  )

labels = label_input.astype(int).as_matrix()

with h5py.File(''.join(['datasets/faces_dataset_new.h5']), 'w') as f:
    dset_face = f.create_dataset("images", data = images)
    dset_headers = f.create_dataset('headers', data = headers)
    dset_label_input = f.create_dataset('label_input', data = label_input)