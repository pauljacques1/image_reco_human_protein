import pandas as pd
import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

import h5py
import time
from os import listdir
import os 

test_labels_data = pd.read_csv('sample_submission.csv')

def merge_rgb(img_id, colours=['red','green','blue'], path = 'test'):

    """
    For each images, returns an array of shape (M,N,4) 
    where each dimension in the 4 are red, blue, green and yellow.
    """
    merged_colour_img = []

    for colour in colours:
        full_path = path + '/' + img_id + '_' + colour + '.png'
        colour_img=mpimg.imread(full_path)
        merged_colour_img.append(colour_img)    

    merged_colour_img = np.dstack((merged_colour_img))
    return merged_colour_img


def test_data_label(test_labels_data):
    
    """
    From the train_labels csv file, create a list of labels, and create a large 
    array for the train data in same order.
    """
    test_ids = [img_id for img_id in test_labels_data['Id']]
    #train_labels = [label for label in train_labels_data['Target']]

    print ('Labels and Ids collected')
    
    i=0
    
    for img_id in test_ids:
        
        print ('Merging Image')
        test_data_img = merge_rgb (img_id)
        print ('Merging done, appending the (M,N,4) array to a list')

        path = 'test_merged_rgb' + '/' + img_id 
        print ('Saving image')
        plt.imsave(path, test_data_img)
        #np.savez_compressed(path,img_id = train_data_img)
        print ('Done appending, going to next image')
        i +=  1
        print(i)



test_data_label(test_labels_data)