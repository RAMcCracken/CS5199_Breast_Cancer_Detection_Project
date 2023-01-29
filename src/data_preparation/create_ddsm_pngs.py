import pandas as pd
import numpy as np
from pathlib import Path
import os
import shutil
import math
import pydicom
import cv2
import tensorflow as tf
import tensorflow_io as tfio
from import_data import import_cbisddsm_training_dataset, import_cbisddsm_testing_dataset, import_cmmd_training_dataset, import_cmmd_testing_dataset
from sklearn.preprocessing import LabelEncoder
from import_data import FolderDef

def load_data(dataset):
    manifest = ''
    parent_dir = ''
    folders_train, folders_test, images_train, images_test, labels_train, labels_test = ''
    
    if dataset == 'DDSM':
        manifest = "/data/ram31/CS5199_project/data/ddsm/manifest-1665504314468/CBIS-DDSM/"
        parent_dir = manifest[:-33]

        folders_train, images_train, labels_train = import_cbisddsm_training_dataset()
        folders_test, images_test, labels_test = import_cbisddsm_testing_dataset()

    elif dataset == 'CMMD':
        manifest = "/data/ram31/CS5199_project/data/cmmd/manifest-1666010201438/CMMD/"
        parent_dir = manifest[:-37]
        folders_train, images_train, labels_train = import_cmmd_training_dataset()
        folders_test, images_test, labels_test = import_cmmd_testing_dataset()

    convert_dicom_to_png(parent_dir, folders_train, images_train, labels_train)
    convert_dicom_to_png(parent_dir, folders_test, images_test, labels_test)

def convert_dicom_to_png(parent_dir, folders, images, labels, dataset):
    # Create directories in parent_dir for PNG/TRAIN and PNG/TEST and subdirectories: CALC, MASS and ALL
    folder_enum = FolderDef(parent_dir)

    train_all_loc = parent_dir + "PNG/TRAIN/ALL"
    test_all_loc = parent_dir + "PNG/TEST/ALL"

    Path(test_all_loc).mkdir(parents=True, exist_ok=True)
    Path(train_all_loc).mkdir(parents=True, exist_ok=True)

    Path(folder_enum.TRAIN_MASS).mkdir(parents=True, exist_ok=True)
    Path(folder_enum.TRAIN_CALC).mkdir(parents=True, exist_ok=True)

    Path(folder_enum.TEST_MASS).mkdir(parents=True, exist_ok=True)
    Path(folder_enum.TEST_CALC).mkdir(parents=True, exist_ok=True)

    # Iterate through images converting from dicom to png and storing result in specific folder(s) 
    # e.g. Mass-Training_P_02079_RIGHT_CC_1 -> PNG/TRAIN/MASS and PNG/TRAIN/ALL
    for i in range(len(images)):
        directory_path = find_folder(folder_enum, folders[i])

        png_image = convert_to_png(images[i])

        class_path = ''
        if labels[i] == "BENIGN":
            class_path = "/BENIGN"

        elif labels[i] == "MALIGNANT":
            class_path = "/MALIGNANT"
        else:
            print("CLASS PROBLEM, class reported:" + str(labels[i]))

        dest_path = directory_path + class_path

        # store in subdirectory based on classification
        if not os.path.exists(dest_path):
            Path(dest_path).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(dest_path + "/img_" + folders[i] + '.png', png_image)
        print("Saved image " + str(folders[i]) + " to " + str(dest_path))

        # also store in ALL folder for TRAIN or TEST 
        if directory_path == folder_enum.TRAIN_CALC or directory_path == folder_enum.TRAIN_MASS:
            dest_path = train_all_loc + class_path
        elif directory_path == folder_enum.TEST_CALC or directory_path == folder_enum.TEST_MASS:  
            dest_path = test_all_loc + class_path
        
        if not os.path.exists(dest_path):
            Path(dest_path).mkdir(parents=True, exist_ok=True)   
        cv2.imwrite(dest_path + "/img_" + folders[i] + '.png', png_image)
        print("Saved image " + str(folders[i]) + " to " + str(dest_path))


def find_folder(folder_enum, foldername):
    is_mass = False

    if foldername[:4] == "Mass":
        is_mass = True
    elif foldername[:4] == "Calc":
        is_mass = False
    else: return folder_enum.UNKNOWN
    
    if foldername[5:13] == "Training":
        if is_mass:
            return folder_enum.TRAIN_MASS
        else:
            return folder_enum.TRAIN_CALC
    elif foldername[5:9] == "Test":
        if is_mass:
            return folder_enum.TEST_MASS
        else:
            return folder_enum.TEST_CALC
    else: return folder_enum.UNKNOWN


def convert_to_png(filename):
    ds = pydicom.dcmread(filename)
    data = ds.pixel_array

    return data      

"""
Function modified by Rhona McCracken from Adam's Jaamour's code: 
DOI: https://doi.org/10.5281/zenodo.3985051
"""
def parse_file(filename, label):
    image_bytes = tf.io.read_file(filename)
    image = tfio.image.decode_dicom_image(image_bytes, dtype=tf.uint16) #color_dim=True
    
    # for resizing:
    image_png = tf.image.encode_png(image[0])
    decoded_png = tf.io.decode_png(image_png, channels=1)
    image = tf.image.resize_with_pad(decoded_png, 224, 224) #set height and width based on model/preprocessing
    image /= 255
    return image, label

if __name__ == "__main__":
    load_data()