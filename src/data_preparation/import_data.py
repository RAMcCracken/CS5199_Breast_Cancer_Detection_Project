"""
Code modified from Adam Jaamour's repository
DOI: https://doi.org/10.5281/zenodo.3985051
"""
import os

from imutils import paths
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

class FolderDef:   
    def __init__(self, parent_dir):
        self.TRAIN_MASS = parent_dir + "PNG/TRAIN/MASS"
        self.TRAIN_CALC = parent_dir + "PNG/TRAIN/CALC"
        self.TEST_MASS = parent_dir + "PNG/TEST/MASS"
        self.TEST_CALC = parent_dir + "PNG/TEST/CALC"
        self.UNKNOWN = parent_dir

def import_cmmd_training_dataset():
    """
    Import the dataset getting the image paths (downloaded on BigTMP) and encoding the labels.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param label_encoder: The label encoder.
    :return: Two arrays, one for the image paths and one for the encoded labels.
    """
    print("Importing CBIS-DDSM training set")
    cmmd_path = "/data/ram31/CS5199_project/rhona_pipeline/data/CMMD/CMMD_metadata_subset.csv"
    df = pd.read_csv(cmmd_path)

    return df

def import_cbisddsm_training_dataset():
    """
    Import the dataset getting the image paths (downloaded on BigTMP) and encoding the labels.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param label_encoder: The label encoder.
    :return: Two arrays, one for the image paths and one for the encoded labels.
    """
    print("Importing CBIS-DDSM training set")
    cbis_ddsm_path = "/data/ram31/CS5199_project/rhona_pipeline/data/CBIS-DDSM/training.csv"
    df = pd.read_csv(cbis_ddsm_path)
    list_IDs = df['img_path'].values
    folders = df['img'].values
    # labels = encode_labels(df['label'].values, label_encoder)
    return folders, list_IDs, df['label'].values

def import_cbisddsm_testing_dataset():
    """
    Import the dataset getting the image paths (downloaded on BigTMP) and encoding the labels.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param label_encoder: The label encoder.
    :return: Two arrays, one for the image paths and one for the encoded labels.
    """
    print("Importing CBIS-DDSM test set")
    cbis_ddsm_path = "../data/CBIS-DDSM/testing.csv"
    df = pd.read_csv(cbis_ddsm_path)
    list_IDs = df['img_path'].values
    folders = df['img'].values
    return folders, list_IDs, df['label'].values

def encode_labels(labels_list: np.ndarray, label_encoder) -> np.ndarray:
    """
    Encode labels using one-hot encoding.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param label_encoder: The label encoder.
    :param labels_list: The list of labels in NumPy array format.
    :return: The encoded list of labels in NumPy array format.
    """
    labels = label_encoder.fit_transform(labels_list)
    if label_encoder.classes_.size == 2:
        return labels
    else:
        return to_categorical(labels)