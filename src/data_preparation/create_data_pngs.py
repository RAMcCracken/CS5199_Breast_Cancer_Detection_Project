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
from import_data import import_cbisddsm_training_dataset, import_cbisddsm_testing_dataset, import_cmmd_training_dataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm 
import random

manifest = "/data/ram31/CS5199_project/data/cmmd/manifest-1666010201438/CMMD/"
parent_dir = manifest[:-29] #/data/ram31/CS5199_project/data/cmmd
benign_loc = parent_dir+"/BENIGN/"
malignant_loc = parent_dir+"/MALIGNANT/"
both_loc = parent_dir+"/BOTH/"
random.seed(111)
    
class FolderDef:   
    def __init__(self, parent_dir):
        self.TRAIN_MASS = parent_dir + "/PNG/TRAIN/MASS"
        self.TRAIN_CALC = parent_dir + "/PNG/TRAIN/CALC"
        self.TRAIN_BOTH = parent_dir + "/PNG/TRAIN/BOTH"
        self.VAL_MASS = parent_dir + "/PNG/VAL/MASS"
        self.VAL_CALC = parent_dir + "/PNG/VAL/CALC"
        self.VAL_BOTH = parent_dir + "/PNG/VAL/BOTH"
        self.TEST_MASS = parent_dir + "/PNG/TEST/MASS"
        self.TEST_CALC = parent_dir + "/PNG/TEST/CALC"
        self.TEST_BOTH = parent_dir + "/PNG/TEST/BOTH"
        self.UNKNOWN = parent_dir

def load_data():
    train_data_df = import_cmmd_training_dataset()
    # print(parent_dir)
    rearrange_cmmd(train_data_df)
    separate_data(train_data_df)
    # convert_dicom_to_png(parent_dir, train_data_df)

def checkBenignMalignant(row, src, basename):
    # store patient sample in Benign or Malignant folder depending on class
    if row['classification'] == 'Benign':
        dest = benign_loc+row['subject_id']+"/"+basename
        Path(benign_loc+row['subject_id']+"/").mkdir(parents=True, exist_ok=True)
    else:
        dest = malignant_loc+row['subject_id']+"/"+basename
        Path(malignant_loc+row['subject_id']+"/").mkdir(parents=True, exist_ok=True)

    # src_file = src 
    if os.path.exists(src):
        shutil.copyfile(src, dest)
    else:
        print("Warning: file not found " + src)

# Code modified from Craig Myles': cggm-mammography-classification
# https://github.com/CraigMyles/cggm-mammography-classification/blob/main/1_stratification_data_split.ipynb
def create_benign_malignant(df):
    matches = ["1-3.dcm", "1-4.dcm"]
    seen = False
    last_id_seen = ''
    last_class = '' #BENIGN/MALIGNANT/BOTH
    last_src = ''
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):  
        src = manifest+row['file_location'][7:]
        basename = os.path.basename(src) #<- basename = file name + extension
        
        # TEMP: avoid D2 for initial investigation
        if row['subject_id'][:2] == 'D2':
            continue 
        
        if row['subject_id'] != last_id_seen and last_id_seen != '':
            # store the previous patient samples in Benign or Malignant or Both folder depending on class
            if last_class == 'Benign':
                dest = benign_loc+last_id_seen +"/"
                Path(dest).mkdir(parents=True, exist_ok=True)
            elif last_class == 'Malignant':
                dest = malignant_loc+last_id_seen +"/"
                Path(dest).mkdir(parents=True, exist_ok=True)
            elif last_class == 'Both':
                dest = both_loc+last_id_seen +"/"
                Path(dest).mkdir(parents=True, exist_ok=True)
                
            src_dir = last_src[:(len(last_src)-len(basename))]
            if os.path.exists(src_dir):
                files = os.listdir(src_dir)
                for file in files:
                    full_file_name = os.path.join(src_dir, file)
                    if os.path.isfile(full_file_name):
                        # print(file)
                        shutil.copyfile(full_file_name, dest + file)
            else:
                print("Warning: file not found " + last_src)
            
            last_class = row['classification']
            last_id_seen = row['subject_id']
            last_src = src
        elif row['subject_id'] == last_id_seen:
            if last_class != row['classification']:
                last_class = "Both"
            last_id_seen = row['subject_id']
            last_src = src
            
        else:
            # first seen
            last_class = row['classification']
            last_id_seen = row['subject_id']
            last_src = src
            
            
# Code modified from Craig Myles': cggm-mammography-classification
# https://github.com/CraigMyles/cggm-mammography-classification/blob/main/1_stratification_data_split.ipynb
def rearrange_cmmd(df):
    #create directory if doesnt exist
    Path(benign_loc).mkdir(parents=True, exist_ok=True)
    #create directory if doesnt exist
    Path(malignant_loc).mkdir(parents=True, exist_ok=True)  
    #create directory if doesnt exist
    Path(both_loc).mkdir(parents=True, exist_ok=True)    
    
    print("Building class folders")
    create_benign_malignant(df)

def rm_dir(directiory):
    ## Try to remove tree
    try:
        shutil.rmtree(directiory)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))

def move_all_images(dir_list, data_location, benign_dest, malignant_dest, df):
    for j in tqdm(range(len(dir_list))):
        subdir = data_location+dir_list[j]
        # move all images in subfolder based on classification
        if not os.path.exists(subdir):
            print("skipped directory not found")
            continue
        subdir_list = os.listdir(subdir)
        
        scans = df[df['subject_id'] == dir_list[j]]
        files = scans['file_location']
        file_loc = files.iloc[0]
        file_loc = file_loc[:-7]
        print(file_loc)
        for file in subdir_list:
            scan_file = scans[scans['file_location'] == file_loc + file]
            file_path = subdir + "/" + file
            new_file_path_src = subdir + "/" + dir_list[j] + "_" + file
            print(os.rename(file_path, new_file_path_src))
            if len(scan_file['classification'].index) > 0:
                if scan_file['classification'].iloc[0] == 'Benign':
                    png_image = convert_to_png(new_file_path_src)
                    png_path = (benign_dest + "/" + dir_list[j] + "_" + file).replace('.dcm','.png')
                    cv2.imwrite(png_path, png_image)
                    # shutil.copy(new_file_path_src, benign_dest)
                else:
                    png_image = convert_to_png(new_file_path_src)
                    png_path = (malignant_dest + "/" + dir_list[j] + "_" + file).replace('.dcm','.png')
                    cv2.imwrite(png_path, png_image)
                    # shutil.copy(new_file_path_src, malignant_dest)
            else: 
                print("skipped: " + file_path)
        rm_dir(subdir)
            

# Modified from Craig Myles' code: https://github.com/CraigMyles/cggm-mammography-classification/blob/main/1_stratification_data_split.ipynb
def select_subset(data_location, benign_dest, malignant_dest, df):
    # Take 20% of data for a TEST/VAL set
    directory_list = os.listdir(data_location)
    count = len(os.listdir(data_location))/5
    count = math.ceil(count)
    # test_set = random.sample(os.listdir(data_location), count)  
    
    test_set = []
    i=0
    while i < count:
        index = random.randint(0,len(directory_list)-1)
        test_set.append(directory_list[index])
        i += 1
       
    
    move_all_images(test_set, data_location, benign_dest, malignant_dest, df)

    
def separate_data(df):
    # Create directories in parent_dir for PNG/TRAIN and PNG/TEST and subdirectories: CALC, MASS and ALL
    folder_enum = FolderDef(parent_dir)

    train_all_benign_loc = parent_dir + "/PNG/TRAIN/ALL/BENIGN"
    train_all_malignant_loc = parent_dir + "/PNG/TRAIN/ALL/MALIGNANT"
    val_all_benign_loc = parent_dir + "/PNG/VAL/ALL/BENIGN"
    val_all_malignant_loc = parent_dir + "/PNG/VAL/ALL/MALIGNANT"
    test_all_benign_loc = parent_dir + "/PNG/TEST/ALL/BENIGN"
    test_all_malignant_loc = parent_dir + "/PNG/TEST/ALL/MALIGNANT"

    Path(train_all_benign_loc).mkdir(parents=True, exist_ok=True)
    Path(train_all_malignant_loc).mkdir(parents=True, exist_ok=True)
    Path(val_all_benign_loc).mkdir(parents=True, exist_ok=True)
    Path(val_all_malignant_loc).mkdir(parents=True, exist_ok=True)
    Path(test_all_benign_loc).mkdir(parents=True, exist_ok=True)
    Path(test_all_malignant_loc).mkdir(parents=True, exist_ok=True)
    
    # Create TEST set
    select_subset(benign_loc, test_all_benign_loc, test_all_malignant_loc, df)
    select_subset(malignant_loc, test_all_benign_loc, test_all_malignant_loc, df)
    # Take 20% from BOTH as well
    select_subset(both_loc, test_all_benign_loc, test_all_malignant_loc, df)
    
    # Create VAL set
    select_subset(benign_loc, val_all_benign_loc, val_all_malignant_loc, df)
    select_subset(malignant_loc, val_all_benign_loc, val_all_malignant_loc, df)
    # Take 20% from BOTH as well
    select_subset(both_loc, val_all_benign_loc, val_all_malignant_loc, df)
    
    # Move remaining to TRAIN folder:
    directory_list = os.listdir(benign_loc)
    move_all_images(directory_list, benign_loc, train_all_benign_loc, train_all_malignant_loc, df)
    directory_list = os.listdir(malignant_loc)
    move_all_images(directory_list, malignant_loc, train_all_benign_loc, train_all_malignant_loc, df)
    directory_list = os.listdir(both_loc)
    move_all_images(directory_list, both_loc, train_all_benign_loc, train_all_malignant_loc, df)

    Path(folder_enum.TRAIN_MASS).mkdir(parents=True, exist_ok=True)
    Path(folder_enum.TRAIN_CALC).mkdir(parents=True, exist_ok=True)
    Path(folder_enum.TRAIN_BOTH).mkdir(parents=True, exist_ok=True)

    Path(folder_enum.VAL_MASS).mkdir(parents=True, exist_ok=True)
    Path(folder_enum.VAL_CALC).mkdir(parents=True, exist_ok=True)
    Path(folder_enum.VAL_BOTH).mkdir(parents=True, exist_ok=True)


    Path(folder_enum.TEST_MASS).mkdir(parents=True, exist_ok=True)
    Path(folder_enum.TEST_CALC).mkdir(parents=True, exist_ok=True)
    Path(folder_enum.TEST_BOTH).mkdir(parents=True, exist_ok=True)
    
    
        
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