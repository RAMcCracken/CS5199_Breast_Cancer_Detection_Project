import cv2
import data_preprocessing.ddsm_crop as dc
import os
from pathlib import Path
from scipy.misc import face
from scipy.signal import wiener
from scipy.signal import convolve2d
from skimage import color, data, restoration, img_as_float, img_as_uint
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 

def run_pipeline(dataset):
    dirs = ["TRAIN/ALL/BENIGN/", "TRAIN/ALL/MALIGNANT/", "TEST/ALL/BENIGN/", "TEST/ALL/MALIGNANT/"]

    if dataset == 'DDSM':
        data_path = "/data/ram31/CS5199_project/data/ddsm/"
        
    elif dataset == 'CMMD':
        data_path = "/data/ram31/CS5199_project/data/cmmd/"
        # ADD VAL folders
        dirs.append("VAL/ALL/BENIGN/")
        dirs.append("VAL/ALL/MALIGNANT/")
        
    parent = data_path + "PNG/"
    new_path = data_path + "PNG-PREPROC/"
    
    for dir in dirs:
        print("Preprocessing images in " + dir)
        Path(new_path + dir).mkdir(parents=True, exist_ok=True)
        if(dataset == 'DDSM'):
            preprocess_dir_ddsm(parent + dir, new_path + dir)
        elif(dataset == 'CMMD'):
            preprocess_dir_cmmd(parent + dir, new_path + dir)
        
        
def preprocess_dir_ddsm(source_path, dest_path):
    directory_list = os.listdir(source_path)
    for file_name in tqdm(directory_list):
        image_path = source_path + file_name
        image = cv2.imread(image_path)
        
        is_left = dc.check_is_left(image_path)
        
        # Pipeline
        cropped_image = dc.segment_image(image, is_left)
        noise_filtered = apply_wiener(cropped_image)
        enhanced_image = apply_clahe(noise_filtered)
        
        cv2.imwrite(dest_path + file_name, enhanced_image)
   
def preprocess_dir_cmmd(source_path, dest_path):
    directory_list = os.listdir(source_path)
    for file_name in tqdm(directory_list):
        image_path = source_path + file_name
        image = cv2.imread(image_path)
        
        noise_filtered = apply_wiener(image)
        enhanced_image = apply_clahe(noise_filtered)
        
        cv2.imwrite(dest_path + file_name, enhanced_image)    


def apply_wiener(image):
    img_gray = color.rgb2gray(img_as_float(image))
    psf = np.ones((3, 3)) / 25
    img = convolve2d(img_gray, psf, 'same')
    rng = np.random.default_rng()
    img += 0.1 * img.std() * rng.standard_normal(img.shape)
    deconvolved_img = restoration.wiener(img, psf, 0.4)
    
    return img_as_uint(deconvolved_img)


def apply_clahe(image):
    # Reading the image from the present directory
    # Resizing the image for compatibility
    image = cv2.resize(image, (500, 600))

    
    # The declaration of CLAHE
    # clipLimit -> Threshold for contrast limiting
    clahe = cv2.createCLAHE(clipLimit = 5)
    final_img = clahe.apply(image) + 30
         
    return final_img

if __name__ == '__main__':
    run_pipeline('DDSM')
    run_pipeline('CMMD')