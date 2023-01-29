# DDSM data:
Once data and CSVs of ground truth are downloaded, alter the paths in data_preparation/CBIS_DDSM_csv_preparation.py and data_preparation/import_data.py to the location of the data and run data_preparation/CBIS_DDSM_csv_combined.py to generate the CSVs needed for training. 
Run python create_ddsm_pngs.py to convert the images into png from dicom and to store in an appropriate directory structure. 
DDSM datasets are split into train and val in main.py
Run python main.py to train and test the model.

# CMMD data:
Download from TCIA and place images, metadata csv and clinical data csv into the same folder. 
Modify paths in data_preparation/CMMD_preparation.py and data_preparation/import_data.py to point to this folder. 
Run CMMD_preparation.py to move all images into BENIGN or MALIGNANT folders
Run create_data_pngs.py to split/stratify the data as TRAIN/TEST/VAL sets and store in new directory structure


# Preprocessing Pipeline
Apply to dataset once in advance of trianing and store in directory to save time:
1. Alter paths in data_preprocessing/shared_preprocessing.py to point to location of separated data for DDSM and CMMD datasets (see above)
2. Call run_pipeline() for DDSM and CMMD to create all the preprocessed images and store them in directories
Preprocessing applied: cropping to breast area (DDSM only), wiener filter for noise reduction, CLAHE for contrast enhancement. 
Note: data augmentation is not applied at this stage to save space - augmentation is applied at training time and can be optionally included or excluded with a runtime flag (-a in command line arguments)

Before training - create two folders **/saved_models/** and **/output/** under /CNN_Mammogram_Classification_Pipeline