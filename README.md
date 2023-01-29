# Deep Learning Techniques for Breast Cancer Detection in Mammograms 
This project forms the software artifact for a Masters Dissertation. 
The basic model is a Transfer Learning model with a base model trained on imagenet weights (defaults to MobileNetV2), 2 additional fully connected layers, one dropout layer and a classification layer with a sigmoid function to predict "Benign" or "Malignant" samples. 
The model has optional command line parameters to alter hyperparameters: learning rate, batch size, base model, image size and number of fine-tuning epochs. Other command line arguments can be found in the Usage section below. 

## Setup 
### DDSM data:
Once data and CSVs of ground truth are downloaded, alter the paths in `data_preparation/CBIS_DDSM_csv_preparation.py` and `data_preparation/import_data.py` to the location of the data and run `data_preparation/CBIS_DDSM_csv_combined.py` to generate the CSVs needed for training. 
Run python `create_ddsm_pngs.py` to convert the images into PNG from dicom and to store in an appropriate directory structure. 
DDSM dataset is provided in train and test sets then the train set is further split into train and validate sets in `main.py`

### CMMD data:
Download from TCIA and place images, metadata csv and clinical data csv into the same folder. 
Modify paths in `data_preparation/CMMD_preparation.py` and `data_preparation/import_data.py` to point to this folder. 
Run `CMMD_preparation.py` to prepare the labelled CSV for easy processing.
Run `create_data_pngs.py` to split/stratify the data as TRAIN/TEST/VAL sets and store in new directory structure in BENIGN or MALIGNANT folders (and converted to PNG images)


### Preprocessing Pipeline
Apply to dataset once in advance of training and store in directory to save time:
1. Alter paths in `data_preprocessing/shared_preprocessing.py` to point to location of separated data for DDSM and CMMD datasets (see above)
2. Call `run_pipeline()` for DDSM and CMMD to create all the preprocessed images and store them in directories
Preprocessing applied: cropping to breast area (DDSM only), wiener filter for noise reduction, CLAHE for contrast enhancement. 
Note: data augmentation is not applied at this stage to save space - augmentation is applied at training time and can be optionally included or excluded with a runtime flag (-a in command line arguments)

Before training - create two folders **/saved_models/** and **/output/** under /CNN_Mammogram_Classification_Pipeline

## Usage
Excluding the testmode flag means the program will train a model, evaluate it on the validation set (reporting results and graphs) and save the model as a `.h5` file using the name provided in the `-n` parameter. This saved model file can be given as the name parameter (excluding the `.h5`) when running in testmode `-t` and the program will load the saved model and evaluate it on the unseen test set. 

Model usage:
```
python main.py -d <dataset> -t -p -a -hy -lr <learning rate> -m <model name> -s <image size> -n <name> -b <batch size> - e <number fine-tune epochs>
```
* `-d` : select dataset for training/testing from "DDSM", "CMMD" or "BOTH, required parameter
* `-t`: testmode flag, sets the program to run in testmode when included
* `-p`: preprocessing flag, sets the program to use the pre-processed dataset when included (note Preprocessing Set Up from above must be completed first)
* `-a`: data augmentation flag, sets the program to trigger pre-specified data augmentations when included (horizontal and vertical flips, 90 degree rotations)
* `-hy`: hyperparameter-tuning flag, sets the program to run a hyperparameter tuning search with Hyperband when included (tunes the number of fully-connected layers after the base model and number of neurons in each layer, number of dropout layers and dropout rate in each layer, optimizer and learning rate
*` -lr`: set the learning rate, defaults to 0.0001 for training while base model is frozen (fine-tuning LR is 0.00001)
* `-m`: select a base model trained on imagenet for transfer learning: "MobileNet" (default), "VGG", "ResNet", "Inception", "DenseNet"
* `-s`: set image size: typically `160` (for 160 x 160 pixels) or `500` (500 x 500 pixels)
* `-n`: name of this trial (can be any value, should be entered the same for train or testmode - no need to include `.h5` at the end)
* `-b`: set the batch size, default 32
* `-e`: set the number of training epochs for fine-tuning, defaults to 50

