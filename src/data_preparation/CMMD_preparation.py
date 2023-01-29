"""
Rhona McCracken modified code from the following:
Code reference: 
Author: Craig Myles
Github: CraigMyles, https://github.com/CraigMyles/cggm-mammography-classification/blob/main/0_Data_Exploration.ipynb
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pydicom
from pydicom import dcmread
import glob
from tqdm import tqdm
import os

cmmd_manifest_directory = "/data/ram31/CS5199_project/data/cmmd/manifest-1666010201438/"
path_to_clinical_data = "/data/ram31/CS5199_project/rhona_pipeline/data/CMMD/CMMD_clinicaldata_revision.csv"

path_to_metadata = cmmd_manifest_directory+"metadata.csv"
clinical_data = pd.read_csv(path_to_clinical_data)
meta_data = pd.read_csv(path_to_metadata)

meta_subset = meta_data.loc[:, ['Subject ID', 'Number of Images', 'File Location']]

df1 = clinical_data.rename(columns={"ID1": "Subject_ID"})
df2 = meta_subset.rename(columns={"Subject ID": "Subject_ID"})

df3 = pd.merge(df1, df2, on = 'Subject_ID')
df3.set_index('Subject_ID', inplace = True)

df3['img_1'] = ''
df3['img_2'] = ''

#Sort for cases where single patient_id's have >2 mammography images
for i in tqdm(range(len(df3))):
    if i==0:
        continue
        
    #If the current file path is the same as the previous path
    if df3.iloc[i]['File Location'] == df3.iloc[i-1]['File Location']:
        #If file path is the same as previous, that means there is >2
        df3.iloc[i, df3.columns.get_loc('img_1')] = "1-3.dcm"
        df3.iloc[i, df3.columns.get_loc('img_2')] = "1-4.dcm"
    else:
        df3.iloc[i, df3.columns.get_loc('img_1')] = "1-1.dcm"
        df3.iloc[i, df3.columns.get_loc('img_2')] = "1-2.dcm"

#Create new empty dataframe
df4 = pd.DataFrame(columns=["subject_id", "leftright", "age", "abnormality",
                            "classification", "subtype", "file_location"])

df3 = df3.reset_index()

# Iterate through each line in the dataframe, determine file
# location based on odd/even integers for scan type. (see pdf
# for data stratificaiton explanation)
def create_row(i, df4, flag=False):
    appended_data = []
    j = 0
    while j < 2:

        if not flag:
            if j == 0:
                file_loc = str(df3.iloc[i, df3.columns.get_loc('File Location')])+"/1-1.dcm"
            else:
                file_loc = str(df3.iloc[i, df3.columns.get_loc('File Location')])+"/1-2.dcm"
        
        if flag:
            if j == 0:
                file_loc = str(df3.iloc[i, df3.columns.get_loc('File Location')])+"/1-3.dcm"
            else:
                file_loc = str(df3.iloc[i, df3.columns.get_loc('File Location')])+"/1-4.dcm"
# Uncomment if debugging
#         print("iteration:"+str(i))
#         print(file_loc)
        new_row = {
            'subject_id':    df3.iloc[i, df3.columns.get_loc('Subject_ID')],
            'leftright':     df3.iloc[i, df3.columns.get_loc('LeftRight')],
            'age':           df3.iloc[i, df3.columns.get_loc('Age')],
            'abnormality':   df3.iloc[i, df3.columns.get_loc('abnormality')],
            'classification':df3.iloc[i, df3.columns.get_loc('classification')],
            'subtype':       df3.iloc[i, df3.columns.get_loc('subtype')],
            'file_location': file_loc
        }
        appended_data.append(new_row)
        df4 = df4.append(new_row, ignore_index=True)
#         print(len(df4))
        j += 1
    return appended_data

#For all items in the manifest
for i in tqdm(range(len(df3))):
    #skip 0th item because cant compare to -1th item.
    if i==0:
        #create regular row
        data_to_append = create_row(i, df4)
        
        df4 = df4.append(data_to_append, ignore_index=True)
        continue
    #if the file location equals the same as the one before...
    if df3.iloc[i]['File Location'] == df3.iloc[i-1]['File Location']:
        #True because this folder has 1,2,3,4 images
        data_to_append = create_row(i, df4, True)
        
        df4 = df4.append(data_to_append, ignore_index=True)
    else:
        data_to_append = create_row(i, df4)
        
        df4 = df4.append(data_to_append, ignore_index=True)
        
df4.to_csv("./CMMD_metadata_subset.csv", index=False)

#Append the /path/to/manifest/ to "1-1.dcm" or "1-2.dcm" etc...
for i in tqdm(range(len(df4))):
    begin_path = cmmd_manifest_directory[:-1]
    acc_file = df4.iloc[i]['file_location']
    my_file_loc = str(begin_path+acc_file[1:])
    if not os.path.isfile(my_file_loc):
        print("WARNING, the following file does not exist:\n"+my_file_loc)
        
df4.to_csv('CMMD_metadata.csv', encoding='utf-8')