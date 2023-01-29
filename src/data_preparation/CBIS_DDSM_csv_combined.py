"""
Simple method to combine mass and calcification CSVs for training and testing
"""
import CBIS_DDSM_csv_preparation as cb
import pandas as pd

def main(): 
    cb.main()
    combine_csv('calc-training.csv', 'mass-training.csv', "training")
    combine_csv('calc-test.csv', 'mass-test.csv', "testing")
    
def combine_csv(csv1, csv2, result_name):
    data_loc = "../data/CBIS-DDSM/"
    df_calc_csv = pd.read_csv(data_loc + csv1)
    df_mass_csv = pd.read_csv(data_loc + csv2)
    
    frames = [df_calc_csv, df_mass_csv]
    df_result = pd.concat(frames)
    df_result.to_csv(data_loc + result_name + ".csv", index=False)

if __name__ == "__main__":
    main()