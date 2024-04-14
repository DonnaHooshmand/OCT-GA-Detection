
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def data_split(scan_metadata_path, slices_path):

    data_scan = pd.read_csv(scan_metadata_path)
    slices = os.listdir(slices_path)
    scan_paths = [item for item in slices if item.endswith('.jpg')]

    filtered_data = data_scan[data_scan['scan_name'].isin(scan_paths)]
    filtered_data['label'] = filtered_data['status'].astype(int)
    sorted_filtered_data = filtered_data.sort_values(by=['patient_id','scan_number'])
    sorted_filtered_data.reset_index(inplace=True)

    df = sorted_filtered_data[['patient_id', 'eye_side', 'scan_number', 'scan_name', 'status', 'label']]

    x = df['scan_name']
    y = df['label']

    # Split the dataset into training and temp (validation + test) sets -> train = 60% / temp = 40%
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)

    # Split the temp set into validation and test sets -> val = 20% / test = 20%
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    return df, x_train, y_train, x_val, y_val, x_test, y_test