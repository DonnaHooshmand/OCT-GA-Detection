
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    return pd.read_csv(filepath)

def stratified_split_data(data):
    # 'label' can be a proxy for stratification if detailed patient conditions are not available
    # Adjust based on actual stratification needs
    patients = data['patient_id'].unique()
    # Split patients into groups keeping the distribution of labels similar across splits
    train_val_patients, test_patients = train_test_split(patients, test_size=0.20, random_state=42, stratify=data.groupby('patient_id')['sequence_label'].first())
    train_patients, val_patients = train_test_split(train_val_patients, test_size=0.25, random_state=42, stratify=data[data['patient_id'].isin(train_val_patients)].groupby('patient_id')['sequence_label'].first())

    train_data = data[data['patient_id'].isin(train_patients)]
    val_data = data[data['patient_id'].isin(val_patients)]
    test_data = data[data['patient_id'].isin(test_patients)]

    return train_data, val_data, test_data

def save_splits(train_data, val_data, test_data, directory='data/data_splits'):
    train_data.to_csv(f'{directory}/train_data.csv', index=False)
    val_data.to_csv(f'{directory}/val_data.csv', index=False)
    test_data.to_csv(f'{directory}/test_data.csv', index=False)

def main():
    filepath = 'FoveaProgression/data/fovea_progression_dataset.csv'
    data = load_data(filepath)
    train_data, val_data, test_data = stratified_split_data(data)
    save_splits(train_data, val_data, test_data)

if __name__ == "__main__":
    main()
