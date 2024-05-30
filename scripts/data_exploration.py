
import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def explore_data(data):
    print("Data shape:", data.shape)
    print("\nFirst few rows:\n", data.head())
    print("\nPatients count:", data['patient_id'].nunique())
    print("Scans per patient:\n", data.groupby('patient_id').size().describe())
    print("\nClass distribution:\n", data['sequence_label'].value_counts())
    print("\nMissing values:\n", data.isnull().sum())

def main():
    filepath = 'FoveaProgression/data/fovea_progression_dataset.csv'
    data = load_data(filepath)
    explore_data(data)

if __name__ == "__main__":
    main()
