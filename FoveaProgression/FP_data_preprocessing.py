import os
import pandas as pd
from sklearn.model_selection import train_test_split

def verify_images(df, base_path):
    """ Verify if image files exist in the filesystem and filter out non-existing files. """
    # Concatenate the folder path and image filename to create a full path
    df['image_path'] = df.apply(lambda row: os.path.join(base_path, row['scan_name']), axis=1)
    
    # # Check if each image file exists
    # existing_images_df = df[df['image_path'].apply(os.path.exists)]
    
    return df

# def split_data(df):
#     """ Split data into training, validation, and testing datasets. """
#     # Splitting the data into train+validation and test sets
#     train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['status'])

#     # Splitting the train+validation set into train and validation sets
#     train_df, val_df = train_test_split(train_val_df, test_size=0.176, random_state=42, stratify=train_val_df['status'])  # 0.176 is approximately 15% of 0.85 (1-0.15)
    
#     return train_df, val_df, test_df

# def save_datasets(train_df, val_df, test_df):
#     """ Save the datasets to CSV files. """
#     train_df.to_csv('data/fovea_progression/train_dataset.csv', index=False)
#     val_df.to_csv('data/fovea_progression/validation_dataset.csv', index=False)
#     test_df.to_csv('data/fovea_progression/test_dataset.csv', index=False)

def main():
    df = pd.read_csv('data/fovea_progression_dataset_excluding_unknowns.csv')
    scan_img_path = r'/Volumes/fsmresfiles/Ophthalmology/Mirza_Images/AMD/dAMD_GA/all_slices_3'

    df_existing = verify_images(df, scan_img_path)

    if df_existing.empty:
        print("No images found. Check your image paths and CSV data.")
    else:
        print(df_existing.head())
        # train_df, val_df, test_df = split_data(df_existing)
        # save_datasets(train_df, val_df, test_df)
        # print(f"Train set size: {train_df.shape}")
        # print(f"Validation set size: {val_df.shape}")
        # print(f"Test set size: {test_df.shape}")

if __name__ == "__main__":
    main()
