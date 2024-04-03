import os
import pandas as pd

# #data paths
# img_labels = pd.read_csv('data/per_scan_data.csv')
# all_slices_path = 'dummy_data'

# #images
# all_slices = os.listdir(all_slices_path)
# all_imgs = [item for item in all_slices if item.endswith('.jpg')]

# #labels
# def find_label_patient_id(img_name, img_labels):
#     name_column = 'scan_name'

#     found_row = img_labels[img_labels[name_column] == img_name]
#     return found_row['status'].item()

# labels = []
# for i in range(len(all_imgs)):
#     img_name = all_imgs[i]

#     label = find_label(img_name, img_labels)
#     if label == True:
#         labels.append(1)
#     else:
#         labels.append(0)

# # Predefined lists of patient IDs for each set
# train_patient_id = [64,47,341]
# test_patient_id = [345,321,190]
# val_patient_id = [578,326]

def split_by_id(img_labels, all_imgs, labels, train_patient_id,test_patient_id,val_patient_id):

    all_imgs_df = pd.DataFrame({'scan_name': all_imgs, 'label': labels})
    # Merge to link each image with its patient ID and other details
    merged_df = pd.merge(all_imgs_df, img_labels, on='scan_name')

    train_df = merged_df[merged_df['patient_id'].isin(train_patient_id)]
    test_df = merged_df[merged_df['patient_id'].isin(test_patient_id)]
    val_df = merged_df[merged_df['patient_id'].isin(val_patient_id)]

    # train_df.info()

    # Splitting the training, testing and validation sets
    X_train = train_df['scan_name']
    Y_train = train_df['label']
    X_test = test_df['scan_name']
    Y_test = test_df['label']

    # Splitting the validation set
    X_val = val_df['scan_name']
    Y_val = val_df['label']

    # print(len(Y_train))
    # print(len(Y_val))
    # print(len(Y_test))

    return X_train, Y_train, X_test, Y_test, X_val, Y_val



