import pandas as pd
import os

img_labels = pd.read_csv('FoveaDetection/fovea_detection_data.csv')
dummy_image_path = 'dummy.jpg'

os.makedirs('dummy_data', exist_ok=True)
with open(dummy_image_path, 'rb') as dummy_file:
    dummy_content = dummy_file.read()

for img_name in img_labels['scan_name']:
    new_image_path = os.path.join('dummy_data', f'{img_name}')
    # Create a copy of the dummy image with the new name
    with open(new_image_path, 'wb') as new_file:
        new_file.write(dummy_content)