import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model

# Load the trained model
model_path = '/home/user/Desktop/Mosquito_Detector.zip/mobilenetv2.h5'
model = load_model(model_path)

# Input and output paths
input_image_folder = '/home/user/Desktop/Mosquito_Detector.zip/train_images'
output_csv_path = 'predictions.csv'

# Load the CSV file containing image details and bounding box coordinates
csv_path = '/home/user/Desktop/Mosquito_Detector.zip/filtered_train.csv'
data = pd.read_csv(csv_path)

# Define mosquito type names based on class indices
mosquito_type_names = {
    0: 'Aedes aegypti',
    1: 'Aedes albopictus',
    2: 'Anopheles gambiae',
    3: 'Culex pipiens',
    4: 'Culex quinquefasciatus',
    5: 'Mansonia',
    6: 'Psorophora'
}

# Initialize a list to store predictions
predictions = []

# Iterate through each image and make predictions
for index, row in data.iterrows():
    img_path = os.path.join(input_image_folder, row['img_fName'])
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0]
    mosquito_type_index = np.argmax(prediction)
    mosquito_type = mosquito_type_names.get(mosquito_type_index, 'Unknown')  # Use 'Unknown' if index not found
    
    predictions.append({
        'img_fName': row['img_fName'],
        'img_w': row['img_w'],
        'img_h': row['img_h'],
        'bbx_xtl': row['bbx_xtl'],
        'bbx_ytl': row['bbx_ytl'],
        'bbx_xbr': row['bbx_xbr'],
        'bbx_ybr': row['bbx_ybr'],
        'mosquito_type': mosquito_type
    })

# Create a DataFrame from the predictions
predictions_df = pd.DataFrame(predictions)

# Save the predictions to a CSV file
predictions_df.to_csv(output_csv_path, index=False)

print("Predictions saved to:", output_csv_path)
