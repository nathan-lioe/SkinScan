## Welcome, to SkinScan

SkinScan is a user-friendly web application that empowers you to take control of your skin health. Simply snap a photo of any skin concern, and our advanced AI model will analyze the image and classify potential issues.
SkinScan can classify any of the following skin diseases.

- Actinic Keratosis (AKIEC)
- Basal Cell Carcinoma (BCC)
- Benign Keratosis-like Lesions (BKL)
- Dermatofibroma (DF)
- Melanoma (MEL)
- Melanocytic Nevi (NV)
- Vascular Lesions (VASC)
   

## How it works:

- Preprocesses the CSV files with image paths and diagnoses 
- Handle missing data
- Create train, validation and test sets 
- Derma Dataset class to load images and labels efficiently
- Pre-trained ResNet-50 model, a powerful Convolution neural network developed by Microsoft
- Fine tuned to match data
- Training_model function that handles the core training loop
- Adam optimizer
- Learning rate scheduler (ReduceLROnPlateau)
- Confusion matrix
- Classification report



## How to run 
- Git clone https://github.com/nathan-lioe/SkinScan.git
- cd to the directory where you have the project
- streamlit run app.py to run skinscan.

  
