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
   

## How it works

Our Model Training Process:
- Preprocesses the CSV files with image paths and diagnoses 
- Handles missing data in the HAM10000 dataset
- Creates train, validation and test sets 
- DermaDataset class to load images and labels efficiently
- Pre-trained ResNet-50 model, a powerful Convolution neural network developed by Microsoft
- Fine tuned to match data
- Training_model function that handles the core training loop
- Adam optimizer, which helps the model learn by adjusting its parameters to minimize errors.
- Learning rate scheduler (ReduceLROnPlateau)
- Confusion matrix for model evaluation
- Classification report



## How to run 

### **Start by cloning the repo to your local machine**
````
Git clone https://github.com/nathan-lioe/SkinScan.git
````
### **Download all Requirements**
````
pip install -r requirements.txt
````


### **Run Streamlit**
````
streamlit run app.py

````````

Upload or take a picture.

  
