import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd

# Information about disease
skin_disease_info = {     
    "Basal Cell Carcinoma": "Basal cell carcinoma is the most common type of skin cancer. It arises from basal cells in the epidermis due to prolonged UV exposure. BCC typically appears as shiny bumps, red patches, or scar-like areas on sun-exposed skin. It grows slowly and is highly treatable when detected early.",
    "Actinic Keratosis": "Actinic keratosis is a common precancerous skin condition caused by chronic exposure to ultraviolet (UV) radiation. It appears as small, rough, scaly patches on sun-exposed areas such as the face, scalp, and hands. If untreated, it can develop into squamous cell carcinoma.",
    "Benign Keratosis-like Lesions": "Benign keratosis-like lesions include non-cancerous growths such as seborrheic keratoses. These are harmless, often pigmented growths that can resemble warts or moles and usually do not require treatment unless they become bothersome.",
    "Dermatofibroma":"Dermatofibromas are benign, firm nodules that commonly appear on the skin of the legs or arms. They are caused by an overgrowth of fibrous tissue and are usually harmless but may be removed for cosmetic reasons or if symptomatic.", 
    "melanoma":"Melanoma is a dangerous form of skin cancer that develops in melanocytes, the cells responsible for skin pigmentation. It is less common than other skin cancers but more likely to spread if not treated early. Melanomas often appear as irregular, dark-colored moles or spots on the skin.", 
    "Melanocytic Nevi": "Melanocytic nevi are commonly known as moles. These are benign growths formed by clusters of melanocytes. While most moles are harmless, some may require monitoring for changes that could indicate melanoma.", 
    "Vascular Lesions":"Vascular lesions include conditions such as livedoid vasculopathy, which is characterized by painful sores and discoloration caused by blood clots in small blood vessels of the skin. These lesions may be linked to clotting disorders or autoimmune diseases."
   
}

# Configure the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = models.resnet50(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(skin_disease_info))  
model.load_state_dict(torch.load("model_skin.pth", map_location=device))
model.to(device).eval()


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class_labels = list(skin_disease_info.keys())

# Function to predict
def predict_with_threshold(model, image_tensor, categories, threshold=0.7):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()  

    max_prob = probabilities.max()
    predicted_index = probabilities.argmax()
    predicted_label = categories[predicted_index] if max_prob >= threshold else "None of the Above"

    return predicted_label, max_prob, probabilities

# UI
st.set_page_config(page_title="SkinScan", layout="centered")

st.title(" SkinScan - AI Dermatology")
st.info(
    "Simply snap a photo of any skin concern, and our AI model will analyze it. "
    "While SkinScan is not a medical tool, it offers early insights to encourage preventive care."
)

# Picture upload
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "png", "jpeg"])
camera_image = st.camera_input("📷 Or take a picture")


image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif camera_image:
    image = Image.open(camera_image).convert("RGB")

if image:
    st.image(image, caption=" Uploaded Image", width=400)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0).to(device)

    
    with st.spinner("🔬 Analyzing image..."):
        predicted_label, confidence, probabilities = predict_with_threshold(
            model, img_tensor, class_labels, threshold=0.7
        )

    # Results
    st.subheader(f" -- Prediction: **{predicted_label}** --")
    st.write(f"**Confidence:** `{confidence:.2%}`")

    # Show histogram
    st.subheader(" Class Probabilities")
    df = pd.DataFrame({"Condition": class_labels, "Probability": probabilities})
    df = df.sort_values("Probability", ascending=False)  
    st.bar_chart(df.set_index("Condition"))

    # More information about the predicted condition
    if predicted_label in skin_disease_info:
        st.subheader("📌 More About This Condition")
        st.write(skin_disease_info[predicted_label])