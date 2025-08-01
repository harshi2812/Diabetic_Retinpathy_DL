from tensorflow.keras.models import load_model 
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
import os

VGG_PATH = os.path.abspath("model/saved_model/vgg16_model.h5")
assert os.path.exists(VGG_PATH), f"Model file not found: {VGG_PATH}"


try:
    vgg16_model = load_model(VGG_PATH)
    print("Models loaded successfully!")
except Exception as e:
    vgg16_model = None
    print(f"Failed to load ResNet101 model: {e}")



# Prediction function
def predict(image_path):
    # Preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict using the model
    predictions = vgg16_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]

    # Map predicted class to label
    class_labels = {
        0: 'No Diabetic Retinopathy',
        1: 'Mild Diabetic Retinopathy',
        2: 'Moderate Diabetic Retinopathy',
        3: 'Severe Diabetic Retinopathy',
        4: 'Proliferative Diabetic Retinopathy'
    }
    return class_labels[predicted_class], confidence
