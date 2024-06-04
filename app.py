from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the trained model
model = load_model('Artifacts\model.h5')

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    # Resize the image to match the model input shape
    img = cv2.resize(img, (64, 64))
    # Convert the image to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize pixel values
    img = img / 255.0
    # Expand the dimensions to match the model input  shape
    img = np.expand_dims(img, axis=0)
    return img

def predict_leukemia(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    # Perform prediction
    prediction = model.predict(img)
    # Get the predicted class label
    predicted_class = np.argmax(prediction)
    # Define class names
    class_names = ["Leukemia", "Myeloma", "Normal"]
    # Get the predicted class name
    predicted_class_name = class_names[predicted_class]
    return predicted_class_name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':        
        file = request.files['file']
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)  # Save the file to the uploads folder
            # Perform prediction
            predicted_class = predict_leukemia(file_path)
            
            return render_template('index.html', filename=filename, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
