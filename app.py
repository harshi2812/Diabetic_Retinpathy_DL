from flask import Flask, render_template, request
import os
from resnet101 import predict as resnet_predict
from vgg16 import predict as vgg16_predict

app = Flask(__name__)

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve the uploaded file
        file = request.files['file']
        if not file:
            return "No file uploaded", 400

        # Save the uploaded file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Get the selected model name
        model_name = request.form['model_name']

        # Call the appropriate model's predict function
        if model_name == 'resnet101':
            predicted_class, confidence = resnet_predict(file_path)
        elif model_name == 'vgg16':
            predicted_class, confidence = vgg16_predict(file_path)
        else:
            return "Invalid model selection", 400

        # Remove the temporary file
        os.remove(file_path)

        # Render the result page
        return render_template('result.html', predicted_class=predicted_class, confidence=confidence)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)








