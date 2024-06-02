from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

model = load_model('DSAstudent-best.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image file in request'}), 400

    file = request.files['file']
    
    try:
        img = Image.open(io.BytesIO(file.read()))
    except IOError:
        return jsonify({'error': 'Invalid image file'}), 400
    
   
    img = img.resize((224, 224))  
    img = np.array(img) / 255.0   
    img = np.expand_dims(img, axis=0)  
    
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return jsonify({'prediction': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
