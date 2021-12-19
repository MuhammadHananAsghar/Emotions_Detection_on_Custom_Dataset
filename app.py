from flask import Flask, request, jsonify
from mtcnn import MTCNN
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import base64
import uuid
import cv2
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Emotions Classification API created by Muhammad Hanan Asghar</h1>"

@app.route("/api/v1", methods=['POST'])
def api():
    pic_data = request.form['URI']
    image = get_image_from_base64(pic_data)
    predicted_value, percentage_value = get_prediction(image)
    response = f"{predicted_value} - {percentage_value}"
    return jsonify(objt=response)

# Predicting Emotion
def get_prediction(imge):
    detector = MTCNN()
    class_names = ['angry', 'happy', 'sad', 'surprised']
    boxes = detector.detect_faces(imge)
    boxes = boxes[0]['box']
    boxes = [int(i) for i in boxes]
    x1, y1, width, height = boxes
    x2, y2 = x1 + width, y1 + height
    face_image = imge[y1:y2, x1:x2]
    face_image = cv2.resize(face_image, (110, 110))

    img_array = keras.preprocessing.image.img_to_array(face_image)
    img_array = tf.expand_dims(img_array, 0)
    model = load_model("emotions.h5")
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted = class_names[np.argmax(score)]
    percentage = round(100 * np.max(score), 2)
    return predicted, percentage

# Converting BASE64 to image
def get_image_from_base64(imageURI):
    unique_filename = str(uuid.uuid4())
    imgdata = base64.b64decode(imageURI)
    filename = f'{unique_filename}.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)
    image = plt.imread(filename)
    os.remove(filename)
    return image

if __name__ == "__main__":
    app.run(debug=True)