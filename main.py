from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import flask
from PIL import Image
import io
import os


# Flask app name
app = flask.Flask(__name__)
# Placeholder for the pre-trained model
model = None


# Method responsible to take input an image in PIL format and
# prepare it for predictions
def prepare_image(image, target):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.
    return image


# Method responsible for generating the actual prediction
# It takes as input, an image in PIL format
# And returns the respective class probabilities and final prediction
def make_prediction(image):
    image_tensor = prepare_image(image, target=(150, 150))
    predictions = model.predict(image_tensor)
    results, final_label = decode_prediction(predictions[0])
    return results, final_label


# This methods acts as a mapper which maps the labels predicted by
# the classifier (0, 1, 2, 3) and maps to its actual labels (0, 180, 270, 360)
def decode_prediction(predictions):
    labels = (0, 180, 270, 90)
    results = list(zip(labels, predictions))
    final_label = labels[np.argmax(predictions)]
    return results, final_label


# Sample method to check if the flask app is working after deployment to
# heroku
@app.route("/")
def hello():
    return 'Hello World!'


# RESTful endpoint for making predictions.
# It returns a JSON response of the class probabilities and final prediction
# It expects an image with the keyword "image" in the POST request, and
# returns a JSON object with "predictions" having class probabilities and
# "final_prediction" having the final prediction made by the model.
@app.route("/predict", methods=['POST'])
def predict():
    if not model:
        model_load()
    data = {"success": False}
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            data["predictions"] = []
            results, final_label = make_prediction(image)
            for label, probability in results:
                r = {'label': label, 'probability': float(probability)}
                data['predictions'].append(r)
                data['final_label'] = final_label
            data["success"] = True
    return flask.jsonify(data)


# RESTful endpoint for making corrections in image orientation.
# It expects an image with the keyword "image" in the POST request,
# and performs the following steps in sequence:
# First, it makes the image ready for predictions
# Second, it is fed to the classifier to predict its current orientation
# Third, it performs a rotate operation to make the images straight
# Fourth, returns the rotated image in the response
@app.route("/correct", methods=['POST'])
def correct():
    if not model:
        model_load()
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            _, final_label = make_prediction(image)
            rotation = int(final_label)
            rotated_image = rotate_image(image, rotation)
            rotated_image.save('converted_image.jpeg', 'JPEG')
    return flask.send_file('converted_image.jpeg',
                           mimetype='image/jpeg',
                           as_attachment=True)


# Method to rotate the images to make them straight
def rotate_image(image, rotation):
    return image.rotate(-rotation)


# Method to load the pre-trained model
def model_load():
    global model
    model_src = 'model.h5'
    model = load_model(model_src)
    model._make_predict_function()


# For running locally
if __name__ == '__main__':
    model_load()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='localhost', port=port)
