from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import flask
from PIL import Image
import io


app = flask.Flask(__name__)
model = None


def prepare_image(image, target):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.
    return image


def make_prediction(image):
    image_tensor = prepare_image(image, target=(150, 150))
    predictions = model.predict(image_tensor)
    results, final_label = decode_prediction(predictions[0])
    return results, final_label


def decode_prediction(predictions):
    labels = (0, 180, 270, 90)
    results = list(zip(labels, predictions))
    final_label = labels[np.argmax(predictions)]
    return results, final_label


@app.route("/predict", methods=['POST'])
def predict():
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


@app.route("/correct", methods=['POST'])
def correct():
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


def rotate_image(image, rotation):
    return image.rotate(-rotation)


def model_load():
    global model
    model_src = 'model.h5'
    model = load_model(model_src)
    model._make_predict_function()


if __name__ == '__main__':
    model_load()
    app.run()