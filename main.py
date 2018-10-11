from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import flask
from PIL import Image
import io
# import tensorflow as tf

app = flask.Flask(__name__)
model = None
graph = None


def prepare_image(image, target):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.
    return image


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
            image_tensor = prepare_image(image, target=(150, 150))
            predictions = model.predict(image_tensor)
            data["predictions"] = []
            results, final_label = decode_prediction(predictions[0])
            for label, probability in results:
                r = {'label': label, 'probability': float(probability)}
                data['predictions'].append(r)
            rotation = int(final_label)
            rotated_image = rotate_image(image, rotation)
            data["success"] = True
    return flask.jsonify(data)


def rotate_image(image, rotation):
    return image.rotate(-rotation)


def model_load():
    global model
    # global graph
    model_src = 'model.h5'
    model = load_model(model_src)
    # graph = tf.get_default_graph()
    model._make_predict_function()


if __name__ == '__main__':
    model_load()
    app.run()