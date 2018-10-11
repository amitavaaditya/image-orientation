from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np


def load_image(img_src):
    image = load_img(img_src, target_size=(150, 150))
    img_tensor = img_to_array(image)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    return img_tensor

def predict(model, image):
    classes = {
        0: '0',
        1: '180',
        2: '270',
        3: '90'
    }
    prediction = model.predict(image)
    print(prediction)
    predicted_class = classes[np.argmax(prediction)]
    return predicted_class


if __name__ == '__main__':
    model_src = 'model.h5'
    model = load_model(model_src)
    img_src = 'final/test/180/Aaron_Peirsol_0001.jpg'
    image = load_image(img_src)
    print(predict(model, image))
