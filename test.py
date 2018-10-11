import unittest
import os
import shutil
import requests
from PIL import Image


IMAGE_PATH = 'final/test/0/Deniz_Baykal_0001.jpg'
KERAS_REST_API_PREDICT_URL = "https://cryptic-atoll-59558.herokuapp.com/predict"
KERAS_REST_API_CORRECT_URL = "https://cryptic-atoll-59558.herokuapp.com/correct"


class ModelTest(unittest.TestCase):

    def setUp(self):
        temp_test_dir = 'temp_test'
        if os.path.exists(temp_test_dir):
            shutil.rmtree(temp_test_dir)
        os.mkdir(temp_test_dir)
        image = Image.open(IMAGE_PATH)
        for rotation in (0, 90, 180, 270):
            image.rotate(rotation).save(
                os.path.join(temp_test_dir, '{}.jpg'.format(rotation))
            )

    def tearDown(self):
        temp_test_dir = 'temp_test'
        if os.path.exists(temp_test_dir):
            shutil.rmtree(temp_test_dir)

    def test_rotation0_predict(self):
        with open('temp_test/0.jpg', "rb") as f:
            image = f.read()
        payload = {"image": image}
        r = requests.post(KERAS_REST_API_PREDICT_URL, files=payload).json()
        predicted_label = r['final_label']
        original_label = 0
        self.assertEqual(predicted_label, original_label)

    def test_rotation90_predict(self):
        with open('temp_test/90.jpg', "rb") as f:
            image = f.read()
        payload = {"image": image}
        r = requests.post(KERAS_REST_API_PREDICT_URL, files=payload).json()
        predicted_label = r['final_label']
        original_label = 90
        self.assertEqual(predicted_label, original_label)

    def test_rotation180_predict(self):
        with open('temp_test/180.jpg', "rb") as f:
            image = f.read()
        payload = {"image": image}
        r = requests.post(KERAS_REST_API_PREDICT_URL, files=payload).json()
        predicted_label = r['final_label']
        original_label = 180
        self.assertEqual(predicted_label, original_label)

    def test_rotation270_predict(self):
        with open('temp_test/270.jpg', "rb") as f:
            image = f.read()
        payload = {"image": image}
        r = requests.post(KERAS_REST_API_PREDICT_URL, files=payload).json()
        predicted_label = r['final_label']
        original_label = 270
        self.assertEqual(predicted_label, original_label)

    def test_rotation0_correct(self):
        with open('temp_test/0.jpg', "rb") as f:
            image = f.read()
        payload = {"image": image}
        r = requests.post(KERAS_REST_API_CORRECT_URL, files=payload)
        r.raw.decode_content = True
        corrected_image = r.content
        payload = {"image": corrected_image}
        r = requests.post(KERAS_REST_API_PREDICT_URL, files=payload).json()
        predicted_label = r['final_label']
        self.assertEqual(predicted_label, 0)

    def test_rotation90_correct(self):
        with open('temp_test/90.jpg', "rb") as f:
            image = f.read()
        payload = {"image": image}
        r = requests.post(KERAS_REST_API_CORRECT_URL, files=payload)
        r.raw.decode_content = True
        corrected_image = r.content
        payload = {"image": corrected_image}
        r = requests.post(KERAS_REST_API_PREDICT_URL, files=payload).json()
        predicted_label = r['final_label']
        self.assertEqual(predicted_label, 0)

    def test_rotation180_correct(self):
        with open('temp_test/180.jpg', "rb") as f:
            image = f.read()
        payload = {"image": image}
        r = requests.post(KERAS_REST_API_CORRECT_URL, files=payload)
        r.raw.decode_content = True
        corrected_image = r.content
        payload = {"image": corrected_image}
        r = requests.post(KERAS_REST_API_PREDICT_URL, files=payload).json()
        predicted_label = r['final_label']
        self.assertEqual(predicted_label, 0)

    def test_rotation270_correct(self):
        with open('temp_test/270.jpg', "rb") as f:
            image = f.read()
        payload = {"image": image}
        r = requests.post(KERAS_REST_API_CORRECT_URL, files=payload)
        r.raw.decode_content = True
        corrected_image = r.content
        payload = {"image": corrected_image}
        r = requests.post(KERAS_REST_API_PREDICT_URL, files=payload).json()
        predicted_label = r['final_label']
        self.assertEqual(predicted_label, 0)
