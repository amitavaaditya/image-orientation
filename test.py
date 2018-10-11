import unittest
import os
import shutil
import requests
from PIL import Image

# Randomly selected one image to be used for all testing purposes.
IMAGE_PATH = 'final/test/0/Deniz_Baykal_0001.jpg'

# REST API endpoint URLs
KERAS_REST_API_PREDICT_URL = "https://cryptic-atoll-59558.herokuapp.com/" \
                             "predict"
KERAS_REST_API_CORRECT_URL = "https://cryptic-atoll-59558.herokuapp.com/" \
                             "correct"


class ModelTest(unittest.TestCase):
    # Tasks to be performed prior to making API calls.
    # This includes setting up a temporary directory for test data
    # and generating rotated instances of the test image.
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

    # Tasks to be performed after unit testing is complete.
    # This includes removing the directory and files created for unit testing.
    def tearDown(self):
        temp_test_dir = 'temp_test'
        if os.path.exists(temp_test_dir):
            shutil.rmtree(temp_test_dir)

    # Test if the unrotated image (straight-orientation) is classified correctly.
    def test_rotation0_predict(self):
        with open('temp_test/0.jpg', "rb") as f:
            image = f.read()
        payload = {"image": image}
        r = requests.post(KERAS_REST_API_PREDICT_URL, files=payload).json()
        predicted_label = r['final_label']
        original_label = 0
        self.assertEqual(predicted_label, original_label)

    # Test if the image rotated by 90 degrees is classified correctly.
    def test_rotation90_predict(self):
        with open('temp_test/90.jpg', "rb") as f:
            image = f.read()
        payload = {"image": image}
        r = requests.post(KERAS_REST_API_PREDICT_URL, files=payload).json()
        predicted_label = r['final_label']
        original_label = 90
        self.assertEqual(predicted_label, original_label)

    # Test if the image rotated by 180 degrees is classified correctly.
    def test_rotation180_predict(self):
        with open('temp_test/180.jpg', "rb") as f:
            image = f.read()
        payload = {"image": image}
        r = requests.post(KERAS_REST_API_PREDICT_URL, files=payload).json()
        predicted_label = r['final_label']
        original_label = 180
        self.assertEqual(predicted_label, original_label)

    # Test if the image rotated by 270 degrees is classified correctly.
    def test_rotation270_predict(self):
        with open('temp_test/270.jpg', "rb") as f:
            image = f.read()
        payload = {"image": image}
        r = requests.post(KERAS_REST_API_PREDICT_URL, files=payload).json()
        predicted_label = r['final_label']
        original_label = 270
        self.assertEqual(predicted_label, original_label)

    # Test if the unrotated image (straight-orientation) remains unrotated after
    # correction.
    # For this, first a prediction is made regarding the degree of rotation (in this
    # case 0 degrees) and then corrected accordingly.
    # Finally after correction it is again verified if it is straight.
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

    # Test if the image rotated by 90 degrees is properly corrected.
    # For this, first a prediction is made regarding the degree of rotation (in this
    # case 90 degrees) and then corrected accordingly.
    # Finally after correction it is again verified if it is straight.
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

    # Test if the image rotated by 90 degrees is properly corrected.
    # For this, first a prediction is made regarding the degree of rotation (in this
    # case 180 degrees) and then corrected accordingly.
    # Finally after correction it is again verified if it is straight.
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

    # Test if the image rotated by 90 degrees is properly corrected.
    # For this, first a prediction is made regarding the degree of rotation (in this
    # case 270 degrees) and then corrected accordingly.
    # Finally after correction it is again verified if it is straight.
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
