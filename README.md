# Human face orientation classification and correction
In many machine learning use cases related to facial reconition, the orientaion of the image plays a crucial role. Images which are not straight, (for example, rotated right by 90 degrees) pose a 
challenge and might lead to bad performance.

The goal here is to build a model to properly detect the orientaion of the face into 4 categories (rotated by 0 degrees, 90 degrees, 180 degrees and 270 degrees).
The model will be wrapped around by a RESTful API capable of not only predicting the image corientation, but also correcting the orienation and returning back images which are always straight facing.

#### Descriptions of the scripts are as follows:
- **preparedata.py** :  Script to setup the dataset and organise it into proper folders according to a train-validation-test split.
- **model.py** : The training script where a CNN model is built using Keras API on Tensorflow backend and final model is saved for later predictions.
- **main.py** : Script implementing the RESTful API using Flask. It has two endpoints, one for just generating predictions of any image, and second to correct the orientation of the image.
- **test.py** : Some unit tests for the final model which is deployed on heroku.

#### Inastructions for use:
- Download the "Labelled Faces in the Wild" dataset and save it in the project directory. Extract the contents of the archive. Ensure the data is extracted into a folder called "lfw".
- Setup the python environment and install dependencies using *pip install -r requirements.txt*
- Run the preparedata.py script
- (Optional) Run the model.py script to train from scratch. Otherwise, the existing 'model.h5' keras model can be used.
- (Optional) Run the main.py script to run a local instance of the flask application. Once loaded, visit the URL to ensure "Hello World" is displayed.
- Run the test script or make an API call using cURL or any other utility to use the API.
