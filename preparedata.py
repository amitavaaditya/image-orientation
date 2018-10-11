import os
import shutil
from PIL import Image
import random


# Check for dataset availibility
def check_dataset_availability():
    if not os.path.exists('data'):
        raise FileNotFoundError('Unable to locate dataset folder. Please '
                                'download the dataset and extract to a folder'
                                '"lfw" in project home directory')


# Prepare directory tree for moving data into.
def prepare_directories(directories):
    if os.path.exists(directories['final_dataset']):
        shutil.rmtree(directories['final_dataset'])
    os.mkdir(directories['final_dataset'])
    for subset in ('train', 'validation', 'test'):
        os.mkdir(directories[subset])
        for orientation in ('0', '90', '180', '270'):
            os.mkdir(directories['{}_{}'.format(subset,
                                                orientation)])


# Rotate images by a degree
def rotate_image(image, rotation):
    rotated_image = image.rotate(rotation)
    return rotated_image


# Generated rotated instances of the images and fill them into their
# corresponding directories.
def rotate_and_save_images(directories, subdirectories, subset, rotation):
    for subdirectory in subdirectories:
        src = os.path.join(directories['original_dataset'],
                           subdirectory,
                           '{}_0001.jpg'.format(subdirectory))
        image = Image.open(src)
        rotated_image = rotate_image(image, rotation)
        rotated_image.save(os.path.join(directories['{}_{}'.format(subset,
                                                                   rotation)],
                                        '{}_0001.jpg'.format(subdirectory)))

# Split all the images into 1000 validation, 1000 test and remaining training
# sets for each of the rotation degrees
def train_validation_test_split(directories, subdirectories):
    for rotation in (0, 90, 180, 270):
        random.seed(rotation)
        random.shuffle(subdirectories)
        rotate_and_save_images(directories, subdirectories[:1000],
                               'validation', rotation)
        rotate_and_save_images(directories, subdirectories[1000:2000],
                               'test', rotation)
        rotate_and_save_images(directories, subdirectories[2000:],
                               'train', rotation)

# Root method to prepare the dataset with rotated images from lfw dataset.
def prepare_custom_dataset():
    directories = dict()
    directories['original_dataset'] = os.path.join('.', 'lfw')
    directories['final_dataset'] = os.path.join('.', 'final')
    for subset in ('train', 'validation', 'test'):
        directories[subset] = os.path.join(directories['final_dataset'],
                                           subset)
        for orientation in ('0', '90', '180', '270'):
            directories['{}_{}'.format(subset, orientation)] = os.path.join(
                directories[subset], orientation
            )
    prepare_directories(directories)
    subdirectories = os.listdir(directories['original_dataset'])
    train_validation_test_split(directories, subdirectories)


if __name__ == '__main__':
    check_dataset_availability()
    prepare_custom_dataset()
