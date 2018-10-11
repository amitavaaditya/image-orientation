import os
import shutil
from PIL import Image
import random


def is_dataset_available():
    if not os.path.exists('data'):
        raise FileNotFoundError('Unable to locate dataset folder. Please '
                                'download the dataset and extract to a folder '
                                '"data" in project home directory')


def prepare_directories(directories):
    if os.path.exists(directories['final_dataset']):
        shutil.rmtree(directories['final_dataset'])
    os.mkdir(directories['final_dataset'])
    for subset in ('train', 'validation', 'test'):
        os.mkdir(directories[subset])
        for orientation in ('0', '90', '180', '270'):
            os.mkdir(directories['{}_{}'.format(subset,
                                                orientation)])


def rotate_image(image, rotation):
    rotated_image = image.rotate(rotation)
    return rotated_image


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
    prepare_custom_dataset()
