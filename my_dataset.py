import random
import os
from tqdm import tqdm
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from PIL import UnidentifiedImageError

def randomize_dataset(images, labels):
    if len(images) != len(labels):
        raise ValueError("Arrays must have the same length")
    
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)
    return np.array(images), np.array(labels)

def add_arrays(arr1, arr2):
    return np.concatenate((np.array(arr1), np.array(arr2))).tolist()

def import_images(directory, lbl):
    images = []
    labels = []
    for img_name in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)
        if not img_path.endswith('.jpg'):
            continue
        try:
            img = load_img(img_path, target_size=(32, 32))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(lbl)
        except UnidentifiedImageError:
            print(f"Cannot identify image file {img_path}. Skipping.")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Skipping.")
    return np.array(images), np.array(labels)

def import_dataset(directory = ''):
    print('Importing dogs images...')
    h_images, h_labels = import_images('./dataset/Dog', 1)
    print('Importing cats images...')
    c_images, c_labels = import_images('./dataset/Cat', 0)

    print('Adding datasets...')
    images = add_arrays(h_images, c_images)
    labels = add_arrays(h_labels, c_labels)

    print('Randomizing dataset...')
    images, labels = randomize_dataset(images, labels)

    print('Generating Train dataset...')
    train_images = images[:(int(len(images)*0.8))]
    train_labels = labels[:(int(len(labels)*0.8))]

    print('Generating Test dataset...')
    test_images = images[(len(images)-len(train_images)):]
    test_labels = labels[(len(labels)-len(train_labels)):]

    return ((train_images, train_labels), (test_images, test_labels))
