import time
import tensorflow as tf
from tensorflow.keras import applications, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import os.path
import glob
import shutil
from joblib import dump, load

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

main_dir = ""
index_data_dir = main_dir + "dataset/index/"
input_dir = main_dir + "input/"
output_dir = main_dir + "output/"

number_of_same_landmark = 10
image_size = (256, 256)


def save_knn_model(model, images_path):
    dump(model, main_dir + 'models/knn_model.joblib')
    dump(images_path, main_dir + 'models/images_path.joblib')


def load_knn_model():
    return load(main_dir + 'models/knn_model.joblib')


def load_images_path():
    return load(main_dir + 'models/images_path.joblib')


def clear_output():
    files = glob.glob(os.path.join(output_dir, "*.jpg"))
    for f in files:
        os.remove(f)


def get_features_model():
    base_model = load_model(main_dir + "models/model.h5")
    features_layer = Flatten()(base_model.get_layer('block5_pool').output)
    return Model(inputs=base_model.input, outputs=features_layer)


def create_knn_model():
    model = get_features_model()

    index_generator = ImageDataGenerator().flow_from_directory(index_data_dir,
                                                               target_size=image_size,
                                                               batch_size=1,
                                                               class_mode='categorical',
                                                               shuffle=False)

    images_path = index_generator.filenames
    process_images = []
    labels = []

    print("Loading index images...")

    for i in range(index_generator.n):
        (img, lbl) = index_generator.next()
        process_images.append(np.reshape(model.predict_on_batch(img), -1))
        labels.append(np.argmax(lbl))

    print("Loading index images to KNN model...")
    knn_model = KNeighborsClassifier(n_neighbors=number_of_same_landmark)
    knn_model.fit(process_images, labels)
    save_knn_model(knn_model, images_path)
    print("Done!")


def find_landmarks():
    model = get_features_model()
    input_img = ImageDataGenerator().flow_from_directory(input_dir,
                                                         target_size=image_size,
                                                         batch_size=1,
                                                         class_mode=None,
                                                         shuffle=False).next()
    input_features = model.predict(input_img)

    print("Loading KNN model...")
    knn_model = load_knn_model()

    print("Calculate landmarks output...")
    positions = knn_model.kneighbors(input_features, return_distance=False)[0]

    print("Loading images path...")
    images_path = load_images_path()

    clear_output()
    distance_position = 0

    for index in positions:
        distance_position += 1
        image_path = index_data_dir + images_path[index]
        shutil.copy(image_path, output_dir + str(distance_position) + ".jpg")

    print("Done!")


if __name__ == '__main__':
    create_knn_model()
