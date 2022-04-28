

import pandas as pd 
import numpy as np 
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import h5py
import cv2 
from keras.layers import Flatten, Dense, Input,concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing import image 
import tensorflow as tf

# imports and creates 
vgg16 = keras.applications.VGG16(weights="imagenet", include_top=True, pooling="max", input_shape=(224, 224, 3))

basemodel = Model(inputs=vgg16.input, outputs=vgg16.get_layer("fc2").output)

def get_feature_vector(img):
    
    img1 = cv2.resize(img, (224, 224))
    feature_vector = basemodel.predict(img1.reshape(1, 224, 224, 3))
    
    return feature_vector


img1 = image.load_image("./img1.jpg")
img2 = image.load_image("./img2.jpg")
img3 = image.load_image("./img3.jpg")
f1 = get_feature_vector(img1)
f2 = get_feature_vector(img2)
f3 = get_feature_vector(img3)