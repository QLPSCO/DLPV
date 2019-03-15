#from tensorflow.python.keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras import applications
from keras.layers import Dense, GlobalAveragePooling2D
from keras import optimizers
import tensorflow as tf
from gradient_checkpointing.memory_saving_gradients import gradients, gradients_memory
from tensorflow.python.keras._impl.keras import backend as K
from dlpv import *

import subprocess
import os
import theano

# monkey patch tf.gradients to point to custom gradient checkpointing
tf.__dict__["gradients"] = gradients_memory
K.__dict__["gradients"] = gradients_memory

# Configure queue and threads
q = tf.FIFOQueue(capacity=100, dtypes=tf.float32)

# Configure theano and tensorflow
"""
os.environ["OMP_NUM_THEADS"] = "4"
theano.config.openmp = True
"""
config = tf.ConfigProto(
    device_count = {'GPU': 1}
)
sess = tf.Session(config=config)

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 5
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#inception_base = applications.ResNet50(weights='imagenet', include_top=False)
#inception_base = applications.VGG16(weights="imagenet", include_top=False)

dlpv_trace(optimization="multithreaded_q")

for layer in inception_base.layers:
    layer.trainable = False

# add a global spatial average pooling layer
x = inception_base.output
x = GlobalAveragePooling2D()(x)
# add a fully-connected layer
x = Dense(512, activation='relu')(x)
# and a fully connected output/classification layer
predictions = Dense(1, activation='sigmoid')(x)
# create the full network so we can train on it
custom_resnet_model = Model(inputs=inception_base.input, outputs=predictions)

custom_resnet_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

dlpv

custom_resnet_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


