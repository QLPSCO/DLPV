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
import os

# Uncomment when using Theano and CPU in order to make it use all CPU threads
"""
import theano
os.environ["OMP_NUM_THEADS"] = "4"
theano.config.openmp = True
"""

# Uncomment when using Tensorflow and GPU in order to make it use the GPU
"""
config = tf.ConfigProto(
    device_count = {'GPU': 1}
)
sess = tf.Session(config=config)

"""

# Preprocessing
img_width, img_height = 150, 150
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
epochs = 25
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Uncomment first line if using resnet base model. uncomment second line if using vgg base model
#base = applications.ResNet50(weights='imagenet', include_top=False)
base = applications.VGG16(weights="imagenet", include_top=False)

# freeze the weights in the bottom layers
for layer in base.layers:
    layer.trainable = False

# construct network
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # sigmoid because binary classification; softmax doesn't work!
custom_model = Model(inputs=base.input, outputs=predictions)
custom_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# pre data for input
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
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

custom_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

custom_model.save_weights('output_weights.h5')
