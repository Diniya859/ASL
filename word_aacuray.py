# # USAGE
# # python train_mask_detector.py --dataset dataset
#
# # import the necessary packages
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.layers import AveragePooling2D
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Input
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.utils import to_categorical
# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from imutils import paths
# import matplotlib.pyplot as plt
# import numpy as np
# import argparse
# import os
#
# # construct the argument parser and parse the arguments
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-d", "--dataset", required=True,
# # 	help="path to input dataset")
# # ap.add_argument("-p", "--plot", type=str, default="plot.png",
# # 	help="path to output loss/accuracy plot")
# # ap.add_argument("-m", "--model", type=str,
# # 	default="mask_detector.model",
# # 	help="path to output face mask detector model")
# # args = vars(ap.parse_args())
#
# # initialize the initial learning rate, number of epochs to train for,
# # and batch size
# INIT_LR = 1e-4
# EPOCHS = 20
# BS = 32
#
# # grab the list of images in our dataset directory, then initialize
# # the list of data (i.e., images) and class images
# print("[INFO] loading images...")
# imagePaths = list(paths.list_images(r"C:\Users\DELL\PycharmProjects\SLR\Myapp\static\dataSet\dataSet\trainingData\\"))
# data = []
# labels = []
#
# # loop over the image paths
# for imagePath in imagePaths:
# 	# extract the class label from the filename
# 	label = imagePath.split(os.path.sep)[-2]
#
# 	# load the input image (224x224) and preprocess it
# 	print(imagePath)
# 	image = load_img(imagePath, target_size=(224, 224))
# 	image = img_to_array(image)
# 	image = preprocess_input(image)
#
# 	# update the data and labels lists, respectively
# 	data.append(image)
# 	labels.append(label)
#
# # convert the data and labels to NumPy arrays
# data = np.array(data, dtype="float32")
# labels = np.array(labels)
#
# # perform one-hot encoding on the labels
# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)
# labels = to_categorical(labels)
#
# # partition the data into training and testing splits using 75% of
# # the data for training and the remaining 25% for testing
# (trainX, testX, trainY, testY) = train_test_split(data, labels,
# 	test_size=0.20, stratify=labels, random_state=42)
#
# # construct the training image generator for data augmentation
# aug = ImageDataGenerator(
# 	rotation_range=20,
# 	zoom_range=0.15,
# 	width_shift_range=0.2,
# 	height_shift_range=0.2,
# 	shear_range=0.15,
# 	horizontal_flip=True,
# 	fill_mode="nearest")
#
# # load the MobileNetV2 network, ensuring the head FC layer sets are
# # left off
# baseModel = MobileNetV2(weights="imagenet", include_top=False,
# 	input_tensor=Input(shape=(224, 224, 3)))
#
# # construct the head of the model that will be placed on top of the
# # the base model
# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(128, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2, activation="softmax")(headModel)
#
# # place the head FC model on top of the base model (this will become
# # the actual model we will train)
# model = Model(inputs=baseModel.input, outputs=headModel)
#
# # loop over all layers in the base model and freeze them so they will
# # *not* be updated during the first training process
# for layer in baseModel.layers:
# 	layer.trainable = False
#
# # compile our model
# print("[INFO] compiling model...")
# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# model.compile(loss="binary_crossentropy", optimizer=opt,
# 	metrics=["accuracy"])
#
# # train the head of the network
# print("[INFO] training head...")
# H = model.fit(
# 	aug.flow(trainX, trainY, batch_size=BS),
# 	steps_per_epoch=len(trainX) // BS,
# 	validation_data=(testX, testY),
# 	validation_steps=len(testX) // BS,
# 	epochs=EPOCHS)
#
# # make predictions on the testing set
# print("[INFO] evaluating network...")
# predIdxs = model.predict(testX, batch_size=BS)
#
# # for each image in the testing set we need to find the index of the
# # label with corresponding largest predicted probability
# predIdxs = np.argmax(predIdxs, axis=1)
#
# # show a nicely formatted classification report
# print(classification_report(testY.argmax(axis=1), predIdxs,
# 	target_names=lb.classes_))
#
# # serialize the model to disk
# print("[INFO] saving mask detector model...")
# model.save("animal_detector.model", save_format="h5")
#
# # plot the training loss and accuracy
# N = EPOCHS
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig("plot.png")




import os

import tensorflow as tf

import keras
# from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np

#------------------------------
# sess = tf.Session()
# keras.backend.set_session(sess)
#------------------------------
#variables
num_classes =17
batch_size = 40
epochs = 10
#------------------------------

import os, cv2, keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
# from keras.engine.saving import load_model
# from keras.engine.saving import load_model
from tensorflow.keras.models import load_model
# manipulate with numpy,load with panda

import numpy as np
# import pandas as pd

# data visualization
import cv2
import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns

# get_ipython().run_line_magic('matplotlib', 'inline')


# Data Import
def read_dataset(path):
    data_list = []
    label_list = []
    i=0
    my_list = os.listdir(r'C:\Users\DELL\PycharmProjects\SLR\Myapp\static\new_data\\')
    for pa in my_list:

        print(pa,"==================",i)
        for root, dirs, files in os.walk(r'C:\Users\DELL\PycharmProjects\SLR\Myapp\static\new_data\\' + pa):

         for f in files:
            file_path = os.path.join(r'C:\Users\DELL\PycharmProjects\SLR\Myapp\static\new_data\\'+pa, f)

            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
            data_list.append(res)

            label = i
            label_list.append(label)
        i=i+1
    return (np.asarray(data_list, dtype=np.float32), np.asarray(label_list))

def read_dataset1(path):
    data_list = []
    label_list = []

    file_path = os.path.join(path)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
    data_list.append(res)
    # label = dirPath.split('/')[-1]

            # label_list.remove("./training")
    return (np.asarray(data_list, dtype=np.float32))

from sklearn.model_selection import train_test_split
# load dataset
x_dataset, y_dataset = read_dataset(r"C:\Users\DELL\PycharmProjects\SLR\Myapp\static\new_data")
X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=0)

y_train1=[]
for i in y_train:
    emotion = keras.utils.to_categorical(i, num_classes)
    print(i,emotion)
    y_train1.append(emotion)

y_train=y_train1
x_train = np.array(X_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(X_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255  # normalize inputs between [0, 1]
x_test /= 255
print("x_train.shape",x_train.shape)
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# ------------------------------
# construct CNN structure

model = Sequential()

# 1st convolution layer
model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

# 2nd convolution layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

# 3rd convolution layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Flatten())

# fully connected neural networks
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))
# ------------------------------
# batch process

gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

# ------------------------------

model.compile(loss='categorical_crossentropy'
              , optimizer=keras.optimizers.Adam()
              , metrics=['accuracy']
              )

# ------------------------------

# Train the model and store the training history

if not os.path.exists("C:\\Users\\DELL\\PycharmProjects\\SLR\\modelnew211.h5"):
    history = model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs)
    print(history)
    model.save("C:\\Users\\DELL\\PycharmProjects\\SLR\\modelnew211.h5")  # train for randomly selected one
    plt.plot(history.history['accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train'], loc='upper left')
    plt.show()

else:
    model = load_model("C:\\Users\\DELL\\PycharmProjects\\SLR\\Myapp\\static\\modelnew211.h5")  # load weights

# Plot accuracy graph

from sklearn.metrics import confusion_matrix
yp=model.predict_classes(x_test,verbose=0)
cf=confusion_matrix(y_test,yp)
print(cf)



plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train'], loc='upper left')
plt.show()