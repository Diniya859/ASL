# coding: utf-8

# In[ ]:
import os

import tensorflow as tf

import keras
from keras.engine.saving import load_model
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
num_classes = 20
batch_size = 40
epochs = 20
#------------------------------

import os, cv2, keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.engine.saving import load_model
# manipulate with numpy,load with panda
import numpy as np
# import pandas as pd

# data visualization
import cv2
import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns

# get_ipython().run_line_magic('matplotlib', 'inline')

def read_dataset1(path):
    data_list = []
    label_list = []

    file_path = os.path.join(path)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(img, (48, 48), interpolation=cv2.INTER_CUBIC)
    data_list.append(res)

    return (np.asarray(data_list, dtype=np.float32))
from keras import backend as K
import os
def predictcnn(fn):
    dataset=read_dataset1(fn)
    (mnist_row, mnist_col, mnist_color) = 48, 48, 1

    dataset = dataset.reshape(dataset.shape[0], mnist_row, mnist_col, mnist_color)
    mo = load_model(r"C:\Users\DELL\PycharmProjects\SLR\word_model.h5")
    dataset /= 255
    # predict probabilities for test set

    yhat_classes = mo.predict_classes(dataset, verbose=0)
    print(yhat_classes)
    result=yhat_classes[0]
    print("RESULT",result)
    res=int(yhat_classes.tolist()[0])        #numpy ndarray type convert into list type
    K.clear_session()                        #session clear(avoid simultaneously running error)
    message_list=["Are you coming.", "Dislike.", "Food", "Friends" ,"Good.","Hai" , "Help me" , "Love you", "Need", "Nice to meet you", "No", "Open.","Please" ,"Reach Safely.", "Thankyou.", "Warning.", "Yes"]
    result=message_list[res]
    # os.remove(fn)
    return result
    # return result





#
#     print(yhat_classes)
# predictcnn(r"C:\Users\DELL\PycharmProjects\SLR\Myapp\static\new_data\no\24.jpg")