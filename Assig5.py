#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Flatten, Dropout, Lambda
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


# In[3]:


train_df = pd.read_csv('fashion-mnist_train.csv')
test_df = pd.read_csv('fashion-mnist_test.csv')

train_df.head()


# In[4]:


# split the training and testing data into X (image) and Y (label) arrays

train_data = np.array(train_df, dtype='float32')
test_data = np.array(test_df, dtype='float32')

x_train = train_data[:, 1:] / 255
y_train = train_data[:, 0]

x_test = test_data[:, 1:] / 255
y_test = test_data[:, 0]


# In[5]:


x_train.shape


# In[6]:


x_test.shape


# In[7]:


# split the training data into train and validate arrays (will be used later)

x_train, x_validate, y_train, y_validate = train_test_split(
    x_train, y_train, test_size=0.2, random_state=10,
)


# In[8]:


x_train.shape


# In[9]:


x_validate.shape


# In[10]:


class_labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]


# In[11]:


image = x_train[50, :].reshape((28, 28))

plt.imshow(image)
plt.show()


# In[12]:


class_labels[int(y_train[50])]


# CNN Model
# - Define
# - Compile
# - Fit

# In[13]:


batch_size = 512
im_shape = (28, 28, 1)

x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_test = x_test.reshape(x_test.shape[0], *im_shape)
x_validate = x_validate.reshape(x_validate.shape[0], *im_shape)

print('x_train shape: {}'.format(x_train.shape))
print('x_test shape: {}'.format(x_test.shape))
print('x_validate shape: {}'.format(x_validate.shape))


# In[14]:


# Model specifications

cnn_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=im_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])


# In[15]:


# compile model

opt = Adam(lr=0.001)
cnn_model.compile(loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)


# In[16]:


# training the model

cnn_model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=10,
    verbose=1,
    validation_data=(x_validate, y_validate)
)


# In[17]:


# testing the model on test data

score = cnn_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# # A deeper CNN

# In[18]:


cnn2 = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=im_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),

    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    Dropout(0.2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])


# In[19]:


cnn2.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[20]:


cnn2.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(x_validate, y_validate))


# In[21]:


cnn2.optimizer.lr = 0.0001

cnn2.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(x_validate, y_validate))


# In[22]:


score = cnn2.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# # Model with 4 Conv Layers and Batch Norm

# In[23]:


mean_px = x_train.mean().astype(np.float32)
std_px = x_train.std().astype(np.float32)
def norm_input(x): return (x-mean_px)/std_px


# In[24]:


cnn3 = Sequential([
    Lambda(norm_input, input_shape=im_shape),
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=im_shape),
    BatchNormalization(),

    Conv2D(32, kernel_size=(3, 3), activation='relu'),    
    BatchNormalization(),
    Dropout(0.25),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),    
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),    
    BatchNormalization(),
    Dropout(0.25),

    Flatten(),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])


# In[25]:


cnn3.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# In[26]:


cnn3.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(x_validate, y_validate))


# In[27]:


cnn3.optimizer.lr = 0.0001


# In[28]:


cnn3.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(x_validate, y_validate))


# In[29]:


score = cnn3.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# With Data Augmentation

# In[30]:


gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.08, shear_range=0.3,
                               height_shift_range=0.08, zoom_range=0.1, horizontal_flip=True)
batches = gen.flow(x_train, y_train, batch_size=batch_size)
val_batches = gen.flow(x_validate, y_validate, batch_size=batch_size)


# In[33]:


cnn3.fit_generator(batches, steps_per_epoch=np.ceil(48000//batch_size), epochs=50, 
                    validation_data=val_batches, validation_steps=12000//batch_size, use_multiprocessing=False)


# In[34]:


score = cnn3.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:




