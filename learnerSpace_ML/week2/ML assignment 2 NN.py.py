#!/usr/bin/env python
# coding: utf-8

# In[38]:


import os
import numpy as np
from sklearn.model_selection import train_test_split


# In[37]:


get_ipython().system('pip install tensorflow')
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# In[40]:


data = tf.keras.preprocessing.image_dataset_from_directory("homer_bart", image_size=(64,64), label_mode = "binary")
train_data = data.take(8)
test_data = data.skip(8)
train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_data = test_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# ### Image Processing

# In[43]:


# resizing the images and scaling the images

rescaled = tf.keras.Sequential([tf.keras.layers.Rescaling(1.0/255)])

model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(64,64,3)),
        rescaled,
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])


# In[47]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[50]:


history = model.fit(train_data, epochs=20, validation_data=test_data)


# In[51]:


loss, acc = model.evaluate(test_data)
print(f"Test Accuracy : {acc*100:.2f}%")


# In[ ]:




