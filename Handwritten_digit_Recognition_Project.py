#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# In[9]:


mnist = tf.keras.datasets.mnist


# In[10]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[11]:


x_train = tf.keras.utils.normalize(x_train, axis=1)


# In[12]:


x_test = tf.keras.utils.normalize(x_test, axis=1)


# In[13]:


model = tf.keras.models.Sequential()


# In[14]:


model.add(tf.keras.layers.Flatten(input_shape=(28,28)))


# In[15]:


model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))


# In[16]:


model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))


# In[17]:


model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))


# In[44]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[45]:


model.fit(x_train, y_train, epochs=3)


# In[46]:


loss, accuracy = model.evaluate(x_test, y_test)


# In[47]:


print(accuracy)


# In[48]:


print(loss)


# In[49]:


model.save('digits.model')


# In[ ]:





# In[51]:


for x in range(1,6):
    try:
        img = cv.imread(f'{x}.png')[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f'The result is probably: {np.argmax(prediction)}')
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        pass


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




