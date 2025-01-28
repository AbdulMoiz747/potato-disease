#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[2]:


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


# # Data Collection, Preprocessing

# In[15]:


IMAGE_SIZE = 256
BATCH_SIZE =32
CHANNELS =3
EPOCHS =50


# In[4]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
"PlantVillage",
shuffle=True,
image_size = (IMAGE_SIZE,IMAGE_SIZE),
batch_size = BATCH_SIZE
)


# In[5]:


class_names = dataset.class_names
class_names
# there are three classes: 0, 1, 2


# In[6]:


len(dataset)


# In[13]:


plt.figure(figsize=(10, 10))
for image_batch, label_batch in dataset.take(1):
    #to visualize the image
    for i in range(12):
        ax = plt.subplot(3,4,i+1)
        plt.imshow(image_batch[0].numpy().astype("uint8")) #printing the first image
        plt.title(class_names[label_batch[i]])
        plt.axis("off")


# In[14]:


len(dataset)


# In[18]:


train_size = 0.8
len(dataset)*train_size


# In[19]:


train_ds= dataset.take(54)
len(train_ds)


# In[20]:


test_ds = dataset.skip(54)
len(test_ds)


# In[21]:


val_size=0.1
len(dataset)*val_size


# In[22]:


test_ds = test_ds.skip(6)
len(test_ds)


# In[23]:


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)
    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


# In[24]:


train_ds, val_ds, test_ds = get_dataset_partitions_tf (dataset)


# In[25]:


len(train_ds)


# In[26]:


len(val_ds)


# In[27]:


len(test_ds)


# In[28]:


#do chaching: for the first time it will read the image and for the next time 
#for it to be used it keeps it in the memory. prefetch is used to improve performance
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[48]:


resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.Rescaling(1.0/255)
])


# In[49]:


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])


# # Model Building

# CNN: neural network to solve classification problem. 
# CNN is like a machine that looks at an image, picks out key details, and uses those details to figure out what the image is showing.

# In[57]:


n_classes = 3


input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

model = models.Sequential([
    # Explicitly define the input shape including the batch size
    layers.InputLayer(input_shape=input_shape),
    
    # Preprocessing layers
    resize_and_rescale,
    
    # Define Conv2D layers
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax')  # Output layer
])

# No need to manually build the model
model.summary()


# In[62]:


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# In[61]:


history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=50,
)


# In[63]:


scores = model.evaluate(test_ds)


# In[64]:


scores


# In[65]:


history


# In[66]:


history.params


# In[67]:


history.history.keys()


# In[68]:


len(history.history['accuracy'])


# In[69]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


# In[70]:


plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[72]:


import numpy as np
for images_batch, labels_batch in test_ds.take(1):
    
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])
    
    batch_prediction = model.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])


# In[73]:


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


# In[74]:


plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]] 
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        plt.axis("off")


# In[80]:


import os

# Extract filenames, strip extensions, and convert to integers
model_version = max([int(i.split('.')[0]) for i in os.listdir("../models") if i.split('.')[0].isdigit()] + [0]) + 1

# Save the model with the new version number
model.save(f"../models/{model_version}.keras")


# In[ ]:




