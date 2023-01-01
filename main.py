# %%
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras
import tensorflow_hub as hub
import tensorflow.keras.layers
import matplotlib.pyplot as plt
from IPython.display import Image
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# %%
import tensorflow as tf

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    print("Device:", tpu.master())
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    strategy = tf.distribute.MirroredStrategy()

# %%
images = list()
labels = list()

for i in os.listdir("C:\\Users\\lenovo\\Downloads\\Isic_Augumented_image\\skin_cancer\\train\\actinic keratosis"): #read all Actinic Keratosis data 
    if ".jpg" in i: #this if block for only read .jpg files
        path = "C:\\Users\\lenovo\\Downloads\\Isic_Augumented_image\\skin_cancer\\train\\actinic keratosis\\"+i # create path
        img = plt.imread(path) # read created path
        img = cv2.resize(img,(224,224)) # resize image for lower processing power
        images.append(img) # append image to images data
        labels.append(0)
        
for i in os.listdir("C:\\Users\\lenovo\\Downloads\\Isic_Augumented_image\\skin_cancer\\train\\basal cell carcinoma\\"): #read all Basal Cell Carcinoma data 
    if ".jpg" in i: #this if block for only read .jpg files
        path = "C:\\Users\\lenovo\\Downloads\\Isic_Augumented_image\\skin_cancer\\train\\basal cell carcinoma\\"+i # create path
        img = plt.imread(path) # read created path
        img = cv2.resize(img,(224,224)) # resize image for lower processing power
        images.append(img) # append image to images data
        labels.append(1)
for i in os.listdir("C:\\Users\\lenovo\\Downloads\\Isic_Augumented_image\\skin_cancer\\train\\dermatofibroma\\"): #read all Dermatofibroma data 
    if ".jpg" in i: #this if block for only read .jpg files
        path = "C:\\Users\\lenovo\\Downloads\\Isic_Augumented_image\\skin_cancer\\train\\dermatofibroma\\"+i # create path
        img = plt.imread(path) # read created path
        img = cv2.resize(img,(224,224)) # resize image for lower processing power
        images.append(img) # append image to images data
        labels.append(2)
for i in os.listdir("C:\\Users\\lenovo\\Downloads\\Isic_Augumented_image\\skin_cancer\\train\\pigmented benign keratosis\\"): #read all Pigmented Benign Keratosis data 
    if ".jpg" in i: #this if block for only read .jpg files
        path = "C:\\Users\\lenovo\\Downloads\\Isic_Augumented_image\\skin_cancer\\train\\pigmented benign keratosis\\"+i # create path
        img = plt.imread(path) # read created path
        img = cv2.resize(img,(224,224)) # resize image for lower processing power
        images.append(img) # append image to images data
        labels.append(3)
for i in os.listdir("C:\\Users\\lenovo\\Downloads\\Isic_Augumented_image\\skin_cancer\\train\\vascular lesion\\"): #read all Vascular Lesion data 
    if ".jpg" in i: #this if block for only read .jpg files
        path = "C:\\Users\\lenovo\\Downloads\\Isic_Augumented_image\\skin_cancer\\train\\vascular lesion/"+i # create path
        img = plt.imread(path) # read created path
        img = cv2.resize(img,(224,224)) # resize image for lower processing power
        images.append(img) # append image to images data
        labels.append(4)
images = np.array(images)

images.shape[0] #array length

# %%
plt.subplot(1,4,1)
plt.imshow(images[20]) # image 1
plt.title(labels[20])
plt.axis("off")
plt.subplot(1,4,2) # image 2
plt.imshow(images[300])
plt.title(labels[300])
plt.axis("off")
plt.subplot(1,4,3) #image 3
plt.imshow(images[2000])
plt.title(labels[2000])
plt.axis("off")
plt.subplot(1,4,4) #image 4 
plt.imshow(images[200])
plt.title(labels[200])
plt.axis("off")
plt.show()

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

# %%
def input_preprocess(images, labels):
    label = tf.one_hot(labels, 5)
    return images, labels

# %%
from sklearn.model_selection import train_test_split
x_train,x_test,Y_train,Y_test = train_test_split(images,labels,train_size = 0.8, random_state = 0)

# %%
x_train,x_val,Y_train,Y_val = train_test_split(x_train, Y_train, test_size = 0.1, random_state = 42)


# %%
x_train = np.array(x_train)
x_test = np.array(x_test)
x_val = np.array(x_val)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
Y_val = np.array(Y_val)

# %%
print("Train Split: ", x_train.shape)
print("Test Split: ", x_test.shape)
print("Validation Split: ", x_val.shape)

# %%
from tensorflow.keras.applications import EfficientNetB0

with strategy.scope():
    inputs = layers.Input(shape=(224, 224, 3))
    x = img_augmentation(inputs)
    outputs = EfficientNetB0(include_top=True, weights=None, classes=5)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
    )

model.summary()

# %%
IMG_SIZE = 224

def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
    )
    model.summary()
    return model

# %%
NUM_CLASSES = 5

with strategy.scope():
    model = build_model(num_classes=NUM_CLASSES)



# %%
def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
    )
    model.summary()
    
unfreeze_model(model)

# %%


# %%
early_stop = EarlyStopping(monitor='val_loss', patience=2)

epochs = 15  # @param {type: "slider", min:8, max:50}

hist = model.fit(x_train,Y_train ,epochs=epochs , validation_data=(x_val,Y_val), verbose=2, callbacks=[early_stop])

# %%
import matplotlib.pyplot as plt

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

plot_hist(hist)

# %%
plot_hist(hist)


# %%
test_loss, test_acc = model.evaluate(x_test, Y_test)
print('Test Accuracy: ', test_acc)

# %%



