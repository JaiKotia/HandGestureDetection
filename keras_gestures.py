#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 11:59:20 2018

@author: jai
"""

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import imutils


EPOCHS = 10
INIT_LR = 1e-3
BS = 2

model = Sequential()
inputShape = (28, 28, 1)
   
model.add(Conv2D(20, (5, 5), padding="same",	input_shape=inputShape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(50, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

# softmax classifier
model.add(Dense(2))
model.add(Activation("softmax"))   


#Capture images
def capture_image():
    video = cv2.VideoCapture(0)
    bbox_initial = (100, 100, 200, 200)
    bbox = bbox_initial
    
    while True:
        success, frame = video.read()
        display = frame.copy()
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        
        roi = frame[100:300, 100:300]
    
    
        if not success:
            break
            
        cv2.imshow("frame", frame)
        
        k = cv2.waitKey(1) & 0xff
        if k == 27:# escape pressed 
            break
        elif k == 115: # s pressed
            fname = input("File name")
            cv2.imwrite('{}.jpg'.format(fname), roi)
        
        
        
cv2.destroyAllWindows()
video.release()

bg_img = cv2.imread('bg_black.jpg')

current_frame_img = cv2.imread('current.jpg')

def preprocess_image_v2(current_frame_img, bg_img):
    blur = cv2.GaussianBlur(current_frame_img, (15, 15), 2)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower = np.array([0,50,50])
    upper = np.array([180,140,180])
    mask = cv2.inRange(hsv, lower, upper)
    masked_img = cv2.bitwise_and(current_frame_img, current_frame_img, mask=mask)
    diff = cv2.absdiff(bg_img, masked_img)
    mask2 = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    th, mask_thresh = cv2.threshold(mask2, 10, 255, cv2.THRESH_BINARY)
    return mask_thresh
    

def preprocess_image(current_frame_img, bg_img):
    diff = cv2.absdiff(bg_img, current_frame_img)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    th, mask_thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    plot_image(mask_thresh)
    return mask_thresh

def bgrtorgb(image):
    return cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)


def plot_image(image, figsize=(8,8), recolour=False):
    if image.shape[-1] == 1 or len(image.shape) == 2:
        plt.imshow(image, cmap='gray')
    else:
        raise Exception("Image has invalid shape.")

mask_thresh = preprocess_image(current_frame_img, bg_img)
plot_image(mask_thresh)
masked = cv2.resize(mask_thresh, (300, 300))

plot_image(masked)

path=os.getcwd()+'/'
print(path)

data = []
labels = []
    
for img in os.listdir(path):
    print(img)
    image = cv2.imread(img)
    image = preprocess_image_v2(image, bg_img)
    image = cv2.resize(image, (28, 28))
    image = img_to_array(image)
    data.append(image)
     
    label = img.rstrip('0123456789')
    label = img.split('.')[0]
    print(label[:4])
    label = 1 if label[:4] == "five" else 0
    labels.append(label)
    
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
 
(trainX, testX, trainY, testY) = train_test_split(data, labels, 
    test_size=0.25, random_state=42)
 
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)    

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
 
# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

print("[INFO] training network...")
H=model.fit(trainX, trainY,
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1, validation_steps=5)

     
# save the model to disk
print("[INFO] serializing network...")
model.save(os.getcwd()["model"])



plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

 
# pre-process the image for classification
def classify_image(image):
    image = preprocess_image_v2(image, bg_img)
    orig = image.copy()
    
    image = cv2.resize(image, (28, 28))
    
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    (two, five) = model.predict(image)[0]
    
    label = "Five" if five > two else "Two"
    proba = five if five > two else two
    label = "{}: {:.2f}%".format(label, proba * 100)
     
    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
    	0.7, (255, 255, 255), 2)
     
    # show the output image
    cv2.imshow("Output", output)

image = cv2.imread('test202.jpg')
classify_image(image)

capture_image()
cv2.destroyAllWindows()

