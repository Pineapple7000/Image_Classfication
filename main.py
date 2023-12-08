import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import cv2 as cv

(trainig_images ,training_labels), (testing_images ,testing_labels) = tensorflow.keras.datasets.cifar10.load_data()

trainig_images = trainig_images / 255
testing_images = testing_images / 255

class_names = ['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

for image in range(16):
    plt.subplot(4,4,image+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(trainig_images[image], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[image][0]])

plt.show()

# python3.9 -m venv venv 
# source venv/bin/activate
# deactivate
# jupyter lab