
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from rnet.models import ResNet
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers.legacy import Adam
import cv2
from imutils import paths
import os



EPOCHS = 100
INIT_LR = 1e-4   
BS = 2 



imagePaths = list(paths.list_images("dataset"))
data = []
labels = []

for imagePath in imagePaths:
    # extract the class label from the filename
    
    label = imagePath.split(os.path.sep)[-2]
    image = load_img(imagePath,target_size=(32, 32))
    
    image = img_to_array(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    data.append(image)
    labels.append(label)


labels = np.array(labels)

data = np.array(data, dtype="float32")
data = np.expand_dims(data, axis=-1)
data /= 255.0


# plt.imshow(data[4], interpolation='nearest')
# plt.show()


le = LabelBinarizer()
labels = le.fit_transform(labels)


classTotals = labels.sum(axis=0)
classWeight = {}


#loop over all classes and calculate the class weight
for i in range(0, len(classTotals)):
  	 classWeight[i] = classTotals.max() / classTotals[i]

(trainX, testX, trainY, testY) = train_test_split(data,
 	labels, test_size=0.10, stratify=labels, random_state=42)

aug = ImageDataGenerator(
 	rotation_range=5,
 	zoom_range=0.05,
 	width_shift_range=0.1,
 	height_shift_range=0.1,
 	shear_range=0.15,
 	horizontal_flip=False,
 	fill_mode="constant"  )

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),
 	(64, 64, 128, 256), reg=0.0005)



model.compile(loss="binary_crossentropy", optimizer=opt,
 	metrics=["accuracy"])


# train the network
print("[INFO] training network...")
H = model.fit(
 	aug.flow(trainX, trainY, batch_size=BS),
 	validation_data=(testX, testY),
 	steps_per_epoch=len(trainX) // BS,
 	epochs=EPOCHS,
 	class_weight=classWeight,
 	verbose=1)

# define the list of label names
labelNames = "#0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]
 

# save the model to disk
print("[INFO] serializing network...")
model.save("char.model", save_format="h5")

# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

