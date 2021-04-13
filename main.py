from configuration import config
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

##
rows = open(config.ANNOTS_PATH).read().strip().split("\n")
data = []
targets = []
filenames = []
##
for row in rows:
    row = row.split(',')
    filename = row[0]

    raw_startX = row[6]
    startX = ''.join([n for n in raw_startX if n.isdigit()])

    raw_startY = row[7]
    startY = ''.join([n for n in raw_startY if n.isdigit()])

    raw_width = row[8]
    width = ''.join([n for n in raw_width if n.isdigit()])

    raw_height = row[9]
    height = ''.join([n for n in raw_height if n.isdigit()])

    startX = float(startX)
    startY = float(startY)

    width = float(width)
    height = float(height)

    endX = startX + height
    endY = startY + width

    # everything working until here
    # reading image
    imagePath = os.path.join(config.IMAGES_PATH, filename)
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    print(filename)

    data.append(image)
    targets.append((startX, startY, endX, endY))

##
#Do a test train split
data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")

# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(512, 512, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)
# construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)

##
# compile the model
opt = Adam(lr=config.INIT_LR)
model.compile(loss="mse", optimizer=opt, metrics = ['accuracy'])
print(model.summary())
# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
	data, targets,
	batch_size=2,
	epochs=5,
	verbose=1)

##

