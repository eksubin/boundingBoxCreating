from configuration import config

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
##
rows = open(config.ANNOTS_PATH).read().strip().split("\n")
data = []
targets = []
filenames = []
##
for row in rows:
    row = row.split(',')
    print(row)
    raw_startX = row[6]
    startX = ''.join([n for n in raw_startX if n.isdigit()])

    raw_startY = row[7]
    startY = ''.join([n for n in raw_startX if n.isdigit()])

    raw_width = row[8]
    width = ''.join([n for n in raw_width if n.isdigit()])

    raw_height = row[9]
    height = ''.join([n for n in raw_height if n.isdigit()])

    print(height)
    #(filename) = row

##


##

