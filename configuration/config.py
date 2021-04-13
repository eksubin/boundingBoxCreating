import os

#paths to each of the locations

BASE_PATH = "dataset"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "sample_annotation.csv"])

BASE_OUTPUT = "output"
# define the path to the output serialized model, model training plot,
# and testing image filenames

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])

# deep learning parameters
INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32