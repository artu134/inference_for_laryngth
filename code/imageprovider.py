from lstm_SA_type_full_stable import *
from glob import glob
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

# Image Provider
class SingleImageProvider(object):
    def __init__(self, img_path, sequence_length, img_size):
        self.img_path = img_path
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.load_data()

    def load_data(self):
        imgs = glob(self.img_path)
        self.sequences = [self.process_image(img) for img in imgs]

    def process_image(self, img):
        img_arr = np.array(load_img(img, target_size=self.img_size))
        img_arr = img_arr[np.newaxis, ...]  # Add an extra dimension for batch
        sequence = np.repeat(img_arr, self.sequence_length, axis=0)  # Repeat image to create sequence
        return sequence

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        return self.sequences[i]