#!/bin/python
from PIL import Image
from random import shuffle
import glob
import os
import numpy as np

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.WARNING)

from util import *

class DataProvider():
    
    def __init__(self, 
                 path, 
                 image_suffix="_rgb.png", 
                 label_suffix="_mask.png",
                 nclasses=3,
                 background=False,
                 background_mask_index=None,
                 shuffle_data=True,
                 debug=False):
        #init
        self._path = path
        self._image_suffix = image_suffix
        self._label_suffix = label_suffix
        
        self._nclasses = nclasses
        self._background = background
        if not background_mask_index and background:
            self._background_mask_index = nclasses
            logging.info("Setting self._background_mask_index to {}.".format(self._background_mask_index))
        
        self._debug = debug
        self._debug_var1 = True
        
        self._idx = -1
        self._filelist = self._get_filelist(shuffle_data)
        
    def _get_filelist(self, shuffle_data=False):
        file_names = []
        for fname in glob.glob(self._path):
            if (self._label_suffix == None and self._image_suffix in fname) or (self._image_suffix in fname and not self._label_suffix in fname):
                file_names.append(fname)
        self._nfiles = len(file_names)
        file_names = sorted(file_names)

        if shuffle_data:
            shuffle(file_names)
        logging.info("DataProvider loaded {} Files.".format(self._nfiles))
        return file_names
    
    def _cylce_file(self):
        self._idx += 1
        if self._idx >= len(self._filelist):
            self._idx = 0 
        
        if self._debug:
            self._idx = 0
        
    def _next_file(self):
        self._cylce_file()
        
        # load image
        image_name = self._filelist[self._idx]
        image = np.asarray(Image.open(image_name))
        if len(image.shape) == 2: image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        
        # load label
        if self._label_suffix != None:
            label_name = image_name.replace(self._image_suffix, self._label_suffix)
            label = np.asarray(Image.open(label_name))
        else:
            ncl = self._nclasses + 1 if self._background else self._nclasses
            shape = (image.shape[0], image.shape[1], ncl)
            label = np.ndarray(shape, dtype=float)
                
        # pre process if needed
        image = self._preprocess_image(image)
        if self._label_suffix != None:
            label = self._preprocess_label(label)
        
        if self._debug and self._debug_var1:
            self._debug_var1 = False
            print("Loaded:", self._filelist[0])
        
        return image, label
     
    def _preprocess_image(self, image):  
        return image
    
    def _preprocess_label(self, label):
        nx = label.shape[0]
        ny = label.shape[1]

        nclasses = self._nclasses
        processed_label = np.zeros((nx, ny, nclasses))
        
        if self._background:
            processed_label = np.zeros((nx, ny, nclasses+1))
            processed_label[..., nclasses] = (label==0)
            
        for i in range(self._nclasses):
            pixel_value = i+1
            processed_label[..., i] =  (label==pixel_value) # starting at 1; (label==1)
            
        return processed_label
    
    def next_batch(self, size=1):            
        image_batch = []
        label_batch = []
        
        for _ in range(size):
            image, label = self._next_file() 
            image_batch.append(image)
            label_batch.append(label)
        
        return np.asarray(image_batch), np.asarray(label_batch) 
    
    def dataset_length(self):
        return self._nfiles
        
    def background_mask_index(self):
        return self._background_mask_index
		
class SequenceDataProvider(DataProvider):
    
    def __init__(self, 
                 path,
                 sequence_length,
                 image_suffix=".jpg", 
                 label_suffix="_mask.png",
                 nclasses=3,
                 background=False,
                 background_mask_index=None,
                 shuffle_data=True,
                 debug=False):
        #init
        self._sequence_length = sequence_length
        DataProvider.__init__(self, path, image_suffix, label_suffix, nclasses, background, background_mask_index, shuffle_data, debug)
        
    def _get_filelist(self, shuffle_data=False):
        file_names = []
        for fname in glob.glob(self._path):
            if (self._label_suffix == None and self._image_suffix in fname) or (self._image_suffix in fname and not self._label_suffix in fname):
                file_names.append(fname)
        self._nfiles = len(file_names)
        file_names = sorted(file_names)
        
        sequences = []
        nsequence = -1
        for i in range(self._nfiles):
            if i % self._sequence_length == 0:
                sequences.append([])
                nsequence += 1
            fname = file_names[i]
            sequences[nsequence].append(fname)
        if shuffle_data:
            shuffle(sequences)
        
        logging.info("SequenceDataProvider loaded {} Sequences a {} Files.".format(self._nfiles//self._sequence_length, self._sequence_length))
        return sequences # = self._filelist
    
    def _next_sequence(self, slength):
        self._cylce_file()
        
        fnames_sequence = self._filelist[self._idx]
        images = []
        labels = []
        
        for i in range(slength):
            # load image
            image_name = fnames_sequence[i]
            image = np.asarray(Image.open(image_name))
            if len(image.shape) == 2: image = np.reshape(image, (image.shape[0], image.shape[1], 1))
            
            # load label
            if self._label_suffix != None:
                label_name = image_name.replace(self._image_suffix, self._label_suffix)
                label = np.asarray(Image.open(label_name))
            else:
                ncl = self._nclasses + 1 if self._background else self._nclasses
                shape = (image.shape[0], image.shape[1], ncl)
                label = np.ndarray(shape, dtype=float)
            
            # pre process if needed
            image = self._preprocess_image(image)
            if self._label_suffix != None:
                label = self._preprocess_label(label)
            
            images.append(image)
            labels.append(label)

        return np.asarray(images), np.asarray(labels) # shape (slength, 256, 256, 3)
    
    def next_batch(self, size=1, slength=None):
        if slength == None: slength = self._sequence_length
            
        image_batch = []
        label_batch = []
        
        for _ in range(size):
            image, label = self._next_sequence(slength) 
            image_batch.append(image)
            label_batch.append(label)
        
        return np.asarray(image_batch), np.asarray(label_batch) # shape (size, sequence_length, 256, 256, 3)
    
    def dataset_length(self):
        return self._nfiles//self._sequence_length


class MidSequenceDataProvider(DataProvider):
    
    def __init__(self, 
                 path,
                 sequence_length,
                 image_suffix="_rgb.png", 
                 label_suffix="_mask.png",
                 nclasses=3,
                 background=False,
                 background_mask_index=None,
                 shuffle_data=True,
                 debug=False):
        #init
        self._sequence_length = sequence_length
        DataProvider.__init__(self, path, image_suffix, label_suffix, nclasses, background, background_mask_index, shuffle_data, debug)
        
    def _get_filelist(self, shuffle_data=False):
        file_names = []
        for fname in glob.glob(self._path):
            if (self._label_suffix == None and self._image_suffix in fname) or (self._image_suffix in fname and not self._label_suffix in fname):
                file_names.append(fname)
        self._nfiles = len(file_names)
        file_names = sorted(file_names)
        
        sequences = []
        for i in range(0, self._nfiles, 100):
            sequences.append(file_names[i:i+100])

        sequences_windowed = []

        for sequence in sequences:
            
            windows = []
            windows.append([sequence[4], sequence[3], sequence[2], sequence[1], sequence[0], sequence[1], sequence[2], sequence[3], sequence[4], sequence[5]])
            nsequence = 1
            for i in range(6, len(sequence), 2):

                window = windows[nsequence-1][2:]
                window.append(file_names[i])
                window.append(file_names[i+1])

                windows.append(window)
                nsequence = nsequence + 1

            windows.append([sequence[i-6], sequence[i-5], sequence[i-4], sequence[i-3], sequence[i-2], sequence[i-1], sequence[i-2], sequence[i-3], sequence[i-4], sequence[i-5]])
            windows.append([sequence[i-4], sequence[i-3], sequence[i-2], sequence[i-1], sequence[i], sequence[i+1], sequence[i], sequence[i-1], sequence[i-2], sequence[i-3]])

            if len(sequences_windowed) == 0: sequences_windowed = windows
            else: sequences_windowed = np.concatenate((sequences_windowed, windows))

        sequences = sequences_windowed

        if shuffle_data:
            shuffle(sequences)
        
        self._num_sequences = len(sequences)
        logging.info("MidSequenceDataProvider loaded {} Sequences a {} Files.".format(self._num_sequences, self._sequence_length))
        return sequences # = self._filelist
    
    def _next_sequence(self, slength):
        self._cylce_file()
        
        fnames_sequence = self._filelist[self._idx]
        images = []
        labels = []
        
        for i in range(slength):
            # load image
            image_name = fnames_sequence[i]
            image = np.asarray(Image.open(image_name))
            if len(image.shape) == 2: image = np.reshape(image, (image.shape[0], image.shape[1], 1))
            
            # load label
            if self._label_suffix != None:
                label_name = image_name.replace(self._image_suffix, self._label_suffix)
                label = np.asarray(Image.open(label_name))
            else:
                ncl = self._nclasses + 1 if self._background else self._nclasses
                shape = (image.shape[0], image.shape[1], ncl)
                label = np.ndarray(shape, dtype=float)
            
            # pre process if needed
            image = self._preprocess_image(image)
            if self._label_suffix != None:
                label = self._preprocess_label(label)
            
            images.append(image)
            labels.append(label)

        return np.asarray(images), np.asarray(labels) # shape (slength, 256, 256, 3)
    
    def next_batch(self, size=1, slength=None):
        if slength == None: slength = self._sequence_length
            
        image_batch = []
        label_batch = []
        
        for _ in range(size):
            image, label = self._next_sequence(slength) 
            image_batch.append(image)
            label_batch.append(label)
        
        return np.asarray(image_batch), np.asarray(label_batch) # shape (size, sequence_length, 256, 256, 3)
    
    def dataset_length(self):
        return self._num_sequences