import numpy as np
import tensorflow as tf
import scipy.io as sio
import time
import logging
import cv2
from skimage.morphology import binary_erosion
from model_code.builder.abstract_builder import AbstractBuilder
from model_code.detector.detector_abstract import DetectorAbstract
from model_code.util import model_restore, save_img_prediction, save_img_mask
import os 
import matplotlib.pyplot as plt
from PIL import Image


class LSTM_SA_Tester():
    
    def __init__(self,
                dataprovider,
                net: AbstractBuilder,
                model_path,
                output_path,
                detector: DetectorAbstract,
                verbose=False):
                
        self._dataprovider = dataprovider
        self._background_mask_index = dataprovider.background_mask_index()
        self._net = net
        self._model_path = model_path
        self._output_path = output_path
        self._verbose = verbose
        self.detector = detector
        self._mask_nr = 1
                
    def _create_graph(self):
        
        g = tf.Graph()
        with g.as_default(): 
            
            logits, x_input, y_input, drop_rate = self._net.build()

            prediction = tf.nn.softmax(logits)
            prediction = tf.round(prediction)
            self._performance_pe_list = []
            self._performance_miou_list = []
            self._performance_pa_list = []
            self._performance_mpa_list = []
            self._performance_d_list = []

            return g, x_input, y_input, drop_rate, prediction

   
    def test(self, batch_size=1):               
        g, xi, yi, dr, p = self._create_graph()
        with tf.Session(graph=g) as sess:
            
            model_restore(sess, self._model_path)
            for i in self._dataprovider.dataset_length() // batch_size:
                #range(self._dataprovider.dataset_length() // batch_size)
                
                x, y = self._dataprovider.next_batch(batch_size)
                
                feed_dict = {xi: x, yi: y, dr:0.}         
                out = sess.run([p], feed_dict)
                out_p = out

                for j in range(len(out_p)):  # For each image in the batch

                    for k in range(out_p[j][0].shape[-1]):  # For each mask in the image
                        mask = out_p[j][0][..., k]
                        image = x[j][0]
                        border, original_with_overlay_red = self.detector.detect(image, mask)

                        #overlay = self.get_bouding_box(k, coords, original_with_overlay)

                        # Save the overlay image
        
        logging.info("Done.")
        return original_with_overlay_red, border
