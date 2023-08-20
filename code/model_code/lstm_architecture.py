#!/bin/python
import numpy as np
import tensorflow as tf
import time
import scipy.io as sio
import cv2
from decimal import Decimal
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import logging
from scipy.ndimage import binary_erosion

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.WARNING)

from util import *
from layer import * 


class LSTM_SA_Architecture():
    
    def __init__(self,
                 xsize=(256, 256),
                 ysize=(256, 256),
                 ndimensions=3,
                 nfilters=64,
                 nsequences=10,
                 nbatches=1,
                 nlayers=6,
                 nclasses=4,
                 loss_function="dice",
                 class_weights=None,
                 learning_rate=0.0001,
                 decay_rate=None,
                 bn=False,
                 reg=None,
                 reg_scale=0.001,
                 image_std=True,
                 crop_concat=True,
                 constant_nfilters=True,
                 name=None,
                 verbose=False,
                 two_gpus=False,
                 gru=False,
                 midseq=False):
        
        tf.reset_default_graph()

        self._xsize = xsize
        self._ysize = ysize
        self._ndimensions = ndimensions
        self._nfilters = nfilters
        self._nsequences = nsequences
        self._nbatches = nbatches
        self._nclasses = nclasses
        self._nlayers = nlayers
        self._loss_function = loss_function
        self._class_weights = class_weights
        self._learning_rate = learning_rate
        self._decay_rate = decay_rate
        self._bn = bn
        self._reg = reg
        self._reg_scale = reg_scale
        self._image_std = image_std
        self._crop_concat = crop_concat
        self._constant_nfilters = constant_nfilters
        self._name = name
        self._verbose = verbose
        self._two_gpus = two_gpus
        self._gru = gru
        self._midseq = midseq
        
        with tf.name_scope("regularizer"):
            if reg == None:
                self._regularizer = None
            elif reg == "L2":
                self._regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)
            else:
                raise Exception("Unknown Regularizer.")  
            
    def create_net(self):
        
        output_node = None

        # inputs
        with tf.name_scope("inputs"):
            
            x_input = tf.placeholder(tf.float32, [self._nbatches, self._nsequences, self._xsize[0], self._xsize[1], self._ndimensions])
            in_node = x_input
            
            # needs testing
            if self._image_std:
                shape = in_node.shape
                in_node = tf.reshape(in_node, [-1, shape[2], shape[3], shape[4]])
                in_node = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), in_node, dtype=tf.float32)
                in_node = tf.reshape(in_node, [-1, shape[1], shape[2], shape[3], shape[4]])

            y_input = tf.placeholder(tf.float32, [self._nbatches, self._nsequences, self._ysize[0], self._ysize[1], self._nclasses])

            drop_rate_input = tf.placeholder(tf.float32)

            global_step = tf.Variable(0, trainable=False)

        # ========================================================================================================
                
        # layers
        down_nodes = [None for _ in range(self._nlayers)]
        for layer in range(1, self._nlayers+1):
            if self._constant_nfilters:
                cur_nfilters = self._nfilters
            else:
                cur_nfilters = self._nfilters * 2**(layer-1)
            
            # down
            with tf.name_scope("downsampling_{}".format(str(layer))):
            
                conv1 = conv2d_from_3d(in_node, cur_nfilters, ksize=3, padding="same", drate=drop_rate_input, bn=self._bn, regularizer=self._regularizer, verbose=self._verbose)
                conv2 = conv2d_from_3d(conv1, cur_nfilters, ksize=3, padding="same", drate=drop_rate_input, bn=self._bn, regularizer=self._regularizer, verbose=self._verbose)
                # added
                lstm_shape = [conv2.shape[2], conv2.shape[3], conv2.shape[4]]
                if self._two_gpus:
                    with tf.device('/device:GPU:1'):
                        if self._gru:
                            rec = bcgru(conv2, "bcgruDown"+str(layer), nfilters=cur_nfilters, input_shape=lstm_shape, ksize=3, verbose=self._verbose)
                        else:
                            rec = bclstm(conv2, "bclstmDown"+str(layer), nfilters=cur_nfilters, input_shape=lstm_shape, ksize=3, verbose=self._verbose)
                else:
                    if self._gru:
                        rec = bcgru(conv2, "bcgruDown"+str(layer), nfilters=cur_nfilters, input_shape=lstm_shape, ksize=3, verbose=self._verbose)
                    else:
                        rec = bclstm(conv2, "bclstmDown"+str(layer), nfilters=cur_nfilters, input_shape=lstm_shape, ksize=3, verbose=self._verbose)

                in_node = rec
                down_nodes[layer-1] = in_node
                                
                if layer < self._nlayers:
                    pool1 = max_pool2d_from_3d(in_node, verbose=self._verbose)
                    in_node = pool1

        output_node = None
        for layer in range(self._nlayers-1, 0, -1):
            if self._constant_nfilters:
                cur_nfilters = self._nfilters
            else:
                cur_nfilters = self._nfilters * 2**(layer-1)

            # up
            with tf.name_scope("upsampling_{}".format(str(layer))):
            
                _node = None
                if self._crop_concat:
                    deconv1 = deconv2d_from_3d(in_node, cur_nfilters, verbose=self._verbose)
                    cc1 = crop_concat_from_3d(down_nodes[layer-1], deconv1)
                    _node = cc1
                    if self._constant_nfilters:
                        conv1x1 = conv2d_from_3d(_node, cur_nfilters, ksize=1, padding="same")
                        _node = conv1x1
                else:
                    deconv1 = deconv2d_from_3d(in_node, cur_nfilters*2, verbose=self._verbose)
                    _node = deconv1
                conv1 = conv2d_from_3d(_node, cur_nfilters, ksize=3, padding="same", drate=drop_rate_input, bn=self._bn, verbose=self._verbose)
                conv2 = conv2d_from_3d(conv1, cur_nfilters, ksize=3, padding="same", drate=drop_rate_input, bn=self._bn, verbose=self._verbose)
                # added
                lstm_shape = [conv2.shape[2], conv2.shape[3], conv2.shape[4]]
                if self._two_gpus:
                    with tf.device('/device:GPU:1'):
                        if self._gru:
                            rec = bcgru(conv2, "bcgruUp"+str(layer), nfilters=cur_nfilters, input_shape=lstm_shape, ksize=3, verbose=self._verbose)
                        else:
                            rec = bclstm(conv2, "bclstmUp"+str(layer), nfilters=cur_nfilters, input_shape=lstm_shape, ksize=3, verbose=self._verbose)
                else:
                    if self._gru:
                        rec = bcgru(conv2, "bcgruUp"+str(layer), nfilters=cur_nfilters, input_shape=lstm_shape, ksize=3, verbose=self._verbose)
                    else:
                        rec = bclstm(conv2, "bclstmUp"+str(layer), nfilters=cur_nfilters, input_shape=lstm_shape, ksize=3, verbose=self._verbose)

                in_node = rec
                                
                if layer == 1:
                    final_conv1 = conv2d_from_3d(in_node, self._nclasses, ksize=1, padding="same", activation=None)
                    if self._verbose: logging.info(final_conv1)
                    output_node = final_conv1
                    
        # ========================================================================================================

        return output_node, x_input, y_input, drop_rate_input, global_step
                    
    def get_loss(self, logits, labels):
        
        with tf.name_scope("loss"): 

            if self._midseq:
                logits = tf.slice(logits, (0, 4, 0, 0, 0), (1, 2, self._xsize[0], self._xsize[1], self._nclasses))
                labels = tf.slice(labels, (0, 4, 0, 0, 0), (1, 2, self._ysize[0], self._ysize[1], self._nclasses))

            flat_logits = tf.reshape(logits, [-1, self._nclasses])
            flat_labels = tf.reshape(labels, [-1, self._nclasses])
            
            if self._loss_function == "softmax":  
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits, labels=flat_labels)

                # weighted loss
                if self._class_weights != None:
                    # deduce weights for batch samples based on their true label
                    weights = tf.reduce_sum(self._class_weights * flat_labels, axis=1)
                    cross_entropy = cross_entropy * weights

                loss = tf.reduce_mean(cross_entropy)
                reg_loss = tf.losses.get_regularization_loss()
                return loss + reg_loss
            
            elif self._loss_function == "dice":
            
                logits = tf.nn.softmax(logits)
                flat_logits = tf.reshape(logits, [-1, self._nclasses])
                flat_labels = tf.reshape(labels, [-1, self._nclasses])
            
                loss = sdc(labels=flat_labels, predictions=flat_logits, nclasses=self._nclasses)

                if self._class_weights != None:
                    loss = [a*b for a,b in zip(loss, self._class_weights)]
                    loss = tf.reduce_mean(loss)
                    loss = np.mean(self._class_weights) - loss
                else:
                    loss = tf.reduce_mean(loss)
                    loss = 1 - loss
                
                reg_loss = tf.losses.get_regularization_loss()
                return loss + reg_loss
            
            else:
                raise Exception("Unknown Loss-Function.")                
    
    def get_optimizer(self, loss, global_step):
        
        with tf.name_scope("optimizer"):
        
            if self._decay_rate != None:
                lr = tf.train.exponential_decay(self._learning_rate, global_step, self._decay_rate, 0.95, staircase=True)
            else:
                lr = self._learning_rate
                
            tf.summary.scalar("learning_rate", lr)

            optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
            return optimizer
        
    def get_architecture_name(self):
        lr = "%.2E" % Decimal(str(self._learning_rate))
        rs = "%.2E" % Decimal(str(self._reg_scale))
        suf = "" if self._name == None else "_{}".format(self._name)
        return "LSTM_SA_L{}_F{}_Dim{}_Seq{}_LF{}_LR{}_C{}_CW{}_DeR{}_BN{}_Reg{}_RegS{}_Std{}_CC{}_constF{}_GRU{}_MidS{}{}".format(self._nlayers, 
                                                       self._nfilters,
                                                       self._ndimensions,
                                                       self._nsequences,
                                                       self._loss_function, 
                                                       lr,
                                                       self._nclasses,
                                                       self._class_weights,
                                                       self._decay_rate, 
                                                       self._bn, 
                                                       self._reg,
                                                       rs,
                                                       self._image_std,
                                                       self._crop_concat,
                                                       self._constant_nfilters,
                                                       self._gru,
                                                       self._midseq,
                                                      suf)
          