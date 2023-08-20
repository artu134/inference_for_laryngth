#!/bin/python
import numpy as np
import tensorflow as tf

import logging

from gru_cell import *

#################
# normal layers #
#################

def conv2d(inputs, nfilters , ksize=3, padding="valid", regularizer=None, drate=0.0, activation=tf.nn.elu, bn=False, verbose=False):
    with tf.name_scope("conv2d"): 
        conv = tf.layers.conv2d(inputs, nfilters, ksize, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), bias_initializer=tf.constant_initializer(0.01), activation=activation, kernel_regularizer=regularizer)
        if bn:
            conv = tf.layers.batch_normalization(conv)
        drop = tf.layers.dropout(conv, rate=drate)
        if bn:
            drop = tf.layers.batch_normalization(drop)
        if verbose: logging.info(conv)
        tf.summary.histogram("activation_conv", drop)
        return drop
        
def conv2d_from_3d(inputs, nfilters , ksize=3, padding="valid", regularizer=None, drate=0.0, activation=tf.nn.elu, bn=False, verbose=False):
    with tf.name_scope("conv2d_from_3d"):
        shape = inputs.shape
        reshape1 = tf.reshape(inputs, [-1, shape[2], shape[3], shape[4]])
        conv1 = conv2d(reshape1, nfilters, ksize=ksize, padding=padding, regularizer=regularizer, drate=drate, activation=activation, bn=bn, verbose=verbose)
        reshape2 = tf.reshape(conv1, [-1, shape[1], shape[2], shape[3], nfilters])
        return reshape2  

def max_pool2d(inputs, size=2, strides=2, verbose=False):
    pool = tf.layers.max_pooling2d(inputs, pool_size=size, strides=strides)
    if verbose: logging.info(pool)
    #tf.summary.histogram("activation_pool", pool)
    return pool
    
def max_pool2d_from_3d(inputs, size=2, strides=2, verbose=False):
    with tf.name_scope("max_pool2d_from_3d"):
        shape_before = inputs.shape
        reshape1 = tf.reshape(inputs, [-1, shape_before[2], shape_before[3], shape_before[4]])
        pool1 = max_pool2d(reshape1, size=size, strides=strides, verbose=verbose)
        shape_after = pool1.shape
        #drop1 = tf.layers.dropout(pool1, rate=drate)
        #print("INFO: drate not used in max_pool2d_from_3d.")
        reshape2 = tf.reshape(pool1, [-1, shape_after[0], shape_after[1], shape_after[2], shape_after[3]])
        return reshape2

def deconv2d(inputs, nfilters, ksize=2, strides=2, padding="valid", regularizer=None, activation=tf.nn.elu, verbose=False):
    deconv = tf.layers.conv2d_transpose(inputs, nfilters, ksize, strides=strides, padding=padding, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False), bias_initializer=tf.constant_initializer(0.01), activation=activation, kernel_regularizer=regularizer)
    if verbose: logging.info(deconv)
    #tf.summary.histogram("activation_deconv", deconv)
    return deconv

def deconv2d_from_3d(inputs, nfilters, ksize=2, strides=2, padding="valid", regularizer=None, activation=tf.nn.elu, verbose=False):
    with tf.name_scope("deconv2d_from_3d"):
        shape_before = inputs.shape
        reshape1 = tf.reshape(inputs, [-1, shape_before[2], shape_before[3], shape_before[4]])
        deconv1 = deconv2d(reshape1, nfilters, ksize=ksize, strides=strides, padding=padding, regularizer=regularizer, activation=activation, verbose=verbose)
        shape_after = deconv1.shape
        #drop1 = tf.layers.dropout(deconv1, rate=drate)
        #print("INFO: drate not used in deconv2d_from_3d.")
        reshape2 = tf.reshape(deconv1, [-1, shape_after[0], shape_after[1], shape_after[2], nfilters])
        return reshape2
        
def crop_concat(inputs_1, inputs_2, verbose=False):
    with tf.name_scope("crop_and_concat"):
            cur_size = inputs_1.shape
            target_size = inputs_2.shape
            offset = [0, int((cur_size[1] - target_size[1]) // 2), int((cur_size[2] - target_size[2]) // 2), 0]
            size = (-1, int(target_size[1]), int(target_size[2]), -1)
            crop = tf.slice(inputs_1, offset, size) 
            concat = tf.concat([crop, inputs_2], 3)
            if verbose: logging.info("{} <".format(concat))
            return concat
            
def crop_concat_from_3d(inputs_1, inputs_2, verbose=False):
    with tf.name_scope("crop_and_concat"):
        shape_inputs1 = inputs_1.shape
        reshape_inputs1_1 = tf.reshape(inputs_1, [-1, shape_inputs1[2], shape_inputs1[3], shape_inputs1[4]])
        shape_inputs2 = inputs_2.shape
        reshape_inputs2_1 = tf.reshape(inputs_2, [-1, shape_inputs2[2], shape_inputs2[3], shape_inputs2[4]])
        _cc = crop_concat(reshape_inputs1_1, reshape_inputs2_1, verbose=verbose)
        reshape2 = tf.reshape(_cc, [-1, shape_inputs2[1], shape_inputs2[2], shape_inputs2[3], shape_inputs2[4]*2])
        return reshape2

# output is of size nfilters, fw/bw pass is computed with nfilters//2 
def bclstm(inputs, name, nfilters=64, input_shape=[256, 256, 64], ksize=3, verbose=False):
    with tf.name_scope("bclstm"):
        cell_fw = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                           input_shape=input_shape,
                                           output_channels=nfilters//2,
                                           kernel_shape=[ksize, ksize])

        cell_bw = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
                                           input_shape=input_shape,
                                           output_channels=nfilters//2,
                                           kernel_shape=[ksize, ksize])

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, dtype=tf.float32, scope=name)
        output_fw, output_bw = outputs
        concated_outputs = tf.concat([output_fw, output_bw], -1)
        if verbose: logging.info("{} <<".format(concated_outputs))
        return concated_outputs

# output is of size nfilters, fw/bw pass is computed with nfilters//2 
def bcgru(inputs, name, nfilters=64, input_shape=[256, 256, 64], ksize=3, verbose=False):
    with tf.name_scope("bcgru"):
        cell_fw = ConvGRUCell(shape=[input_shape[0], input_shape[1]],
                                           filters=nfilters//2,
                                           kernel=[ksize, ksize],
                                           activation=tf.identity,
                                           normalize=False)

        cell_bw = ConvGRUCell(shape=[input_shape[0], input_shape[1]],
                                           filters=nfilters//2,
                                           kernel=[ksize, ksize],
                                           activation=tf.identity,
                                           normalize=False)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, dtype=tf.float32, scope=name)
        output_fw, output_bw = outputs
        concated_outputs = tf.concat([output_fw, output_bw], -1)
        if verbose: logging.info("{} <<".format(concated_outputs))
        return concated_outputs


#######################
# acc/loss functions #
#######################
def _dice(predictions, labels, nclasses, smooth=1e-5):

    with tf.name_scope("dice"):

        dices = []
        for i in range(nclasses):

            p = predictions[..., i]
            l = labels[..., i]

            tp = tf.reduce_sum(p * l)

            npixel_pred = tf.reduce_sum(p)
            npixel_label = tf.reduce_sum(l)

            dice = (2. * tp + smooth) / (npixel_pred + npixel_label + smooth)
            dices.append(dice)

        return tf.reduce_mean(dices)

def dice(predictions, labels, nclasses, smooth=1e-5, class_index=None):

    predictions = tf.round(predictions)
    labels = tf.round(labels)

    if class_index == None:
        return _dice(predictions, labels, nclasses, smooth)

    else:

        with tf.name_scope("dice_class_" + str(class_index)):

            predictions = predictions[..., class_index]
            labels = labels[..., class_index]

            tp = tf.reduce_sum(predictions * labels)
            
            npixel_pred = tf.reduce_sum(predictions)
            npixel_label = tf.reduce_sum(labels)

            dice = (2. * tp + smooth) / (npixel_pred + npixel_label + smooth)

            return dice

def sdc(predictions, labels, nclasses=3, smooth=1e-5):
    with tf.name_scope("sdc"):
        
        avg_dice = []
        for i in range(nclasses):
            
            per_class_pred = predictions[..., i]
            per_class_label = labels[..., i]
            
            tp = tf.reduce_sum(per_class_pred * per_class_label)
            npixel_pred = tf.reduce_sum(per_class_pred)
            npixel_label = tf.reduce_sum(per_class_label)

            dice = (2. * tp + smooth) / (npixel_pred + npixel_label + smooth)
            avg_dice.append(dice)
        
        return avg_dice

def metric_pixel_accuraccy(predictions, labels):
    with tf.name_scope("metrics_pa"):
        predictions = tf.round(predictions)
        
        npixel_correct = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), dtype=tf.float32))
        npixel_all = tf.cast(tf.size(labels), dtype=tf.float32)
        
        return npixel_correct / npixel_all
    
def metric_pixel_error(predictions, labels):
    with tf.name_scope("metrics_pe"):
        return 1 - metric_pixel_accuraccy(predictions, labels)

def metric_mean_pa(predictions, labels, nclasses):
    with tf.name_scope("metrics_mean_pa"):
        predictions = tf.round(predictions)
        
        metrics = []
        for i in range(nclasses):
            per_class_pred = predictions[..., i]
            per_class_label = labels[..., i]
            
            tp = tf.reduce_sum(per_class_pred * per_class_label)
            npixel_per_class = tf.reduce_sum(per_class_label)

            m = tp / npixel_per_class
            m = tf.where(tf.is_nan(m), 1., m)
            metrics.append(m)

        return tf.reduce_mean(metrics)
    
def metric_mean_iou(predictions, labels, nclasses):
    # IoU = true_positive / (true_positive + false_positive + false_negative)
    with tf.name_scope("metrics_mean_iou"):
        predictions = tf.round(predictions)
        
        ones_like_labels = tf.ones_like(labels[..., 0])
        zeros_like_labels = tf.zeros_like(labels[..., 0])
        ones_like_predictions = tf.ones_like(predictions[..., 0])
        zeros_like_predictions = tf.zeros_like(predictions[..., 0])
        
        metrics = []
        for i in range(nclasses):
            per_class_pred = predictions[..., i]
            per_class_label = labels[..., i]
            
            tp = tf.reduce_sum(tf.cast(tf.logical_and(
                                            tf.equal(per_class_label, ones_like_labels), 
                                            tf.equal(per_class_pred, ones_like_predictions)), "float"))
            fp = tf.reduce_sum(tf.cast(tf.logical_and(
                                            tf.equal(per_class_label, zeros_like_labels), 
                                            tf.equal(per_class_pred, ones_like_predictions)), "float"))
            fn = tf.reduce_sum(tf.cast(tf.logical_and(
                                            tf.equal(per_class_label, ones_like_labels), 
                                            tf.equal(per_class_pred, zeros_like_predictions)), "float"))

            m = tp / (tp + fp + fn)
            m = tf.where(tf.is_nan(m), 1., m)
            metrics.append(m)

        return tf.reduce_mean(metrics)
