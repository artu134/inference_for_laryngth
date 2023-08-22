#!/bin/python
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from skimage import segmentation

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.WARNING)

def model_save(sess, path, global_step):
    saver = tf.train.Saver(max_to_keep=100000)
    model_path = "{}model".format(path)
    save_path = saver.save(sess, model_path, global_step=global_step)
    logging.info("Model saved to file: {}".format(save_path))
    return save_path
    
def model_restore(sess, path):
    logging.info("Restoring model from file: {} ...".format(path))
    saver = tf.train.Saver()
    saver.restore(sess, path)
    logging.info("Model successfully restored.")

def to_rgb(img):    
    img = np.atleast_3d(img)
    channels = img.shape[3]
    if channels < 3:
        img = np.tile(img, 3)

    img -= np.amin(img)
    if np.amax(img) != 0:
        img /= np.amax(img)
    img *= 255
    return img

def save_img_prediction(batch_x, batch_y, batch_pred, path, step=None, image_name=None, background_mask=None):

    is_gw_image = (batch_x.shape[3] == 1)
    if is_gw_image:
        batch_x = np.tile(batch_x, 3)

    batch_x = batch_x.astype(float)
    nbatch = batch_x.shape[0]
    nx = batch_x.shape[1]
    ny = batch_x.shape[2]
    nch = batch_x.shape[3]
    ncl = batch_pred.shape[3]

    gt = [None for _ in range(ncl)]
    prediction = [None for _ in range(ncl)]
    for i in range(ncl):
        gt[i] = to_rgb(np.reshape(batch_y[..., i], (-1, nx, ny, 1)))
        prediction[i] = to_rgb(np.reshape(batch_pred[..., i], (-1, nx, ny, 1)))

    batch_img = [None for _ in range(nbatch)]
    for i in range(nbatch):
        
        original = batch_x[i]
        original_with_overlay = image_overlay(original, batch_pred[i], background_mask=background_mask)
        batch_img[i] = original_with_overlay
        
        for j in range(ncl):
            _gt = gt[j][i]
            _pred = prediction[j][i]
            
            batch_img[i] = np.concatenate( (batch_img[i], _gt, _pred), axis=1 )

    img = batch_img[0]
    for i in range(1, nbatch):
        img = np.concatenate( (img, batch_img[i]), axis=0 )

    if not os.path.exists(path):
        os.makedirs(path)
    if image_name != None:
        img_path = "{}{}.png".format(path, image_name)
    else:
        img_path = "{}prediction_step_{}.png".format(path, step)
        
    Image.fromarray(img.astype(np.uint8)).save(img_path, "PNG", dpi=[300,300], quality=100)

def save_img_mask(batch_pred, path, image_name, background_mask=None):
    if not os.path.exists(path):
        os.makedirs(path)
    img_path = "{}{}_pred.png".format(path, image_name)
        
    nx = batch_pred.shape[0]
    ny = batch_pred.shape[1]
    ncl = batch_pred.shape[2]
    
    masks = [np.ones_like(batch_pred[...,0]) for _ in range(ncl)]
    masks = masks[:background_mask] + masks[background_mask+1:]

    label_value = 1
    for i in range(0, ncl):
        if i != background_mask:
            
            pred = batch_pred[..., i]
            mask = masks[i]
            mask = mask * pred
            mask = mask * label_value
            masks[i] = mask
            
            label_value += 1
        
    r = masks[0]
    for i in range(1, len(masks)):
        r = np.add(r, masks[i])
    
    Image.fromarray(r.astype(np.uint8)).save(img_path, "PNG", dpi=[300,300], quality=100)

# currently only for 11 classes
def image_overlay(img, masks, background_mask=None):

    colors = [(255,0,0), (255,255,0), (0,0,255), (128,0,255), (255,128,0), (0,128,0), (128,0,0), (0,0,128), (128,128,0), (128,0,128), (0,128,128)]
    nmasks = masks.shape[2]
    r = img
    for i in range(nmasks):
        if i != background_mask:
            _mask = masks[..., i].astype(int)
            r = segmentation.mark_boundaries(r, _mask, colors[i], mode="inner")

    return np.asarray(r)

def clear_folders(*folders):
    for folder in folders:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
