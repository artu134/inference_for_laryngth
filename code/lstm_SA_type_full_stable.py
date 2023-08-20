#!/bin/python
import numpy as np
import tensorflow as tf
import time
import scipy.io as sio
import cv2
from decimal import Decimal
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import logging

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
                                                       
class LSTM_SA_Trainer():
    
    def __init__(self, 
                 dataprovider_train, 
                 dataprovider_valid, 
                 log_path, 
                 model_path,
                 drop_rate=0.5,
                 batch_size=1, 
                 epochs=20, 
                 display_step=1000,
                 save_model=True,
                 load_model_path=None,
                 skip_val=False):
        
        self._dataprovider_train = dataprovider_train
        self._dataprovider_valid = dataprovider_valid
        if dataprovider_train.background_mask_index() != dataprovider_valid.background_mask_index():
            logging.info("background_mask_index Error.")
            exit()
        self._background_mask_index = dataprovider_train.background_mask_index()
        
        self._log_path = log_path
        self._model_path = model_path
        self._drop_rate = drop_rate
        self._batch_size = batch_size
        self._epochs = epochs
        
        self._iterations_per_epoch = dataprovider_train.dataset_length() // batch_size
        self._iterations_train = self._iterations_per_epoch * epochs
        self._iterations_valid = dataprovider_valid.dataset_length() // batch_size

        self._display_step = display_step
        self._save_model = save_model
        self._load_model_path = load_model_path
        self._skip_val = skip_val
        
    def train(self, lstm_net):
        
        net = lstm_net
        
        # =================================== log folders =====================================
        _names = net.get_architecture_name() + self.get_trainer_name()
        self._log_path = "{}{}/".format(self._log_path, _names)
        self._model_path = "{}{}/".format(self._model_path, _names)
        if not self._load_model_path:
            clear_folders(self._log_path, self._model_path)
        
        # =================================== create net ======================================
        logits, x_input, y_input, dr_input, global_step = net.create_net()
        loss = net.get_loss(logits, y_input)
        optimizer = net.get_optimizer(loss, global_step)
        
        # prediction
        with tf.name_scope("prediction"):
            prediction = tf.nn.softmax(logits)

        with tf.name_scope("summary"):
            # ============================ performance metrics =================================
            val_pred = prediction
            val_y_input = y_input
            if net._midseq:
                val_pred = tf.slice(prediction, (0, 4, 0, 0, 0), (1, 2, net._xsize[0], net._xsize[1], net._nclasses))
                val_y_input = tf.slice(y_input, (0, 4, 0, 0, 0), (1, 2, net._ysize[0], net._ysize[1], net._nclasses))

            performance_pixel_error = metric_pixel_error(val_pred, val_y_input)
            performance_pa = metric_pixel_accuraccy(val_pred, val_y_input)
            performance_mean_iou = metric_mean_iou(val_pred, val_y_input, net._nclasses)
            performance_mpa = metric_mean_pa(val_pred, val_y_input, net._nclasses)
            performance_dice = dice(val_pred, val_y_input, net._nclasses)
            
            performance_dice_classes = []
            for i in range(net._nclasses):
                performance_dice_classes.append(dice(val_pred, val_y_input, net._nclasses, class_index=i))

            # internal use
            self._performance_pixel_error = performance_pixel_error
            self._performance_pa = performance_pa
            self._performance_mean_iou = performance_mean_iou
            self._performance_mpa = performance_mpa
            self._performance_dice = performance_dice

            self._performance_dice_classes = performance_dice_classes
            
            # ======================= scalar summaries for training ============================
            tf.summary.scalar("train_loss", loss)
            tf.summary.scalar("train_pixel_error", performance_pixel_error)
            tf.summary.scalar("train_dice", performance_dice)

            train_merged_summary = tf.summary.merge_all()

            # ============================== val summaries ===================================
            val_pixel_error_ph = tf.placeholder(tf.float32, shape=None)
            val_pa_ph = tf.placeholder(tf.float32, shape=None)
            val_mean_iou_ph = tf.placeholder(tf.float32, shape=None)
            val_mpa_ph = tf.placeholder(tf.float32, shape=None)
            val_dice_ph = tf.placeholder(tf.float32, shape=None)
            val_loss_ph = tf.placeholder(tf.float32, shape=None)

            val_dice_classes_ph = []
            for i in range(net._nclasses):
                val_dice_classes_ph.append(tf.placeholder(tf.float32, shape=None))
            
            # internal use
            self._val_pixel_error_ph = val_pixel_error_ph
            self._val_pa_ph = val_pa_ph
            self._val_mean_iou_ph = val_mean_iou_ph
            self._val_mpa_ph = val_mpa_ph
            self._val_dice_ph = val_dice_ph
            self._val_loss_ph = val_loss_ph

            self._val_dice_classes_ph = val_dice_classes_ph

            val_pixel_error_summary = tf.summary.scalar("val_pixel_error", val_pixel_error_ph)
            val_pa_summary = tf.summary.scalar("val_pa", val_pa_ph)
            val_mean_iou_summary = tf.summary.scalar("val_mean_iou", val_mean_iou_ph)
            val_mpa_summary = tf.summary.scalar("val_mpa", val_mpa_ph)
            val_dice_summary = tf.summary.scalar("val_dice", val_dice_ph)
            val_loss_summary = tf.summary.scalar('val_loss', val_loss_ph)

            merged = [val_pixel_error_summary, val_pa_summary, val_mean_iou_summary, val_mpa_summary, val_dice_summary, val_loss_summary]

            for i in range(net._nclasses):
                val_dc_sum = tf.summary.scalar('val_dice_class_' + str(i), val_dice_classes_ph[i])
                merged.append(val_dc_sum)
            
            val_merged_summary = tf.summary.merge(merged)
        
        # =============================== for internal methods ===============================
        self._x_input = x_input
        self._y_input = y_input
        self._dr_input = dr_input
        self._loss = loss
        self._logits = logits
        self._prediction = prediction
        self._global_step = global_step

        # ================================== start training ===================================       
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            
            if self._load_model_path:
                model_restore(sess, self._load_model_path)
            
            # summary writer
            train_writer = tf.summary.FileWriter(self._log_path + "/train")
            train_writer.add_graph(sess.graph)
            
            val_writer = tf.summary.FileWriter(self._log_path + "/val")

            # ==================================== val =======================================
            if not self._skip_val:
                self._validate(net, sess, 0, val_writer, val_merged_summary)
            
            # ==================================== train =====================================
            logging.info("Starting Training with {} Iterations (Batch Size={}, Epochs={}).".format(self._iterations_train, self._batch_size, self._epochs))
            for step in range(1, self._iterations_train+1):
                
                batch_x, batch_y = self._dataprovider_train.next_batch(self._batch_size)

                feed_dict = {x_input: batch_x, y_input: batch_y, dr_input: self._drop_rate}
                sess.run(optimizer, feed_dict=feed_dict)
                
                # ============================ summary writer =================================
                if step % 100 == 0:
                    summ = sess.run(train_merged_summary, feed_dict=feed_dict)
                    train_writer.add_summary(summ, step)
                
                # ============================== display step ================================
                if step % self._display_step == 0:
                    self._output_batch_stats(sess, batch_x, batch_y, step)
                    
                # ========================= epoch finished, val ==============================
                if step % self._iterations_per_epoch == 0:
                    # val
                    if not self._skip_val:
                        self._validate(net, sess, step, val_writer, val_merged_summary)
                    
            logging.info("Done.")
    
    def _validate(self, net, sess, cur_step, writer, summary):
        
        # ================================== start validate ===================================
        logging.info("Starting Validation.")
        
        # manual metric computation
        performance_dice = []
        performance_pe_list = []
        performance_pa_list = []
        performance_mpa_list = []
        performance_miou_list = []
        performance_loss_list = []
        performance_dice_classes_list = []

        for i in range(net._nclasses):
            performance_dice_classes_list.append([])
        
        for step in range(self._iterations_valid):

            batch_x, batch_y = self._dataprovider_valid.next_batch(self._batch_size)
                
            feed_dict = {self._x_input: batch_x, self._y_input: batch_y, self._dr_input: 0.0}
            pred = sess.run(self._prediction, feed_dict)
            
            # ============================= metric computation  ==============================
            pred, dice_acc, pe, pa, mpa, miou, loss = sess.run([self._prediction, 
                                                      self._performance_dice,
                                                      self._performance_pixel_error,
                                                      self._performance_pa,
                                                      self._performance_mpa,
                                                      self._performance_mean_iou,
                                                      self._loss], feed_dict=feed_dict)
            
            dice_classes = sess.run(self._performance_dice_classes, feed_dict=feed_dict)
            for i in range(net._nclasses):
                performance_dice_classes_list[i].append(dice_classes[i])

            # manual metric computation
            performance_dice.append(dice_acc)
            performance_pe_list.append(pe)
            performance_pa_list.append(pa)
            performance_mpa_list.append(mpa)
            performance_miou_list.append(miou)
            performance_loss_list.append(loss)
            
            # ================================= save model ===================================
            if self._save_model and step == self._iterations_valid-1:
                self._save_current_model(sess, batch_x, batch_y, pred)
                print()
                
        # =================================== summary metrics ================================
        val_pixel_error = np.mean(performance_pe_list) 
        val_pa = np.mean(performance_pa_list)
        val_mean_iou = np.mean(performance_miou_list) 
        val_mpa = np.mean(performance_mpa_list) 
        val_dice = np.mean(performance_dice)
        val_loss = np.mean(performance_loss_list)
        val_dice_classes = [np.mean(item) for item in performance_dice_classes_list]
                
        feed_dict={self._val_pixel_error_ph: val_pixel_error,
                   self._val_pa_ph: val_pa,
                   self._val_mean_iou_ph: val_mean_iou,
                   self._val_mpa_ph: val_mpa,
                   self._val_dice_ph: val_dice,
                   self._val_loss_ph: val_loss}
        
        for i in range(net._nclasses):
            feed_dict[self._val_dice_classes_ph[i]] = val_dice_classes[i]

        summ = sess.run(summary, feed_dict)
        writer.add_summary(summ, cur_step)

        logging.info("Validation Pixel Error: {:.8f}, mIoU: {:.8f}, PA: {:.8f}, mPA: {:.8f}, Dice: {:.8f}, Loss: {:.8f}.".format(val_pixel_error, val_mean_iou, val_pa, val_mpa, val_dice, val_loss))
        
    def _output_batch_stats(self, sess, batch_x, batch_y, step):
        feed_dict = {self._x_input: batch_x, self._y_input: batch_y}
        loss_value, acc_pw = sess.run([self._loss, self._performance_pixel_error], feed_dict) 
        logging.info("Iteration {}, Batch Loss {}, Pixel Error {}.".format(step, loss_value, acc_pw))
        
    def _save_current_model(self, sess, x, y, p):
        # round prediction -> 0,1
        p = np.round(p)
        xxsize, xysize, xch = x.shape[2], x.shape[3], x.shape[4]
        yxsize, yysize, ych = y.shape[2], y.shape[3], y.shape[4]
        x = np.reshape(x, [-1, xxsize, xysize, xch])
        y = np.reshape(y, [-1, yxsize, yysize, ych])
        p = np.reshape(p, [-1, yxsize, yysize, ych])
        save_img_prediction(x, y, p, self._model_path, sess.run(self._global_step), background_mask=self._background_mask_index)
        model_save(sess, self._model_path, self._global_step)
        
    def get_trainer_name(self):
        return "_BS{}_E{}".format(self._batch_size, self._epochs)

class LSTM_SA_Tester():
    
    def __init__(self,
                dataprovider,
                net,
                model_path,
                output_path,
                verbose=False):
                
        self._dataprovider = dataprovider
        self._background_mask_index = dataprovider.background_mask_index()
        self._net = net
        self._model_path = model_path
        self._output_path = output_path
        self._verbose = verbose
        
        self._mask_nr = 1
                
    def _create_graph(self):
        
        g = tf.Graph()
        with g.as_default(): 
            
            logits, x_input, y_input, drop_rate, global_step = self._net.create_net()
            loss = self._net.get_loss(logits, y_input)
            optimizer = self._net.get_optimizer(loss, global_step)

            prediction = tf.nn.softmax(logits)
            prediction = tf.round(prediction)
            
            performance_pixel_error = metric_pixel_error(prediction, y_input)
            performance_mean_iou = metric_mean_iou(prediction, y_input, self._net._nclasses)
            performance_pa = metric_pixel_accuraccy(prediction, y_input)
            performance_mpa = metric_mean_pa(prediction, y_input, self._net._nclasses)
            performance_dice = dice(prediction, y_input, self._net._nclasses)
            self._performance_pe_list = []
            self._performance_miou_list = []
            self._performance_pa_list = []
            self._performance_mpa_list = []
            self._performance_d_list = []

            return g, x_input, y_input, drop_rate, prediction, performance_pixel_error, performance_mean_iou, performance_pa, performance_mpa, performance_dice
    
    def _image_overlay(self, img, pred, background_mask=None):
    # pred is the mask obtained from your model (one-hot encoded or binary)
        overlay = img.copy()
        
        if background_mask is not None:
            pred = pred[..., :background_mask] + pred[..., background_mask+1:]
        
        overlay[pred > 0] = (255, 0, 0)  # Overlaying with red color where mask has objects
        
        return overlay

    def _save_coordinates(self, coords, batch_nr, image_nr):
        path = self._output_path + "coordinates_batch_{}_image_{}.txt".format(batch_nr, image_nr)
        with open(path, "w") as file:
            for coord in coords:
                file.write('{}, {}, {}, {}\n'.format(*coord))
    
    def get_object_coordinates(self, mask):
        # The mask is a binary image (0 for background, 1 for object)
        # Find contours in the mask
        mask = (mask.astype(np.uint8) * 255)
        if len(mask.shape) == 3:  # if the mask is 3D,
            mask = mask[:, :, 0]  # use only the first channel
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        # List to hold coordinates of objects
        object_coordinates = []
            
        for contour in contours:
            # Get bounding box coordinates for each contour
            x, y, w, h = cv2.boundingRect(contour)
            object_coordinates.append((x, y, w, h))
            
        return object_coordinates
    
    def test(self, batch_size=1, save_validate_image=False):
        
        # runtime
        _runtime_load_batch = np.asarray([])
        _runtime_feedforward_batch = np.asarray([])
        _runtime_save_batch = np.asarray([])
        _runtime_t1 = time.time() # runtime
        
        g, xi, yi, dr, p, ppe, pmiou, ppa, pmpa, pd = self._create_graph()
        with tf.Session(graph=g) as sess:
            
            model_restore(sess, self._model_path)
            
            _runtime_t2 = time.time() # runtime
            print(f"dataset length: {self._dataprovider.dataset_length()}; Batch size : {batch_size}")
            for i in range(self._dataprovider.dataset_length() // batch_size):
                #range(self._dataprovider.dataset_length() // batch_size)
                _runtime_t3 = time.time() # runtime
                
                x, y = self._dataprovider.next_batch(batch_size)
                
                _runtime_t4 = time.time() # runtime
                
                feed_dict = {xi: x, yi: y, dr:0.}         
                out = sess.run([p, ppe, pmiou, ppa, pmpa, pd], feed_dict)
                out_p, out_ppe, out_pmiou, out_ppa, out_pmpa, out_pd = out

                # Adding visualization here
                for j in range(len(out_p)):  # For each image in the batch
                    print(out_p.shape)
                    mask = np.argmax(out_p[j], axis=-1)
                    coords = self.get_object_coordinates(mask)

                    fig, ax = plt.subplots(1, 1)
                    if x[j].shape[-1] == 1:  # if grayscale
                        ax.imshow(np.squeeze(x[j][0]), cmap='gray')
                    else:  # if RGB
                        ax.imshow(x[j][0])

                    for coord in coords:
                        rect = patches.Rectangle((coord[0], coord[1]), coord[2], coord[3], linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                    
                    plt.savefig(f"{self._output_path}image_batch_{i}_idx_{j}.png")  # Saving the figure
                    plt.close(fig)  # Close the figure to free up memory

                _runtime_t5 = time.time() # runtime
                
                self._add_performance(out_ppe, out_pmiou, out_ppa, out_pmpa, out_pd)
                #self._save_mask(out_p)
                if save_validate_image:
                    self._save_image(x, y, out_p, i)
                
                _runtime_t6 = time.time() # runtime
                
                # runtime
                _runtime_load_batch = np.append(_runtime_load_batch, _runtime_t4 - _runtime_t3)
                _runtime_feedforward_batch = np.append(_runtime_feedforward_batch, _runtime_t5 - _runtime_t4)
                _runtime_save_batch = np.append(_runtime_save_batch, _runtime_t6 - _runtime_t5)
                
        # runtime
        _runtime_t7 = time.time() # runtime
        _runtime_restore_model = _runtime_t2 - _runtime_t1
        _runtime_total = _runtime_t7 - _runtime_t1
        logging.info("Runtime: Restore Model: {:.4f}".format(_runtime_restore_model))
        logging.info("Runtime: Load Batch: Sum {:.4f} Mean {:.4f} Std {:.4f}".format(np.sum(_runtime_load_batch), np.mean(_runtime_load_batch), np.std(_runtime_load_batch)))
        logging.info("Runtime: Feedforward Batch: Sum {:.4f} Mean {:.4f} Std. {:.4f}".format(np.sum(_runtime_feedforward_batch), np.mean(_runtime_feedforward_batch), np.std(_runtime_feedforward_batch)))
        logging.info("Runtime: Save Batch: Sum {:.4f} Mean {:.4f} Std {:.4f}".format(np.sum(_runtime_save_batch), np.mean(_runtime_save_batch), np.std(_runtime_save_batch)))
        logging.info("------------------------------------------------")
        logging.info("Runtime: Total Runtime: {:.4f}".format(_runtime_total))
        
        path = self._output_path + "runtime.mat"
        sio.savemat(path, {"lstm_sa_full_normal_load_batch":_runtime_load_batch,
                            "lstm_sa_full_normal_feedforward_batch":_runtime_feedforward_batch,
                            "lstm_sa_full_normal_save_batch":_runtime_save_batch})
                            
        perf = self._save_performance()
        logging.info(perf)
        logging.info("Done.")
        
    def _add_performance(self, ppe, pmiou, ppa, pmpa, pd):
        self._performance_pe_list.append(ppe)
        self._performance_miou_list.append(pmiou)
        self._performance_pa_list.append(ppa)
        self._performance_mpa_list.append(pmpa)
        self._performance_d_list.append(pd)
    
    def _compute_performance(self):
        r_pe = np.mean(self._performance_pe_list)
        r_miou = np.mean(self._performance_miou_list)
        r_pa = np.mean(self._performance_pa_list)
        r_mpa = np.mean(self._performance_mpa_list)
        r_d = np.mean(self._performance_d_list)
        
        print(r_pe, r_miou, r_pa, r_mpa, r_d)
        return r_pe, r_miou, r_pa, r_mpa, r_d
        
    def _save_performance(self):
        path = self._output_path + "performance.txt"
        r_pe, r_miou, r_pa, r_mpa, r_d = self._compute_performance()
        performance = "Pixel Error: {}, mIoU: {}, Pixel Accuracy: {}, mPA: {}, Dice: {}.".format(r_pe, r_miou, r_pa, r_mpa, r_d)
        
        file = open(path, "w") 
        file.write(performance) 
        file.close() 
        
        return performance

    def _save_image(self, x, y, p, nr):
        xxsize, xysize, xch = x.shape[2], x.shape[3], x.shape[4]
        yxsize, yysize, ych = y.shape[2], y.shape[3], y.shape[4]
        x = np.reshape(x, [-1, xxsize, xysize, xch])
        y = np.reshape(y, [-1, yxsize, yysize, ych])
        p = np.reshape(p, [-1, yxsize, yysize, ych])
        save_img_prediction(x, y, p, self._output_path, image_name=str(nr), background_mask=self._background_mask_index)

    def _save_mask(self, p):
        yxsize, yysize, ych = p.shape[2], p.shape[3], p.shape[4]
        p = np.reshape(p, [-1, yxsize, yysize, ych])
        for i in range(p.shape[0]):
            image_name = str(self._mask_nr).zfill(5)
            save_img_mask(p[i], self._output_path, image_name=image_name, background_mask=self._background_mask_index)
            self._mask_nr = self._mask_nr + 1
            
class LSTM_SA_Tester_MidSeq():
    
    def __init__(self,
                dataprovider,
                net,
                model_path,
                output_path,
                verbose=False):
                
        self._dataprovider = dataprovider
        self._background_mask_index = dataprovider.background_mask_index()
        self._net = net
        self._model_path = model_path
        self._output_path = output_path
        self._verbose = verbose
        
        self._mask_nr = 1
                
    def _create_graph(self):
        
        g = tf.Graph()
        with g.as_default(): 
            
            logits, x_input, y_input, drop_rate, global_step = self._net.create_net()
            loss = self._net.get_loss(logits, y_input)
            optimizer = self._net.get_optimizer(loss, global_step)

            prediction = tf.nn.softmax(logits)
            prediction = tf.round(prediction)
            
            # Mid.Seq.
            _prediction = tf.slice(prediction, [0, 4, 0, 0, 0], [prediction.shape[0], 2, prediction.shape[2], prediction.shape[3], prediction.shape[4]])
            _y_input = tf.slice(y_input, [0, 4, 0, 0, 0], [y_input.shape[0], 2, y_input.shape[2], y_input.shape[3], y_input.shape[4]])
            
            performance_pixel_error = metric_pixel_error(_prediction, _y_input)
            performance_mean_iou = metric_mean_iou(_prediction, _y_input, self._net._nclasses)
            performance_pa = metric_pixel_accuraccy(_prediction, _y_input)
            performance_mpa = metric_mean_pa(_prediction, _y_input, self._net._nclasses)
            performance_dice = dice(_prediction, _y_input, self._net._nclasses)
            self._performance_pe_list = []
            self._performance_miou_list = []
            self._performance_pa_list = []
            self._performance_mpa_list = []
            self._performance_d_list = []

            return g, x_input, y_input, drop_rate, _prediction, performance_pixel_error, performance_mean_iou, performance_pa, performance_mpa, performance_dice
    
    def test(self, save_validate_image=False):

        # runtime
        _runtime_load_batch = np.asarray([])
        _runtime_feedforward_batch = np.asarray([])
        _runtime_save_batch = np.asarray([])
        _runtime_t1 = time.time() # runtime
        
        g, xi, yi, dr, p, ppe, pmiou, ppa, pmpa, pd = self._create_graph()
        with tf.Session(graph=g) as sess:
        
            model_restore(sess, self._model_path)
            
            _runtime_t2 = time.time() # runtime
                        
            nsequences = self._dataprovider.dataset_length() * 2 // 100
            for j in range(nsequences):
                
                _runtime_t3 = time.time() # runtime
                
                _x1, _y1 = self._dataprovider.next_batch(1)
                _x2, _y2 = self._dataprovider.next_batch(1)
                _x3, _y3 = self._dataprovider.next_batch(1)
                
                x = np.concatenate([_x3, _x2, _x1, _x2, _x3], axis=1)
                y = np.concatenate([_y3, _y2, _y1, _y2, _y3], axis=1)
                
                _runtime_t4 = time.time() # runtime
                
                for i in range(50):
                    if i != 0:
                    
                        _runtime_t3 = time.time() # runtime
                        
                        if i > 47:
                            _x = x[:,6:8,...]
                            _y = y[:,6:8,...]
                        else:
                            _x, _y = self._dataprovider.next_batch(1)
                        
                        x = x[:,2:,...]
                        y = y[:,2:,...]
                        
                        x = np.concatenate([x, _x], axis=1)
                        y = np.concatenate([y, _y], axis=1)
                    
                        _runtime_t4 = time.time() # runtime
                    
                    feed_dict = {xi: x, yi: y, dr:0.}         
                    out = sess.run([p, ppe, pmiou, ppa, pmpa, pd], feed_dict)
                    out_p, out_ppe, out_pmiou, out_ppa, out_pmpa, out_pd = out
                    
                    _runtime_t5 = time.time() # runtime
                    
                    self._add_performance(out_ppe, out_pmiou, out_ppa, out_pmpa, out_pd)
                    self._save_mask(out_p)
                    
                    _runtime_t6 = time.time() # runtime
                    
                    # runtime
                    _runtime_load_batch = np.append(_runtime_load_batch, _runtime_t4 - _runtime_t3)
                    _runtime_feedforward_batch = np.append(_runtime_feedforward_batch, _runtime_t5 - _runtime_t4)
                    _runtime_save_batch = np.append(_runtime_save_batch, _runtime_t6 - _runtime_t5)
                    
                    if save_validate_image:
                        output_x = x[:,4:6,...]
                        output_y = y[:,4:6,...]

                        if i == 0:
                            #init
                            sequence_output_x = output_x
                            sequence_output_y = output_y
                            sequence_output_p = out_p
                        else:
                            sequence_output_x = np.concatenate([sequence_output_x, output_x], axis=1)
                            sequence_output_y = np.concatenate([sequence_output_y, output_y], axis=1)
                            sequence_output_p = np.concatenate([sequence_output_p, out_p], axis=1)
                
                if save_validate_image:
                    self._save_image(sequence_output_x, sequence_output_y, sequence_output_p, j)

        # runtime
        _runtime_t7 = time.time() # runtime
        _runtime_restore_model = _runtime_t2 - _runtime_t1
        _runtime_total = _runtime_t7 - _runtime_t1
        logging.info("Runtime: Restore Model: {:.4f}".format(_runtime_restore_model))
        logging.info("Runtime: Load 10/2: Sum {:.4f} Mean {:.4f} Std {:.4f}".format(np.sum(_runtime_load_batch), np.mean(_runtime_load_batch), np.std(_runtime_load_batch)))
        logging.info("Runtime: Feedforward Batch: Sum {:.4f} Mean {:.4f} Std. {:.4f}".format(np.sum(_runtime_feedforward_batch), np.mean(_runtime_feedforward_batch), np.std(_runtime_feedforward_batch)))
        logging.info("Runtime: Save 2: Sum {:.4f} Mean {:.4f} Std {:.4f}".format(np.sum(_runtime_save_batch), np.mean(_runtime_save_batch), np.std(_runtime_save_batch)))
        logging.info("------------------------------------------------")
        logging.info("Runtime: Total Runtime: {:.4f}".format(_runtime_total))
        
        # path = self._output_path + "runtime.mat"
        # sio.savemat(path, {"lstm_sa_full_midseq_load_batch":_runtime_load_batch,
        #                     "lstm_sa_full_midseq_feedforward_batch":_runtime_feedforward_batch,
        #                     "lstm_sa_full_midseq_save_batch":_runtime_save_batch})
                            
        perf = self._save_performance()
        logging.info(perf)
        logging.info("Done.")
        
    def _add_performance(self, ppe, pmiou, ppa, pmpa, pd):
        self._performance_pe_list.append(ppe)
        self._performance_miou_list.append(pmiou)
        self._performance_pa_list.append(ppa)
        self._performance_mpa_list.append(pmpa)
        self._performance_d_list.append(pd)
    
    def _compute_performance(self):
        r_pe = np.mean(self._performance_pe_list)
        r_miou = np.mean(self._performance_miou_list)
        r_pa = np.mean(self._performance_pa_list)
        r_mpa = np.mean(self._performance_mpa_list)
        r_d = np.mean(self._performance_d_list)
        
        print(r_pe, r_miou, r_pa, r_mpa, r_d)
        return r_pe, r_miou, r_pa, r_mpa, r_d
        
    def _save_performance(self):
        path = self._output_path + "performance.txt"
        r_pe, r_miou, r_pa, r_mpa, r_d = self._compute_performance()
        performance = "Pixel Error: {}, mIoU: {}, Pixel Accuracy: {}, mPA: {}, Dice: {}.".format(r_pe, r_miou, r_pa, r_mpa, r_d)
        
        file = open(path, "w") 
        file.write(performance) 
        file.close() 
        
        return performance

    def _save_image(self, x, y, p, nr):
        xxsize, xysize, xch = x.shape[2], x.shape[3], x.shape[4]
        yxsize, yysize, ych = y.shape[2], y.shape[3], y.shape[4]
        x = np.reshape(x, [-1, xxsize, xysize, xch])
        y = np.reshape(y, [-1, yxsize, yysize, ych])
        p = np.reshape(p, [-1, yxsize, yysize, ych])
        save_img_prediction(x, y, p, self._output_path, image_name=str(nr), background_mask=self._background_mask_index)
    
    def _save_mask(self, p):
        yxsize, yysize, ych = p.shape[2], p.shape[3], p.shape[4]
        p = np.reshape(p, [-1, yxsize, yysize, ych])
        for i in range(p.shape[0]):
            image_name = str(self._mask_nr).zfill(5)
            save_img_mask(p[i], self._output_path, image_name=image_name, background_mask=self._background_mask_index)
            self._mask_nr = self._mask_nr + 1
