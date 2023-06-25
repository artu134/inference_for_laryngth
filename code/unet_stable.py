#!/bin/python
import numpy as np
import tensorflow as tf
import time
import scipy.io as sio

from decimal import Decimal

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.WARNING)

from util import *
from layer import * 

class UNet_Architecture():

    def __init__(self,
                 xsize=(256, 256),
                 ysize=(256, 256),
                 ndimension=3,
                 nfilters=64,
                 reduced_nfilters = None,
                 nlayers=6,
                 nclasses=4,
                 loss_function="dice",
                 optimizer="adam",
                 class_weights=None,
                 learning_rate=0.0001,
                 decay_rate=None,
                 bn=True,
                 reg=None,
                 reg_scale=0.00001,
                 image_std=True,
                 crop_concat=True,
                 constant_nfilters=False,
                 name=None,
                 verbose=False):
        
        tf.reset_default_graph()

        self._xsize = xsize
        self._ysize = ysize
        self._ndimension = ndimension
        self._nfilters = nfilters
        self._reduced_nfilters = reduced_nfilters
        self._nclasses = nclasses
        self._nlayers = nlayers
        self._loss_function = loss_function
        self._optimizer = optimizer
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
        
        self._custom_nfilters = False
        
        if isinstance(self._nfilters, tuple):
            self._custom_nfilters = True
            if not len(self._nfilters) == nlayers:
                raise Exception("Length of nfilters must be same as nlayers or one single value which determines the starting nfilters.")
        
        if self._reduced_nfilters != None:
            if not isinstance(self._reduced_nfilters, tuple):
                raise Exception("reduced_nfilters must be a tuple of length: (nlayers*2)-1")
            if not len(self._reduced_nfilters) == nlayers:
                raise Exception("Length of reduced_nfilters must be same as nlayers.")
                    
        with tf.name_scope("regularizer"):
            if reg == None:
                self._regularizer = None
            elif reg == "L2":
                self._regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)
            else:
                raise Exception("Unknown Regularizer.")    
    
    def create_net(self):
        
        # cut off points for lstm use
        cut_offs = []
        
        # inputs
        with tf.name_scope("inputs"):
            x_input = tf.placeholder(tf.float32, [None, self._xsize[0], self._xsize[1], self._ndimension])
            in_node = x_input
            
            if self._image_std:
                in_node = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x_input, dtype=tf.float32)
            
            y_input = tf.placeholder(tf.float32, [None, self._ysize[0], self._ysize[1], self._nclasses])

            drop_rate_input = tf.placeholder(tf.float32)
            
            global_step = tf.Variable(0, trainable=False)

        # layers
        down_nodes = [None for _ in range(self._nlayers)]
        for layer in range(1, self._nlayers+1):
            if self._custom_nfilters:
                cur_nfilters = self._nfilters[layer-1]
            elif self._constant_nfilters:
                cur_nfilters = self._nfilters
            else:
                cur_nfilters = self._nfilters * 2**(layer-1)
            
            # down
            with tf.name_scope("downsampling_{}".format(str(layer))):
                conv1 = conv2d(in_node, cur_nfilters, drate=drop_rate_input, padding="same", bn=self._bn, regularizer=self._regularizer, verbose=self._verbose)
                conv2 = conv2d(conv1, cur_nfilters, drate=drop_rate_input, padding="same", bn=self._bn, regularizer=self._regularizer, verbose=self._verbose)
                if self._reduced_nfilters:
                    conv2 = conv2d(conv2, self._reduced_nfilters[layer-1], ksize=1, drate=drop_rate_input, padding="same", bn=self._bn, regularizer=self._regularizer, verbose=self._verbose)
                down_nodes[layer-1] = conv2
                in_node = conv2
                
                cut_offs.append(in_node)
                
                if layer < self._nlayers:
                    pool1 = max_pool2d(in_node, verbose=self._verbose)
                    in_node = pool1

        output_node = None
        for layer in range(self._nlayers-1, 0, -1):
            if self._custom_nfilters:
                cur_nfilters = self._nfilters[layer-1]
            elif self._constant_nfilters:
                cur_nfilters = self._nfilters
            else:
                cur_nfilters = self._nfilters * 2**(layer-1)

            # up
            with tf.name_scope("upsampling_{}".format(str(layer))):
                _node = None
                if self._crop_concat:
                    deconv1 = deconv2d(in_node, cur_nfilters, regularizer=self._regularizer, verbose=self._verbose)
                    cc1 = crop_concat(down_nodes[layer-1], deconv1)
                    _node = cc1
                    if self._constant_nfilters:
                        conv1x1 = tf.layers.conv2d(_node, cur_nfilters, 1, padding="same")
                        _node = conv1x1
                else:
                    deconv1 = deconv2d(in_node, cur_nfilters*2, regularizer=self._regularizer, verbose=self._verbose)
                    _node = deconv1
                conv1 = conv2d(_node, cur_nfilters, drate=drop_rate_input, padding="same", bn=self._bn, regularizer=self._regularizer, verbose=self._verbose)
                conv2 = conv2d(conv1, cur_nfilters, drate=drop_rate_input, padding="same", bn=self._bn, regularizer=self._regularizer, verbose=self._verbose)
                if self._reduced_nfilters:
                    conv2 = conv2d(conv2, self._reduced_nfilters[layer-1], ksize=1, drate=drop_rate_input, padding="same", bn=self._bn, regularizer=self._regularizer, verbose=self._verbose)
                in_node = conv2
                
                cut_offs.append(in_node)
                
                if layer == 1:
                    final_conv1 = tf.layers.conv2d(in_node, self._nclasses, 1, padding="same", activation=None)
                    if self._verbose: logging.info(final_conv1)
                    output_node = final_conv1
        
        return output_node, x_input, y_input, drop_rate_input, global_step, cut_offs
                    
    def get_loss(self, logits, labels):
        
        with tf.name_scope("loss"): 
            
            if self._loss_function == "softmax":  
            
                flat_logits = tf.reshape(logits, [-1, self._nclasses])
                flat_labels = tf.reshape(labels, [-1, self._nclasses])

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
                lr = tf.train.exponential_decay(self._learning_rate, global_step, self._decay_rate, 0.96, staircase=True)
            else:
                lr = self._learning_rate
            
            tf.summary.scalar("learning_rate", lr)

            if self._optimizer == "adam":
                optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
                
            elif self._optimizer == "rmsprop":
                optimizer = tf.train.RMSPropOptimizer(lr).minimize(loss, global_step=global_step)
                
            else:
                raise Exception("Unknown Optimizer.")
                
            return optimizer
        
    def get_architecture_name(self):
        lr = "%.2E" % Decimal(str(self._learning_rate))
        rs = "%.2E" % Decimal(str(self._reg_scale))
        suf = "" if self._name == None else "_{}".format(self._name)
        return "UNet_L{}_F{}_LF{}_O{}_LR{}_C{}_CW{}_DeR{}_BN{}_Reg{}_RegS{}_Std{}_CC{}_constF{}{}".format(self._nlayers, 
                                                       self._nfilters, 
                                                       self._loss_function, 
                                                       self._optimizer,
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
                                                       suf)
                                                       
class UNet_Trainer():
    
    def __init__(self, 
                 dataprovider_train, 
                 dataprovider_valid, 
                 log_path, 
                 model_path,
                 drop_rate=0.6,
                 batch_size=10, 
                 epochs=20, 
                 display_step=1000,
                 save_model=True,
                 load_model_path=None,
                 summary_images=False,
                 skip_val=False):
        
        self._dataprovider_train = dataprovider_train
        self._dataprovider_valid = dataprovider_valid

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
        self._summary_images = summary_images
        self._skip_val = skip_val
        
        self._complete_name = None
        
    def train(self, net):
        
        # =================================== log folders =====================================
        _names = net.get_architecture_name() + self.get_trainer_name()
        self._complete_name = _names
        self._log_path = "{}{}/".format(self._log_path, _names)
        self._model_path = "{}{}/".format(self._model_path, _names)
        if not self._load_model_path:
            clear_folders(self._log_path, self._model_path)
        
        # =================================== create net ======================================
        logits, x_input, y_input, dr_input, global_step, _ = net.create_net()
        loss = net.get_loss(logits, y_input)
        optimizer = net.get_optimizer(loss, global_step)
        
        # prediction
        with tf.name_scope("prediction"):
            prediction = tf.nn.softmax(logits)

        with tf.name_scope("summary"):
            # ============================ performance metrics =================================
            performance_pixel_error = metric_pixel_error(prediction, y_input)
            performance_pa = metric_pixel_accuraccy(prediction, y_input)
            performance_mean_iou = metric_mean_iou(prediction, y_input, net._nclasses)
            performance_mpa = metric_mean_pa(prediction, y_input, net._nclasses)
            performance_dice = dice(prediction, y_input, net._nclasses)

            performance_dice_classes = []
            for i in range(net._nclasses):
                performance_dice_classes.append(dice(prediction, y_input, net._nclasses, class_index=i))
            
            # internal use
            self._performance_pixel_error = performance_pixel_error
            self._performance_pa = performance_pa
            self._performance_mean_iou = performance_mean_iou
            self._performance_mpa = performance_mpa
            self._performance_dice = performance_dice
            
            self._performance_dice_classes = performance_dice_classes

            # ============================== scalar summaries =================================
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
        feed_dict = {self._x_input: batch_x, self._y_input: batch_y, self._dr_input: 0.0}
        loss_value, acc_pw, acc_dice = sess.run([self._loss, self._performance_pixel_error, self._performance_dice], feed_dict=feed_dict) 
        logging.info("Iteration {}, Batch Loss {}, Pixel Error {}, Dice {}.".format(step, loss_value, acc_pw, acc_dice))
        
    def _save_current_model(self, sess, val_batch_x, val_batch_y, val_batch_pred):
        # round prediction -> 0,1
        val_batch_pred = np.round(val_batch_pred)
        save_img_prediction(val_batch_x, val_batch_y, val_batch_pred, self._model_path, sess.run(self._global_step))
        model_save(sess, self._model_path, self._global_step)
    
    def get_name(self):
        return self._complete_name
    
    def get_trainer_name(self):
        return "_DR{}_BS{}_E{}".format(self._drop_rate, self._batch_size, self._epochs)

class UNet_Tester():
    
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
            
            logits, x_input, y_input, drop_rate, global_step, _ = self._net.create_net()
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
            
            nsequences = self._dataprovider.dataset_length() // 100
            for j in range(nsequences):
                
                for i in range(10):
                    
                    _runtime_t3 = time.time() # runtime
                    
                    x, y = self._dataprovider.next_batch(10)
                    
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
                        if i == 0:
                            #init
                            sequence_output_x = x
                            sequence_output_y = y
                            sequence_output_p = out_p
                        else:
                            sequence_output_x = np.concatenate([sequence_output_x, x], axis=1)
                            sequence_output_y = np.concatenate([sequence_output_y, y], axis=1)
                            sequence_output_p = np.concatenate([sequence_output_p, out_p], axis=1)
                        
                if save_validate_image:
                    self._save_image(sequence_output_x, sequence_output_y, sequence_output_p, j)
        
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
        sio.savemat(path, {"unet_load_batch":_runtime_load_batch,
                            "unet_feedforward_batch":_runtime_feedforward_batch,
                            "unet_save_batch":_runtime_save_batch})
                                
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
        save_img_prediction(x, y, p, self._output_path, image_name=str(nr), background_mask=self._background_mask_index)
        
    def _save_mask(self, p):
        yxsize, yysize, ych = p.shape[1], p.shape[2], p.shape[3]
        p = np.reshape(p, [-1, yxsize, yysize, ych])
        for i in range(p.shape[0]):
            image_name = str(self._mask_nr).zfill(5)
            save_img_mask(p[i], self._output_path, image_name=image_name, background_mask=self._background_mask_index)
            self._mask_nr = self._mask_nr + 1
            
class UNet_Runner():
    
    def __init__(self,
                 net,
                 load_model_path,
                 cut_off_point=None,
                 verbose=False):
        
        self._net = net
        self._load_model_path = load_model_path
        self._cut_off_point = cut_off_point
        self._verbose = verbose
        
        self._init = False
        
    def _create_graph(self):
        
        g = tf.Graph()
        with g.as_default(): 

            logits, x_input, y_input, dr_input, global_step, cut_offs = self._net.create_net()
            loss = self._net.get_loss(logits, y_input)
            optimizer = self._net.get_optimizer(loss, global_step)

            if self._verbose: logging.info("{} Cut-Off Points.".format(len(cut_offs)))

            if self._cut_off_point:
                output = cut_offs[self._cut_off_point]

            else:
                output = tf.nn.softmax(logits)
                output = tf.round(output)
        
            return g, x_input, output, dr_input
        
    def run_sticky_session(self, x):
        
        if not self._init:
            logging.info("Init. Sticky UNet Session.")
            g, i, o, dr = self._create_graph()
            self._g = g
            self._i = i
            self._o = o
            self._dr = dr
            
            with g.as_default():
                sess = tf.Session()
                model_restore(sess, self._load_model_path)
                self._sess = sess
                
            self._init = True
        
        with self._g.as_default():

            feed_dict = {self._i: x, self._dr: 0.0}
            out = self._sess.run(self._o, feed_dict)

        return out
    
    def close(self):
        if self._sess:
            self._sess.close()
            self._init = False
            logging.info("Closed Sticky UNet Session.")
        
    def run(self, x):
        
        g, i, o, dr = self._create_graph()

        with tf.Session(graph=g) as sess:

            model_restore(sess, self._load_model_path)

            feed_dict = {i: x, dr: 0.0}
            out = sess.run(o, feed_dict)

        return out
