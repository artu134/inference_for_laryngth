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

    def _image_overlay_opposite(self, image, mask, background_mask=None):
        if len(image.shape) == 2:  # if grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        overlay = image.copy()
        color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Change colors for different masks
    
        mask = mask.astype(bool)
        border = mask ^ binary_erosion(mask).astype(bool)  # Finding the border of the mask

        # Only apply overlay if border is correctly formed
        if np.any(border):
            overlay[border] = (255, 0, 0)  # Overlaying with different color for each mask
        else:
            print(f"Skipping overlay for mask due to incorrectly formed border.")

        return overlay
    
    def _get_the_stream(self, video_path = "./predictions/inferenced_",
                         frame_size = (256, 256),
                         fps = 5.0) -> cv2.VideoWriter:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video_path = f"{video_path}segmented_video_new.mp4"  # Set this to the expected width and height of your frames
        out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
        return out

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

    def get_predicted_class_for_masks(self, out_p):
        for j in range(len(out_p)):  # For each image in the batch
            for k in range(out_p[j].shape[0]):  # For each predicted map in the image
                # Aggregate the softmax maps into one map
                aggregated_map = np.argmax(out_p[j], axis=-1)
                # Get the class with the highest frequency of being the most probable
                predicted_class = np.bincount(aggregated_map.flatten()).argmax()
                print(f"Prediction {k} in Image {j} has predicted class: {predicted_class}")
                # Check if the predicted class is the background class (assuming it's class 0)
                if predicted_class == 0:
                    print(f"Prediction {k} in Image {j} does not contain a glottis.")


    
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
            vid_stream = self._get_the_stream(video_path=f"{self._output_path}inferenced_with_bounding_box_", fps=5.0)
            bound_stream = self._get_the_stream(video_path=f"{self._output_path}inferenced_with_red_", fps=5.0)
            print(f"dataset length: {self._dataprovider.dataset_length()}; Batch size : {batch_size}")
            for i in range(40):
                #range(self._dataprovider.dataset_length() // batch_size)
                _runtime_t3 = time.time() # runtime
                
                x, y = self._dataprovider.next_batch(batch_size)
                
                logging.info("X shape: {}".format(x.shape))
                logging.info("Y shape: {}".format(y.shape))
                
                _runtime_t4 = time.time() # runtime
                
                feed_dict = {xi: x, yi: y, dr:0.}         
                out = sess.run([p, ppe, pmiou, ppa, pmpa, pd], feed_dict)
                out_p, out_ppe, out_pmiou, out_ppa, out_pmpa, out_pd = out

                # Adding visualization here
                print("Shape of out_p: ", np.shape(out_p))
                #self.get_predicted_class_for_masks(out_p)

                for j in range(len(out_p)):  # For each image in the batch

                    for k in range(out_p[j][0].shape[-1]):  # For each mask in the image
                        mask = out_p[j][0][..., k]
                        coords = self.get_object_coordinates(mask)

                        # Create an overlay image
                        original_with_overlay = x[j][0].copy()
                        original_with_overlay_red = self._image_overlay_opposite(x[j][0], mask, background_mask=None)
                        print(f"K {k}; J {j}; I {i}")
                        middle_coord = coords[len(coords)//2]
                        coord = middle_coord
                        #for coord in coords:
                            # Draw bounding boxes on the overlay image
                        if k == 2:
                            cv2.rectangle(original_with_overlay, (coord[0], coord[1]), 
                                        (coord[0] + coord[2], coord[1] + coord[3]), 
                                        (0, 255, 0), 2)

                        # Save the overlay image
                        original_with_overlay_bgr = cv2.cvtColor(original_with_overlay, cv2.COLOR_RGB2BGR)
                        original_with_overlay_red = cv2.cvtColor(original_with_overlay_red, cv2.COLOR_RGB2BGR)
                        vid_stream.write(original_with_overlay_bgr)
                        bound_stream.write(original_with_overlay_red)
                        #Image.fromarray(original_with_overlay).save(f"{self._output_path}image_batch_{i}_idx_{j}_mask_{k}.png")

                _runtime_t5 = time.time() # runtime
                
                self._add_performance(out_ppe, out_pmiou, out_ppa, out_pmpa, out_pd)
                #self._save_mask(out_p)
                #if save_validate_image:
                #    self._save_image(x, y, out_p, i)
                
                _runtime_t6 = time.time() # runtime
                
                # runtime
                _runtime_load_batch = np.append(_runtime_load_batch, _runtime_t4 - _runtime_t3)
                _runtime_feedforward_batch = np.append(_runtime_feedforward_batch, _runtime_t5 - _runtime_t4)
                _runtime_save_batch = np.append(_runtime_save_batch, _runtime_t6 - _runtime_t5)
            
            vid_stream.release()
            bound_stream.release()
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
            
