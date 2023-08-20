import layer 
from layer import conv2d_from_3d, max_pool2d_from_3d, deconv2d_from_3d, crop_concat_from_3d, bclstm, bcgru
import logging
import tensorflow as tf
import abstract_factory as af


class ConvFactory(af.AbstractFactory):
    def __init__(self, nlayers, nfilters, ndimensions, nsequences, nclasses, bn, reg, 
                 reg_scale, image_std, crop_concat, constant_nfilters, gru, midseq, verbose, two_gpus):
        self._nlayers = nlayers
        self._nfilters = nfilters
        self._ndimensions = ndimensions
        self._nsequences = nsequences
        self._nclasses = nclasses
        self._bn = bn
        self._reg = reg
        self._reg_scale = reg_scale
        self._image_std = image_std
        self._crop_concat = crop_concat
        self._constant_nfilters = constant_nfilters
        self._gru = gru
        self._midseq = midseq
        self._verbose = verbose
        self._two_gpus = two_gpus
    
    def create(self, in_node, drop_rate_input):
        in_node, down_nodes = self.__down_sample(in_node, drop_rate_input)
        output_node = self.__up_sample(in_node, drop_rate_input, down_nodes)                  

        return output_node

    def __down_sample(self, in_node, drop_rate_input):
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
        return in_node,down_nodes

    def __up_sample(self, in_node, drop_rate_input, down_nodes):
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
        return output_node
