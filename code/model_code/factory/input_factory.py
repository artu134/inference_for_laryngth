import tensorflow as tf
import abstract_factory as af


class InputProcessingFactory(af.AbstractFactory):
    def __init__(self, config):
        self.config = config


    def create(self):
        with tf.name_scope("inputs"):
            res, res1, res2, res3 = self.process_input(), self.process_labels(), self.process_drop_rate(), self.process_global_step()
        return res, res1, res2, res3
    
    def process_input(self):
        # Define the input placeholder
        x_input = tf.placeholder(tf.float32, [self.config['_nbatches'], self.config['_nsequences'], self.config['_xsize'][0], self.config['_xsize'][1], self.config['_ndimensions']])
        
        # Standardize the input if needed
        in_node = x_input
        if self.config['_image_std']:
            shape = in_node.shape
            in_node = tf.reshape(in_node, [-1, shape[2], shape[3], shape[4]])
            in_node = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), in_node, dtype=tf.float32)
            in_node = tf.reshape(in_node, [-1, shape[1], shape[2], shape[3], shape[4]])
        return in_node
    
    def process_labels(self):
        # Define the label placeholder
        y_input = tf.placeholder(tf.float32, [self.config['_nbatches'], self.config['_nsequences'], self.config['_ysize'][0], self.config['_ysize'][1], self.config['_nclasses']])
        return y_input
    
    def process_drop_rate(self):
        # Define the drop rate placeholder
        drop_rate_input = tf.placeholder(tf.float32)
        return drop_rate_input
    
    def process_global_step(self):
        # Define the global step variable
        global_step = tf.Variable(0, trainable=False)
        return global_step
