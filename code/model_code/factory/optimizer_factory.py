import tensorflow as tf

import abstract_factory as af

class OptimizerBuilder(af.AbstractFactory):
    
    def __init__(self, learning_rate, decay_rate):
        self._learning_rate = learning_rate
        self._decay_rate = decay_rate
    

    def create(self, loss, global_step):
        return self.__get_optimizer(loss, global_step)
    
    def __get_optimizer(self, loss, global_step):
        
        with tf.name_scope("optimizer"):
        
            if self._decay_rate != None:
                lr = tf.train.exponential_decay(self._learning_rate, global_step, self._decay_rate, 0.95, staircase=True)
            else:
                lr = self._learning_rate
                
            tf.summary.scalar("learning_rate", lr)

            optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)
            return optimizer
