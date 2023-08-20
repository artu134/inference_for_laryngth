import tensorflow as tf
from layer import sdc
import tensorflow as tf
import numpy as np
import abstract_factory as af

class LossFactory(af.AbstractFactory):
    
    def __init__(self, loss_function, class_weights, midseq, xsize, ysize, nclasses):
        self._loss_function = loss_function
        self._class_weights = class_weights
        self._midseq = midseq
        self._xsize = xsize
        self._ysize = ysize
        self._nclasses = nclasses
    
    def create(self, logits, labels):
        return self.__get_loss(logits, labels)

    def __get_loss(self, logits, labels):
        
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