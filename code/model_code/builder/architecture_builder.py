import abc
from typing import Dict, Optional 
import abstract_builder 
from abstract_builder import AbstractBuilder
from model_code.factory.abstract_factory import AbstractFactory
import tensorflow as tf

class ArchitectureBuilder(AbstractBuilder):
    """
        This class is responsible for building the architecture of the model.
        It uses the abstract factory pattern to create the input processing,
    """
    def __init__(self, input_processing: AbstractFactory, 
                 conv_layers: AbstractFactory, 
                 loss_function:AbstractFactory, 
                 optimizer: AbstractFactory):
        self.input_processing = input_processing
        self.conv_layers = conv_layers
        self.loss_function = loss_function
        self.optimizer = optimizer        
    
    def build(self):
       #with tf.name_scope("architecture"):
        input, label, drop_rate, global_step = self.input_processing.create() 
        logits = self.conv_layers.create(input, drop_rate)
        #loss = self.loss_function.create(logits, label)
        #optimizer = self.optimizer.create(loss, global_step)
        return input, label, drop_rate, global_step, logits,# loss, optimizer