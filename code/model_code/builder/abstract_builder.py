import abc
from typing import Dict, Optional 


class AbstractBuilder(abc.ABC):
    """
        This class is responsible for building the architecture of the model.
    """
    def __init__(self):
        pass

    @abc.abstractmethod 
    def build(self, **kwargs):
        pass