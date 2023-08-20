import abc 


class AbstractFactory(abc.ABC):

    def __init__(self):
        pass
   
    @abc.abstractmethod 
    def create(self):
        pass