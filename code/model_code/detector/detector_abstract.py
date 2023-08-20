import abc 

class DetectorAbstract(abc.ABC):
    
        def __init__(self):
            pass
        
        @abc.abstractmethod 
        def detect(self, image, mask) -> (list, list):
            pass