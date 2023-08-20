
from model_code.detector.detector_abstract import DetectorAbstract
from skimage.morphology import binary_erosion

import cv2 
import numpy as np

class DetectorBounding(DetectorAbstract): 
        
    def __init__(self):
        pass
    
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


    
    def get_bouding_box(self, coords, original_with_overlay):
        """
            Get the bounding box of the middle mask in the image.
        """
        cop_orig = original_with_overlay.copy()
        middle_coord = coords[len(coords)//2]
        coord = middle_coord
        cv2.rectangle(cop_orig, (coord[0], coord[1]), 
                                    (coord[0] + coord[2], coord[1] + coord[3]), 
                                    (0, 255, 0), 2)
        return cop_orig

    def detect(self, image, mask):
        coords = self.get_object_coordinates(mask)
                        # Create an overlay image
        overlay = self.get_bouding_box(coords, image)
        return coords, overlay