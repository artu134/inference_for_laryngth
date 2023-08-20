
from model_code.detector.detector_abstract import DetectorAbstract
from skimage.morphology import binary_erosion

import cv2 
import numpy as np

class DetectorMask(DetectorAbstract): 
        
    def __init__(self):
        pass
    
    
    def _image_overlay(self, img, pred, background_mask=None):
    # pred is the mask obtained from your model (one-hot encoded or binary)
        overlay = img.copy()
        
        if background_mask is not None:
            pred = pred[..., :background_mask] + pred[..., background_mask+1:]
        
        overlay[pred > 0] = (255, 0, 0)  # Overlaying with red color where mask has objects
        
        return overlay

    def _image_overlay_opposite(self, image, mask):

        overlay = image.copy()
        if len(image.shape) == 2:  # if grayscale
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)       
    
        mask = mask.astype(bool)
        border = mask ^ binary_erosion(mask).astype(bool)  # Finding the border of the mask

        # Only apply overlay if border is correctly formed
        if np.any(border):
            overlay[border] = (255, 0, 0)  # Overlaying with different color for each mask
        else:
            print("Skipping overlay for mask due to incorrectly formed border.")

        return overlay, border


    def detect(self, image, mask):
                        # Create an overlay image
        original_with_overlay_red, border = self._image_overlay_opposite(image, mask)
        return border, original_with_overlay_red