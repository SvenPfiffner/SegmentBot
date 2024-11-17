from .Editor import Editor
import numpy as np
import cv2

NAME = "Inverse Pixel"
DESCRIPTION = "Pixelates the entire image except for the objects you specified to keep. The strength parameter controls the intensity of the pixelation effect."

class InversePixel(Editor):

    @staticmethod
    def edit(img, mask_pos, mask_neg, strength):
        
        censor = img.copy()
        mask_p = mask_pos.copy()
        mask_n = mask_neg.copy()

        # Ensure empty predictions are passive masks
        if mask_p == []:
            mask_p = [np.zeros(img.shape[:2], dtype=np.bool)]
        if mask_n == []:
            mask_n = [np.zeros(img.shape[:2], dtype=np.bool)]
        # -----------------------------------------------------------------

        mask_p = np.ones(img.shape[:2], dtype=np.bool)
        mask_n = np.bitwise_or.reduce(mask_n)

        mask = mask_p & ~mask_n

        mask = mask_p & ~mask_n
        strength *= 100
        print(strength)
        # Apply a pixelation effect to the image
        h, w = censor.shape[:2]
        # Resize the image to a small size
        temp = cv2.resize(censor, (w // int(strength), h // int(strength)), interpolation=cv2.INTER_LINEAR)
        # Resize the image back to its original size
        censor = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        # Remove the pixelation from outside the region
        censor[~mask] = img[~mask]

        # -----------------------------------------------------------------
        return censor

    @staticmethod
    def getName():
        return NAME
    
    @staticmethod
    def getDescription():
        return DESCRIPTION