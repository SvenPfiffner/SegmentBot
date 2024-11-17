from .Editor import Editor
import numpy as np
import cv2

NAME = "Inverse Blur"
DESCRIPTION = "Blurs the entire image except for the objects you specified to keep. The strength parameter controls the intensity of the blur effect."

class InverseBlur(Editor):

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
        strength *= 100
        # Gaussian blur kernel size must be odd
        strength = int(strength) - (int(strength + 1) % 2)

        # Apply a Gaussian blur to the image
        censor = cv2.GaussianBlur(censor, (strength, strength), 0, dst=censor, borderType=cv2.BORDER_DEFAULT)
        # Remove the blur from outside the region
        censor[~mask] = img[~mask]

        # -----------------------------------------------------------------
        return censor

    @staticmethod
    def getName():
        return NAME
    
    @staticmethod
    def getDescription():
        return DESCRIPTION