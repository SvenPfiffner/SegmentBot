from .Editor import Editor
import numpy as np

NAME = "Mask Editor"
DESCRIPTION = "Processes the image according to the given prompts and returns the mask of the detection as a binary image."

class MaskingEditor(Editor):

    @staticmethod
    def edit(img, mask_pos, mask_neg, strength):
        
        out = img.copy()
        mask_p = mask_pos.copy()
        mask_n = mask_neg.copy()

        if mask_p == []:
            mask_p = [np.zeros(img.shape[:2], dtype=np.bool)]
        if mask_n == []:
            mask_n = [np.zeros(img.shape[:2], dtype=np.bool)]

        # -----------------------------------------------------------------

        out = np.zeros(img.shape, dtype=np.uint8)
        mask_p = np.bitwise_or.reduce(mask_p)
        mask_n = np.bitwise_or.reduce(mask_n)

        mask = mask_p & ~mask_n
        out[mask] = [255, 255, 255]

        # -----------------------------------------------------------------
        return out

    @staticmethod
    def getName():
        return NAME
    
    @staticmethod
    def getDescription():
        return DESCRIPTION