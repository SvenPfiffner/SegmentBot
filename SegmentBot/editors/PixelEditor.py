# This is an example censorer that remains the image unchanged.
# To add your own censorer to the program, create a new python file in the censorers directory named after your censorer,
# rename the class to your censorer's name, and change the following:
#     NAME = Put a short name for your censorer here. This will be displayed in the dropdown menu for selecting censorers.
#     DESCRIPTION = Put a short description of what your censorer does here.
#     censor(img) = Implement the censoring logic here. This method recieves the base image, a binary mask of the
#     detected region that the user would like to censor, and a binary mask of the detected region that the user would like to keep.
#     Also, a strength parameter is passed to the censor method, which can be used to control the intensity of the censoring effect
#     if you want.

from .Editor import Editor
import numpy as np
import cv2

NAME = "Pixel Editor"
DESCRIPTION = "Pixelates the detected region in the image. The strength parameter controls the intensity of the pixelation effect."

class PixelEditor(Editor):

    @staticmethod
    def edit(img, mask_pos, mask_neg, strength):

        censor = img.copy()
        mask_p = mask_pos.copy()
        mask_n = mask_neg.copy()

        if mask_p == []:
            mask_p = [np.zeros(img.shape[:2], dtype=np.bool)]
        if mask_n == []:
            mask_n = [np.zeros(img.shape[:2], dtype=np.bool)]

        # -----------------------------------------------------------------

        mask_p = np.bitwise_or.reduce(mask_p)
        mask_n = np.bitwise_or.reduce(mask_n)

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