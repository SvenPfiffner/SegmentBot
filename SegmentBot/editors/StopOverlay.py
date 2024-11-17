from .Editor import Editor
import numpy as np
import cv2

NAME = "Stop Overlay"
DESCRIPTION = "Overlay the detected regions with a stop sign texture. The strength parameter controls the intensity of the effect."

class StopOverlay(Editor):

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

        mask_p = np.bitwise_or.reduce(mask_p)
        mask_n = np.bitwise_or.reduce(mask_n)

        stop_texture = StopOverlay.build_stop_texture(img)

        mask = mask_p & ~mask_n
        _strength = strength * 100
        # Gaussian blur kernel size must be odd
        _strength = int(_strength) - (int(_strength + 1) % 2)

        # Apply a Gaussian blur to the image
        censor = cv2.GaussianBlur(censor, (_strength, _strength), 0, dst=censor, borderType=cv2.BORDER_DEFAULT)
        # Overlay the stop sign texture on the image
        censor[mask] = cv2.addWeighted(censor, 1 - strength, stop_texture, strength, 0)[mask]

        # Remove the blur from outside the region
        censor[~mask] = img[~mask]

        # -----------------------------------------------------------------
        return censor
    
    @staticmethod
    def build_stop_texture(img):
        # Load the stop sign texture
        stop_texture = cv2.imread("censorers/media/stoptexture.jpg")

        # Resize the stop sign texture to be exactly 1/75th the size of the smaller dimension of the image
        smaller_dim = min(img.shape[0], img.shape[1])
        stop_texture = cv2.resize(stop_texture, (smaller_dim // 50, smaller_dim // 50))

        # Stitch the stop sign texture together to form a grid that covers the entire image
        stop_texture = np.tile(stop_texture, (img.shape[0] // stop_texture.shape[0] + 1, img.shape[1] // stop_texture.shape[1] + 1, 1))
        # Resize the stop sign texture to the size of the image
        stop_texture = stop_texture[:img.shape[0], :img.shape[1]]

        return cv2.cvtColor(stop_texture, cv2.COLOR_BGR2RGB)
    @staticmethod
    def getName():
        return NAME
    
    @staticmethod
    def getDescription():
        return DESCRIPTION