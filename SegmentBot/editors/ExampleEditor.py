# This is an example editor that remains the image unchanged.
# To add your own editor to the program, create a new python file in the editors directory named after your editor,
# rename the class to your editors's name, and change the following:
#     NAME = Put a short name for your editor here. This will be displayed in the dropdown menu for selecting editors.
#     DESCRIPTION = Put a short description of what your editor does here.
#     censor(img) = Implement the edit logic here. This method recieves the base image, a binary mask of the
#     detected region that the user would like to edit, and a binary mask of the detected region that the user would like to keep as is.
#     Also, a strength parameter is passed to the edit method, which can be used to control the intensity of the edit effect
#     if you want. 

from .Editor import Editor
import numpy as np

NAME = "Identity Editor"
DESCRIPTION = "Example Edit Mode used to demonstrate the extension system. This mode does not apply any edits to the image and leaves it unchanged."

class ExampleEditor(Editor):

    @staticmethod
    def edit(img, mask_pos, mask_neg, strength):
        
        # This makes sure the changes of your editor do not affect the original image.
        # keep this unchanged
        edit_img = img.copy()
        mask_p = mask_pos.copy()
        mask_n = mask_neg.copy()

        # Ensure empty predictions are passive masks
        if mask_p == []:
            mask_p = [np.zeros(img.shape[:2], dtype=np.bool)]
        if mask_n == []:
            mask_n = [np.zeros(img.shape[:2], dtype=np.bool)]

        # Your editing logic goes here. modify the edit_img (the image)
        # and regions (the masks) as needed.
        #  - mask_p is the masking of things to censor
        #  - mask_n is the masking of things explicitly asked to keep
        # -----------------------------------------------------------------

        # -----------------------------------------------------------------
        # This makes sure your edit is only applied to your region.
        # keep this unchanged
        return edit_img

    @staticmethod
    def getName():
        return NAME
    
    @staticmethod
    def getDescription():
        return DESCRIPTION