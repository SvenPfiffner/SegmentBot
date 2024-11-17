from gsam import predict
from editors import Editor
import numpy as np
import cv2

class ProcessorSettings:

    def __init__(self, strength, pos_prompt, neg_prompt, censorer_name):
        self.strength = strength
        self.pos_prompt = pos_prompt
        self.neg_prompt = neg_prompt
        self.censorer_name = censorer_name

def process_image(image, settings):
    prompt = settings.pos_prompt.split(",") + settings.neg_prompt.split(",")
    predictions, classes = predict(image, prompt)


    pos_pred = [predictions.mask[i] for i in range(len(predictions.class_id)) if classes[predictions.class_id[i]] in settings.pos_prompt]
    neg_pred = [predictions.mask[i] for i in range(len(predictions.class_id)) if classes[predictions.class_id[i]] in settings.neg_prompt]

    # Perform edit with the SAM masks
    out_image = image.copy()
    out_image = Editor.get_Subclasses()[settings.censorer_name].edit(out_image, pos_pred, neg_pred, settings.strength)

    return out_image