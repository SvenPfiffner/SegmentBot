import cv2
import numpy as np 
import supervision as sv

import torch
import torchvision

import sys
import os

# Add the path to the groundingdino module
#sys.path.append(os.path.abspath("groundingdino"))

from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import Model

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "checkpoints/groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "checkpoints/sam_vit_h_4b8939.pth"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(image, prompt):
    """
    Predicts and annotates objects in an image using GroundingDINO and SAM models.

    Args:
        image (np.ndarray): The input image in which objects are to be detected.
        prompt (str): A comma-separated string of class names to be detected in the image.

    Returns:
        tuple: A tuple containing:
            - annotated_frame (np.ndarray): The image annotated with bounding boxes from GroundingDINO.
            - annotated_image (np.ndarray): The image annotated with both bounding boxes and masks from SAM.
    """
    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)


    # Predict classes and hyper-param for GroundingDINO
    CLASSES = prompt
    BOX_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.8

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    # Prompting SAM with detected boxes
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)


    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )
    
    return detections, CLASSES
