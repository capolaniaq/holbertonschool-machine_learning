#!/usr/bin/env python3
"""
Object Detection project
"""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Class Yolo
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initialization  the Yolo class
        """
        self.class_t = class_t
        self.model = K.models.load_model(model_path)
        self.nms_t = nms_t
        self.anchors = anchors
        with open(classes_path) as f:
            self.class_names = [line.strip() for line in f]
