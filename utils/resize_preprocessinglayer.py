#!/usr/bin/env python

import cv2 as cv
from numpy import ndarray

from .preprocessinglayer import PreprocessLayer

class ResizePreprocessingLayer(PreprocessLayer):
    def __init__(self, height,width) -> None:
        self._h = height
        self._w = width
        super().__init__()

    def process(self, image: ndarray) -> ndarray:
        # print(len(image))
        img = cv.resize(image,(self._w, self._h), interpolation= cv.INTER_LINEAR)
        return img