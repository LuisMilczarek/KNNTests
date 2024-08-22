#!/usr/bin/env python

import numpy as np

class PreprocessLayer(object):
    def __init__(self) -> None:
        pass

    def process(image : np.ndarray) -> np.ndarray:
        raise NotImplementedError("This methos must be extended in PreprocessLayer child classes")
