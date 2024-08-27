    #!/usr/bin/env python
import cv2 as cv

from copy import deepcopy

import numpy as np
from .preprocessinglayer import PreprocessLayer

class RGB2RGBPreprocessingLayer(PreprocessLayer):
    def __init__(self) -> None:
        super().__init__()
    
    def process(self, image: np.ndarray) -> np.ndarray:
        img = deepcopy(image)
        try:
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        except Exception as e:
            print(img.shape)
            raise e
        return img_gray
    
# if __name__ == "__main__":
#     from matplotlib import pyplot as plt

#     img = cv.imread("input.jpg")
#     processLayer = RGB2RGBPreprocessingLayer()
#     img = processLayer.process(img)
#     # plt.imshow(img)
#     # plt.show()
#     cv.imshow("Teste",img)
#     cv.waitKey()