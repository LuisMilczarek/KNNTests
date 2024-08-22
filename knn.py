#!/usr/bin/env python

import os
import json
import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from utils import ResizePreprocessingLayer, PreprocessLayer
from copy import deepcopy

class KNN(object):
    def __init__(self) -> None:
        self._preprocessLayers = []
        self._dataset = []
        self._labels = []
        

    def predict(self, image : np.ndarray) -> bool:
        '''
        Classify if the input image is of the dataset type.
        ### Parameters
        1. image : numpy.array  Image to predict
        ### Returns
        - bool- True if the image if of the type        
        '''
        img = deepcopy(image)
        img = self._preprocess(img)
        plt.imshow(img)
        plt.show()
        img = img.flatten()
        print(self._dataset - img)
        return False
    
    def addPreprocessLayer(self, layer : PreprocessLayer):
        self._preprocessLayers.append(layer)
        return self
    
    def _preprocess(self, image : np.ndarray) -> np.ndarray:
        img = deepcopy(image)
        layer : PreprocessLayer
        for layer in self._preprocessLayers:
            img = layer.process(img)
            print(img.shape)
        return img

    def loadDataset(self, path : str) -> None:
        '''
        Loads the dataset into the model
        ### Parameters
        1. path : str
            - Path where the dataset is located
        ### Raises:
        - InvalidArgumment : if the path or the config file is not valid
        '''
        if not os.path.exists(path):
            raise ValueError("Dataset path doesnt")
        if not os.path.exists(f"{path}/config.json"):
            raise ValueError("Dataset config file not found.")
        config : dict = json.load(open(f"{path}/config.json"))
        if not "data" in config.keys():
            raise Exception("Bad config json format")
        for entry in config["data"]:
            img = cv.imread(f"{path}/{entry['file']}")
            img = self._preprocess(img)
            self._dataset.append(img.flatten())
            self._labels.append(bool(entry["label"]))
        # print(self._dataset)
        # print(self._labels)

        # size = (5,5)
        # total = size[0]*size[1]
        # fig, axis = plt.subplots(size[0],size[1])
        # counter = 0
        # fig_counter = 0
        # if not os.path.exists("output"):
        #     os.mkdir("output")
        # for img, label in zip(self._dataset, self._labels):
        #     axis[counter//size[0], counter%size[0]].imshow(img)
        #     axis[counter//size[0], counter%size[0]].set_title(f"Label: {bool(label)}")
        #     axis[counter//size[0], counter%size[0]].axis("off")
        #     counter += 1
        #     if counter >= total:
        #         counter = 0
        #         fig.savefig(f"output/output_{fig_counter}")
        #         fig_counter += 1
        #         plt.close(fig)
        #         fig, axis = plt.subplots(size[0],size[1])
            
        

if __name__ == "__main__":
    knn = KNN()
    knn.addPreprocessLayer(ResizePreprocessingLayer(256,256))
    knn.loadDataset("./dataset")
    
    img = cv.imread("input.jpg")
    knn.predict(img)