#!/usr/bin/env python

import os
import json
import numpy as np
import cv2 as cv


class KNN(object):
    def __init__(self) -> None:
        self._preprocessLayers = []
        self._dataset = []
        self._labels = []
        

    def predict(self, image ) -> bool:
        '''
        Classify if the input image is of the dataset type.
        ### Parameters
        1. image : numpy.array  Image to predict
        ### Returns
        - bool- True if the image if of the type        
        '''
        

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
            self._dataset.append(img)
            self._labels.append(entry["label"])
        print(len(self._dataset))
        print(len(self._labels))
        

if __name__ == "__main__":
    knn = KNN()
    knn.loadDataset("./dataset")