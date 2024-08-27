#!/usr/bin/env python

import os
import json
import numpy as np
import cupy as cp
import cv2 as cv
import matplotlib.pyplot as plt

from typing import Tuple
from time import perf_counter

from sklearn import metrics
from utils import ResizePreprocessingLayer, PreprocessLayer, RGB2RGBPreprocessingLayer
from copy import deepcopy

class KNN(object):
    def __init__(self, k) -> None:
        self._preprocessLayers = []
        self._dataset = []
        self._valDataset = []
        self._labels = []
        self._valLabels = []
        self._k = k
        self._labelsRepr = []
        self._datasetSummary = None
        

    def predict(self, image : np.ndarray) -> Tuple[float, float]:
        '''
        Classify if the input image is of the dataset type.
        ### Parameters
        1. image : numpy.array  Image to predict
        ### Returns
        - float- percent of
        '''

        img = deepcopy(image)
        img = self._preprocess(img)
        img = cp.array(img.flatten())

        dataset = cp.array(self._dataset)

        distances = cp.linalg.norm(dataset - img,axis=1).get()
        selection = []
        for i in range(len(distances)):
            if len(selection) <= self._k:
                selection.append((distances[i], self._labels[i]))
                selection.sort(key= lambda x : x[0])
            else:
                if distances[i] < selection[-1][0]:
                    selection.append((distances[i], self._labels[i]))
                    selection.sort(key= lambda x : x[0])
                    selection.pop()
        votes = {}
        # for ar in selection:
        #     ar = ar.get()
        for vote in np.array(selection)[:,1]:
            if vote in votes.keys():
                votes[vote] += 1
            else:
                votes[vote] = 1
        winner = None
        for key in votes.keys():
            if winner == None or votes[key] > votes[winner]:
                winner = key
        return (winner, votes[winner] / self._k)
    
    def addPreprocessLayer(self, layer : PreprocessLayer):
        self._preprocessLayers.append(layer)
        return self
    
    def _preprocess(self, image : np.ndarray) -> np.ndarray:
        img = deepcopy(image)
        layer : PreprocessLayer
        for layer in self._preprocessLayers:
            img = layer.process(img)
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
        self._labelsRepr = config["labels"]
        self._datasetSummary = config["summary"]
        for type in config["data"].keys():
            for label in config["data"][type]:
                for entry in config["data"][type][label]:
                    if not os.path.exists(f"{path}/{entry}"):
                        print(f"File doesnt exist: {entry['file']}")
                        continue
                    img = cv.imread(f"{path}/{entry}")
                    if type == "train":
                        img = self._preprocess(img)
                        self._dataset.append(cp.array(img).flatten().get())
                        self._labels.append(self._labelsRepr.index(label))
                    elif type == "val":
                        self._valDataset.append(img)
                        self._valLabels.append(self._labelsRepr.index(label))
                    else:
                        raise Exception(f"Invalid sample type on image {entry['file']}: {entry['type']}")
        
    def validate(self):
        total_samples = len(self._valDataset)
        positives = 0
        predictions = []
        print(self._datasetSummary)
        start = perf_counter()
        for img in self._valDataset:
            pred,_ = knn.predict(img)
            predictions.append(pred)
        matrix = metrics.confusion_matrix(self._valLabels, predictions)
        matrix = matrix / matrix.sum(axis=1)
        total_time = perf_counter() - start
        confusion_matrix_plot =  metrics.ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = self._labelsRepr)
        confusion_matrix_plot.plot(cmap="Blues")
        plt.show()
        print(f"Overall acc: {positives/ total_samples}, Total time: {total_time:.4f}s, Avg. Time per iteration: {total_time/ len(self._valDataset):.4f}")


if __name__ == "__main__":
    knn = KNN(5)
    knn.addPreprocessLayer(RGB2RGBPreprocessingLayer()).addPreprocessLayer(ResizePreprocessingLayer(128,128))

    knn.loadDataset("./dataset")
    knn.validate()
    # img = cv.imread("input.jpeg")
    # print(knn.predict(img))