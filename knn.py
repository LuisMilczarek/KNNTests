#!/usr/bin/env python

import os
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from typing import Tuple

from sklearn import metrics
from utils import ResizePreprocessingLayer, PreprocessLayer
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
        

    def predict(self, image : np.ndarray) -> Tuple[float, float]:
        '''
        Classify if the input image is of the dataset type.
        ### Parameters
        1. image : numpy.array  Image to predict
        ### Returns
        - float- percent of
        '''
        # print("predict")

        img = deepcopy(image)
        img = self._preprocess(img)
        img = img.flatten()
        # distances = np.sqrt(np.sum(np.power(self._dataset - img,2),axis=1))
        distances = np.linalg.norm(self._dataset - img,axis=1)
        selection = []
        for i in range(len(distances)):
            if len(selection) <= self._k:
                selection.append((distances[i], self._labels[i]))
                selection.sort(key= lambda x : x[0])
            else:
                # print(distances)
                if distances[i] < selection[-1][0]:
                    selection.append((distances[i], self._labels[i]))
                    selection.sort(key= lambda x : x[0])
                    selection.pop()
        # print("end predict")

        votes = {}
        for vote in np.array(selection)[:,1]:
            if vote in votes.keys():
                votes[vote] += 1
            else:
                votes[vote] = 1
        winner = None
        for key in votes.keys():
            if winner == None or votes[key] > votes[winner]:
                winner = key
        # print(winner)
        return (winner, votes[winner] / self._k)
    
    def addPreprocessLayer(self, layer : PreprocessLayer):
        self._preprocessLayers.append(layer)
        return self
    
    def _preprocess(self, image : np.ndarray) -> np.ndarray:
        img = deepcopy(image)
        layer : PreprocessLayer
        for layer in self._preprocessLayers:
            img = layer.process(img)
            # print(img.shape)
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
            if not os.path.exists(f"{path}/{entry['file']}"):
                print(f"File doesnt exist: {entry['file']}")
                continue
            img = cv.imread(f"{path}/{entry['file']}")
            img = self._preprocess(img)
            if entry["type"] == "train":
                self._dataset.append(img.flatten())
                self._labels.append(float(entry["label"]))
            elif entry["type"] == "val":
                self._valDataset.append(img)
                self._valLabels.append(float(entry["label"]))
            else:
                raise Exception(f"Invalid sample type on image {entry['file']}: {entry['type']}")
        self._labelsRepr = config["labels"]
        
    def validate(self):

        confusion_matrix = {}
        # labelsN = {}
        total_samples = len(self._valDataset)
        positives = 0
        predictions = []
        for img, label in zip(self._valDataset, self._valLabels):
            if label not in confusion_matrix.keys():
                confusion_matrix[label] = dict([ (l , 0.0) for l in confusion_matrix.keys()])
                for key in confusion_matrix.keys(): 
                    confusion_matrix[key][label] = 0.0
            pLabel, _ = self.predict(img)
            predictions.append(pLabel)
            if pLabel not in confusion_matrix.keys():
                confusion_matrix[pLabel] = dict([ (l , 0.0) for l in confusion_matrix.keys()])
                for key in confusion_matrix.keys():
                    confusion_matrix[key][pLabel] = 0.0
            # print(confusion_matrix)
            # print(f"{label} vs {pLabel}")
            confusion_matrix[label][pLabel] += 1
            if label == pLabel:
                positives += 1

        for label_i in confusion_matrix.keys():
            n = sum(confusion_matrix[label_i].values())
            for label_j in confusion_matrix.keys():
                confusion_matrix[label_i][label_j] /= n
        
        matrix = []
        line : dict
        for line in confusion_matrix.values():
            matrix.append(list(line.values()))

        confusion_matrix_plot =  metrics.ConfusionMatrixDisplay(confusion_matrix = np.array(matrix), display_labels = self._labelsRepr)
        confusion_matrix_plot.plot(cmap="Blues")
        plt.show()
        print(confusion_matrix)
        print(matrix)
        print(f"Overall acc:{positives/ total_samples}")

        
        

if __name__ == "__main__":
    knn = KNN(20)
    knn.addPreprocessLayer(ResizePreprocessingLayer(128,128))
    knn.loadDataset("./dataset")
    knn.validate()
    # img = cv.imread("input.jpeg")
    # print(knn.predict(img))