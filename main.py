#!/usr/bin/env python

import numpy as np
from knn import KNN
def main() -> None:
    model = KNN()
    model.loadDataset()
    model.predict()

if __name__ == "__main__":
    main()