import random
import numpy as np
import pandas as pd


def createDesignValues(condition, numOfBlock, isShuffle=True):
    designValues = list()
    for block in range(numOfBlock):
        if isShuffle:
            random.shuffle(condition)
        designValues.append(condition)
    designValues = np.array(designValues).flatten().tolist()
    return designValues


def samplePosition(xBoundary, yBoundary):
    positionX = np.random.uniform(xBoundary[0], xBoundary[1])
    positionY = np.random.uniform(yBoundary[0], yBoundary[1])
    position = [positionX, positionY]
    return position


def crateVariableProduct(variableDict):
    levelNames = list(variableDict.keys())
    levelValues = list(variableDict.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    productDictList = []
    productDictList = [{levelName: str(modelIndex.get_level_values(levelName)[modelIndexNumber]) for levelName in levelNames} for modelIndexNumber in range(len(modelIndex))]
    return productDictList
