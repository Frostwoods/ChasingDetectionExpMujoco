import os
import sys
import collections as co
import numpy as np
import random
import itertools as it

import json
import pygame
from pygame import time
from pygame.color import THECOLORS

# import pandas as pd

os.chdir(sys.path[0])
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.design import createDesignValues
from src.experiment import Experiment
from src.trial import ChaseTrialMujocoFps, CheckHumanResponseWithSpace
from src.visualization import InitializeScreen, DrawStateWithRope, DrawImage, DrawBackGround, DrawImageClick, DrawText, DrawFixationPoint, DrawStateWithRope,DrawState
from src.pandasWriter import WriteDataFrameToCSV
from src.loadChaseData import ScaleTrajectory,AdjustDfFPStoTraj,loadFromPickle,HorizontalRotationTransformTrajectory,RotationTransformTrajectory
# def crateVariableProduct(variableDict):
#     levelNames = list(variableDict.keys())
#     levelValues = list(variableDict.values())
#     modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
#     productDictList=[]
#     productDictList=[{levelName:str(modelIndex.get_level_values(levelName)[modelIndexNumber]) \
#             for levelName in levelNames} for modelIndexNumber in range(len(modelIndex))]
#     return productDictList
class GetSavePath:
    def __init__(self, dataDirectory, extension, fixedParameters={}):
        self.dataDirectory = dataDirectory
        self.extension = extension
        self.fixedParameters = fixedParameters

    def __call__(self, parameters):
        allParameters = dict(list(parameters.items()) + list(self.fixedParameters.items()))
        sortedParameters = sorted(allParameters.items())
        nameValueStringPairs = [parameter[0] + '=' + str(parameter[1]) for parameter in sortedParameters]

        fileName = '_'.join(nameValueStringPairs) + self.extension
        fileName = fileName.replace(" ", "")

        path = os.path.join(self.dataDirectory, fileName)
        return path


class InterpolateState:
    def __init__(self, numFramesToInterpolate):
        self.numFramesToInterpolate = numFramesToInterpolate

    def __call__(self, state, nextState):
        # print(state[0][0])
        interpolatedStates = [state]
        actionForInterpolation = (np.array(nextState) - np.array(state)) / (self.numFramesToInterpolate + 1)
        for frameIndex in range(self.numFramesToInterpolate):
            nextStateForInterpolation = np.array(state) + np.array(actionForInterpolation)
            interpolatedStates.append(list(nextStateForInterpolation))
            state = nextStateForInterpolation
        # print(list(interpolatedStates))
        return interpolatedStates

class  ScaleTrajectoryInTime:
    def __init__(self,interpolateState):
        self.interpolateState=interpolateState

    def __call__(self, trajs):
        rescaleTrajs = []
        for traj in trajs:
            # print(traj[0][1])
            newTraj=[]
            for t in range(len(traj)-1):
                newTraj = newTraj+self.interpolateState(traj[t],traj[t+1])
            # print(newTraj)
            rescaleTrajs.append(newTraj)

        return  rescaleTrajs


def main():
    numOfAgent = 4


    manipulatedVariables = co.OrderedDict()
    # manipulatedVariables['damping'] = [0.0]  # [0.0, 1.0]
    # manipulatedVariables['frictionloss'] = [0.0]  # [0.0, 0.2, 0.4]
    # manipulatedVariables['masterForce'] = [0.0]
    manipulatedVariables['damping'] = [0.5]  # [0.0, 1.0]
    manipulatedVariables['frictionloss'] = [1.0]  # [0.0, 0.2, 0.4]
    manipulatedVariables['masterForce'] = [0.0]  # [0.0, 2.0]
    manipulatedVariables['offset'] = [-0.5] #[0.0,-1.0,-0.5,0.5,1.0]
    manipulatedVariables['hideId'] = [3]
    manipulatedVariables['fps'] = [40]  
    manipulatedVariables['displayTime'] = [10]

    chaseTrailVariables = manipulatedVariables.copy()
    # catchTrailVariables = manipulatedVariables.copy()
    # chaseTrailVariables['hideId'] = [3,4] #0 wolf 1 sheep 2 master 3 4 distractor
    # catchTrailVariables['hideId'] = [1]
    chaseTrailconditions = [dict(list(specificValueParameter)) for specificValueParameter in it.product(*[[(key, value) for value in values] for key, values in chaseTrailVariables.items()])]
    # catchTrailconditions = [dict(list(specificValueParameter)) for specificValueParameter in it.product(*[[(key, value) for value in values] for key, values in catchTrailVariables.items()])]

    conditionsWithId =list (zip(range(len(chaseTrailconditions)),chaseTrailconditions))
    print(conditionsWithId)
    conditions = chaseTrailconditions
    conditions = [condition.update({'conditionId': condtionId}) for condtionId,condition in zip(range(len(conditions)),conditions )]
   
    chaseTrailNum = 10
    chaseTrailTrajetoryIndexList =[0,9]#[0] #range (chaseTrailNum)
    chaseTrailManipulatedVariablesForExp =  co.OrderedDict()
    chaseTrailManipulatedVariablesForExp['conditonId'] = range(len(chaseTrailconditions))
    chaseTrailManipulatedVariablesForExp['trajetoryIndex'] = chaseTrailTrajetoryIndexList
    chaseTrailProductedValues = it.product(*[[(key, value) for value in values] for key, values in chaseTrailManipulatedVariablesForExp.items()])
    
    # catchTrailNum = 0 
    # catchTrailTrajetoryIndexList = range (catchTrailNum)
    # catchTrailManipulatedVariablesForExp =  co.OrderedDict()
    # catchTrailManipulatedVariablesForExp['conditonId'] = range(len(chaseTrailconditions),len(chaseTrailconditions)+len(catchTrailconditions))
    # catchTrailManipulatedVariablesForExp['trajetoryIndex'] = catchTrailTrajetoryIndexList
    # catchTrailProductedValues = it.product(*[[(key, value) for value in values] for key, values in catchTrailManipulatedVariablesForExp.items()])

    # print(productedValues)
    exprimentVarableList = [dict(list(specificValueParameter)) for specificValueParameter in chaseTrailProductedValues]

    [exprimentVarable.update({'condition': conditionsWithId[exprimentVarable['conditonId']][1]}) for exprimentVarable in exprimentVarableList ]
    # exprimentVarableList = [exprimentVarable.update({'condition': conditionsWithId[exprimentVarable['conditonId']][1]}) for exprimentVarable in exprimentVarableList ]
    # print(exprimentVarableList)
    # print(len(exprimentVarableList))
    numOfBlock = 1
    numOfTrialsPerBlock = 1
    isShuffle = False
    designValues = createDesignValues(exprimentVarableList * numOfTrialsPerBlock, numOfBlock,isShuffle)


    positionIndex = [0, 1]
    standardFPS = 50
    rawXRange = [200, 600]
    rawYRange = [200, 600]
    scaledXRange = [200, 600]
    scaledYRange = [200, 600]
    scaleTrajectoryInSpace = ScaleTrajectory(positionIndex, rawXRange, rawYRange, scaledXRange, scaledYRange)
    oldFPS = 50
    numFramesToInterpolate = int(standardFPS/oldFPS - 1)
    interpolateState = InterpolateState(numFramesToInterpolate)
    scaleTrajectoryInTime = ScaleTrajectoryInTime(interpolateState)
    horizontalRotationTransformTrajectory = HorizontalRotationTransformTrajectory(positionIndex, rawXRange, rawYRange )
    rotationTransformTrajectory = RotationTransformTrajectory(positionIndex, rawXRange, rawYRange )
    def transFormTrajectory(trajList,randomId):
        randomSeed = np.mod(randomId, 8)
        rotationAngle =0 #np.mod(randomSeed, 4) * np.pi / 2
        rotationTraj = rotationTransformTrajectory(trajList, rotationAngle)
        # if np.mod(randomSeed//4,2) ==1:
            # finalTrajs = horizontalRotationTransformTrajectory(rotationTraj)
        # else:
        finalTrajs = rotationTraj
        return trajList#finalTrajs
    # trajectoriesSaveDirectory ='../PataData/preExpMasterOffset'
    trajectoriesSaveDirectory ='../PataData/offsetMasterTrajNewForRecheck'
    
    # trajectoriesSaveDirectory =os.path.join(dataFolder, 'trajectory', modelSaveName)
    trajectorySaveExtension = '.pickle'

    selctDict = {3:chaseTrailNum,4:chaseTrailNum}
    evaluateEpisode = 120000
    evalNum = 20
    fixedParameters = {'distractorNoise': 3.0,'evaluateEpisode': evaluateEpisode}
    generateTrajectoryLoadPath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    trajectoryDf = lambda condition: loadFromPickle(generateTrajectoryLoadPath({'offset':condition['offset'],'hideId':condition['hideId'],'damping': condition['damping'], 'frictionloss': condition['frictionloss'], 'masterForce': condition['masterForce'],'select':selctDict[condition['hideId']] }))

    getTrajectory = lambda trajectoryDf: scaleTrajectoryInTime(scaleTrajectoryInSpace(trajectoryDf))
    
    print('loading')
    # stimulus = {conditionId:getTrajectory(trajectoryDf(condition))  for conditionId,condition in conditionsWithId}
# 
    transformedStimulus = {conditionId: transFormTrajectory(getTrajectory(trajectoryDf(condition)),conditionId)  for conditionId,condition in conditionsWithId}
    print('loding success')

    # print(transformedStimulus[0][0][230])
    # print(transformedStimulus[1][0][230])
    # print(transformedStimulus[2][0][230])
    # print(transformedStimulus[3][0][230])

    # print(len(transformedStimulus[0][0]))
    # print(len(transformedStimulus[1][0]))
    # print(len(transformedStimulus[2][0]))
    # print(len(transformedStimulus[3][0]))
    # [print(len(transformedStimulus[sti])) for sti in range(32)]
    experimentValues = co.OrderedDict()
    experimentValues["name"] = input("Please enter your name:").capitalize()
    
    
    screenWidth = 800
    screenHeight = 800

    fullScreen = False
    initializeScreen = InitializeScreen(screenWidth, screenHeight, fullScreen)
    screen = initializeScreen()
 
    leaveEdgeSpace = 180
    circleSize = 10
    clickImageHeight = 80
    lineWidth = 3
    fontSize = 50
    xBoundary = [leaveEdgeSpace, screenWidth - leaveEdgeSpace * 2]
    yBoundary = [leaveEdgeSpace, screenHeight - leaveEdgeSpace * 2]

    screenColor = THECOLORS['black']
    lineColor = THECOLORS['white']
    textColor = THECOLORS['white']
    fixationPointColor = THECOLORS['white']

    colorSpace=[(203,164,4,255),(49,153,0,255),(255,90,16,255),(251,7,255,255),(9,204,172,255),(3,28,255,255)]


    picturePath = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'pictures')
    resultsPath = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'results','preExpMasterOffset')
    if not os.path.exists(resultsPath):
        os.makedirs(resultsPath)
    introductionImage1 = pygame.image.load(os.path.join(picturePath, 'IdOnlyIntro1.png'))
    introductionImage2 = pygame.image.load(os.path.join(picturePath, 'IdOnlyIntro2.png'))
    finishImage = pygame.image.load(os.path.join(picturePath, 'over.jpg'))
    introductionImage1 = pygame.transform.scale(introductionImage1, (screenWidth, screenHeight))
    introductionImage2 = pygame.transform.scale(introductionImage2, (screenWidth, screenHeight))


    finishImage = pygame.transform.scale(finishImage, (int(screenWidth * 2 / 3), int(screenHeight / 4)))
    clickWolfImage = pygame.image.load(os.path.join(picturePath, 'clickwolf.png'))
    clickSheepImage = pygame.image.load(os.path.join(picturePath, 'clicksheep.png'))
    restImage = pygame.image.load(os.path.join(picturePath, 'rest.jpg'))

    drawImage = DrawImage(screen)
    drawText = DrawText(screen, fontSize, textColor)
    drawBackGround = DrawBackGround(screen, screenColor, xBoundary, yBoundary, lineColor, lineWidth)
    drawFixationPoint = DrawFixationPoint(screen, drawBackGround, fixationPointColor)
    drawImageClick = DrawImageClick(screen, clickImageHeight, drawText)

    
    drawState = DrawState(screen, circleSize, numOfAgent, positionIndex, drawBackGround)
   

    writerPath = os.path.join(resultsPath, experimentValues["name"]) + '.csv'
    writer = WriteDataFrameToCSV(writerPath)

    # displayFrames = 500
    reactionWindowStart = 50
    keysForCheck = {'space': 1}
    checkHumanResponse = CheckHumanResponseWithSpace(keysForCheck)
    trial = ChaseTrialMujocoFps(conditionsWithId,reactionWindowStart, drawState, drawImage, transformedStimulus, checkHumanResponse, colorSpace, numOfAgent, drawFixationPoint, drawText, drawImageClick, clickWolfImage, clickSheepImage)
    
    experiment = Experiment(trial, writer, experimentValues,drawImage,restImage,drawBackGround)
   
    restDuration=120

    drawImage(introductionImage1)
    drawImage(introductionImage2)
    

    experiment(designValues,restDuration)
    # self.darwBackground()
    drawImage(finishImage)


    print("Result saved at {}".format(writerPath))


if __name__ == '__main__':
    main()
