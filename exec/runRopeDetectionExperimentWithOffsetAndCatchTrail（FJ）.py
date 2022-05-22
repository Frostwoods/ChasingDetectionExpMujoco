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

import pandas as pd

os.chdir(sys.path[0])
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.design import createDesignValues, samplePosition
from src.experiment import Experiment
from src.trial import ChaseTrialMujoco, CheckHumanResponse,ChaseTrialMujocoWithLineRope
from src.visualization import InitializeScreen, DrawImage, DrawBackGround, DrawImageClick, DrawText, DrawFixationPoint, DrawStateWithRope,DrawState,DrawStateWithLineRope
from src.pandasWriter import WriteDataFrameToCSV
from src.loadChaseData import ScaleTrajectory,AdjustDfFPStoTraj,loadFromPickle,HorizontalRotationTransformTrajectory,RotationTransformTrajectory
def crateVariableProduct(variableDict):
    levelNames = list(variableDict.keys())
    levelValues = list(variableDict.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    productDictList=[]
    productDictList=[{levelName:str(modelIndex.get_level_values(levelName)[modelIndexNumber])  for levelName in levelNames} for modelIndexNumber in range(len(modelIndex))]
    return productDictList
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
    manipulatedVariables['masterForce'] = [1.0]  # [0.0, 2.0]
    manipulatedVariables['offset'] = [0.0]
    manipulatedVariables['tiedPairs'] = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]

    chaseTrailVariables = manipulatedVariables.copy()
    catchTrailVariables = manipulatedVariables.copy()
    chaseTrailVariables['hideId'] = [3,4] #0 wolf 1 sheep 2 master 3 4 distractor
    catchTrailVariables['hideId'] = [1]
    chaseTrailconditions = [dict(list(specificValueParameter)) for specificValueParameter in it.product(*[[(key, value) for value in values] for key, values in chaseTrailVariables.items()])]
    # chaseTrailconditionsWithId =list (zip(range(len(conditions)),conditions ))
    catchTrailconditions = [dict(list(specificValueParameter)) for specificValueParameter in it.product(*[[(key, value) for value in values] for key, values in catchTrailVariables.items()])]
    # catchTrailconditionsWithId =list (zip(range(len(conditions)),conditions ))
    # print('state',chaseTrailVariables,catchTrailVariables)
    conditionsWithId =list (zip(range(len(chaseTrailconditions)+len(catchTrailconditions)),chaseTrailconditions + catchTrailconditions ))
    # print(conditionsWithId)
    # conditions = [dict(list(specificValueParameter)) for specificValueParameter in it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])]
    # conditionsWithId =list (zip(range(len(conditions)),conditions ))
    # print(conditionsWithId,'`12`3')
    conditions = chaseTrailconditions + catchTrailconditions
    conditions = [condition.update({'conditionId': condtionId}) for condtionId,condition in zip(range(len(conditions)),conditions )]
    # print (conditions)
    
    
    
    chaseTrailNum=10
    chaseTrailTrajetoryIndexList = range (chaseTrailNum)
    chaseTrailManipulatedVariablesForExp =  co.OrderedDict()
    chaseTrailManipulatedVariablesForExp['conditonId'] = range(len(chaseTrailconditions))
    chaseTrailManipulatedVariablesForExp['trajetoryIndex'] = chaseTrailTrajetoryIndexList
    chaseTrailProductedValues = it.product(*[[(key, value) for value in values] for key, values in chaseTrailManipulatedVariablesForExp.items()])

    catchTrailNum=20
    catchTrailTrajetoryIndexList = range (catchTrailNum)
    catchTrailManipulatedVariablesForExp =  co.OrderedDict()
    catchTrailManipulatedVariablesForExp['conditonId'] = range(len(chaseTrailconditions),len(chaseTrailconditions)+len(catchTrailconditions))
    catchTrailManipulatedVariablesForExp['trajetoryIndex'] = catchTrailTrajetoryIndexList
    catchTrailProductedValues = it.product(*[[(key, value) for value in values] for key, values in catchTrailManipulatedVariablesForExp.items()])

    # print(productedValues)
    exprimentVarableList = [dict(list(specificValueParameter)) for specificValueParameter in chaseTrailProductedValues]+[dict(list(specificValueParameter)) for specificValueParameter in catchTrailProductedValues]

    [exprimentVarable.update({'condition': conditionsWithId[exprimentVarable['conditonId']][1]}) for exprimentVarable in exprimentVarableList ]
    # exprimentVarableList = [exprimentVarable.update({'condition': conditionsWithId[exprimentVarable['conditonId']][1]}) for exprimentVarable in exprimentVarableList ]
    print(exprimentVarableList)
    print(len(exprimentVarableList))
    numOfBlock = 1
    numOfTrialsPerBlock = 1
    isShuffle = False
    designValues = createDesignValues(exprimentVarableList * numOfTrialsPerBlock, numOfBlock,isShuffle)


    positionIndex = [0, 1]
    FPS = 50
    rawXRange = [200, 600]
    rawYRange = [200, 600]
    scaledXRange = [200, 600]
    scaledYRange = [200, 600]
    scaleTrajectoryInSpace = ScaleTrajectory(positionIndex, rawXRange, rawYRange, scaledXRange, scaledYRange)
    oldFPS = 50
    numFramesToInterpolate = int(FPS/oldFPS - 1)
    interpolateState = InterpolateState(numFramesToInterpolate)
    horizontalRotationTransformTrajectory = HorizontalRotationTransformTrajectory(positionIndex, scaledXRange, scaledYRange )
    rotationTransformTrajectory = RotationTransformTrajectory(positionIndex, scaledXRange, scaledYRange )
    def transFormTrajectory(trajList,randomId):
        if randomId<12:
            isMirror = randomId//8
            rotationAngle =np.mod(randomId//2,4)* np.pi / 2
        else:
            isMirror = randomId//16
            rotationAngle = np.mod(randomId,4)* np.pi / 2
        rotationTraj = rotationTransformTrajectory(trajList, rotationAngle)
        
        if isMirror:
            finalTrajs = horizontalRotationTransformTrajectory(rotationTraj)

        finalTrajs = rotationTraj
        return finalTrajs


    scaleTrajectoryInTime = ScaleTrajectoryInTime(interpolateState)

    trajectoriesSaveDirectory ='../PataData/chasingDetectionRope'
    # trajectoriesSaveDirectory =os.path.join(dataFolder, 'trajectory', modelSaveName)
    trajectorySaveExtension = '.pickle'
    selctDict = {1:catchTrailNum,3:chaseTrailNum,4:chaseTrailNum}
    evaluateEpisode = 120000
    evalNum = 20
    fixedParameters = {'distractorNoise': 3.0,'evaluateEpisode': evaluateEpisode}
    # fixedParameters = {'distractorNoise': 3.0,'evaluateEpisode': evaluateEpisode, 'evalNum': evalNum}
    generateTrajectoryLoadPath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    # trajectoryDf = lambda condition: pd.read_pickle(generateTrajectoryLoadPath({'offset':condition['offset'],'hideId':condition['hideId'],'damping': condition['damping'], 'frictionloss': condition['frictionloss'], 'masterForce': condition['masterForce'], 'evalNum': evalNum,}))
    trajectoryDf = lambda condition: loadFromPickle(generateTrajectoryLoadPath({'offset':condition['offset'],'hideId':condition['hideId'],'damping': condition['damping'], 'frictionloss': condition['frictionloss'], 'masterForce': condition['masterForce'], 'select': selctDict[condition['hideId']]
                       }))

    # trajectoryDf = lambda condition: pd.read_pickle(generateTrajectoryLoadPath(condition))
    getTrajectory = lambda trajectoryDf: scaleTrajectoryInTime(scaleTrajectoryInSpace(trajectoryDf))
    # getTrajectory = lambda trajectoryDf: scaleTrajectoryInSpace(trajectoryDf)
    
    print('loading')
    # stimulus = {conditionId:transFormTrajectory(getTrajectory(trajectoryDf(condition)))  for conditionId,condition in conditionsWithId}
    # print(stimulus[0][0][0])
    # print(stimulus[20][0][0])
    stimulus = {conditionId: transFormTrajectory(getTrajectory(trajectoryDf(condition)),conditionId)  for conditionId,condition in conditionsWithId}
    print('loding success')

    # print(stimulus[1][1])
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
    resultsPath = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'results')

    introductionImage1 = pygame.image.load(os.path.join(picturePath, 'mujocoIntro1.png'))
    introductionImage2 = pygame.image.load(os.path.join(picturePath, 'mujocoIntro2.png'))
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

    ropeColor = THECOLORS['grey']
    ropeWidth = 4
    drawState = DrawStateWithLineRope(screen, circleSize, numOfAgent, positionIndex, ropeColor, ropeWidth, drawBackGround)
    # drawStateWithRope = DrawStateWithRope(screen, circleSize, numOfAgent, positionIndex, ropeColor, drawBackground)    
   

    writerPath = os.path.join(resultsPath, experimentValues["name"]) + '.csv'
    writer = WriteDataFrameToCSV(writerPath)

    displayFrames = 500
    keysForCheck = {'f': 0, 'j': 1}
    checkHumanResponse = CheckHumanResponse(keysForCheck)
    trial = ChaseTrialMujocoWithLineRope(conditionsWithId,displayFrames, drawState, drawImage, stimulus, checkHumanResponse, colorSpace, numOfAgent, drawFixationPoint, drawText, drawImageClick, clickWolfImage, clickSheepImage, FPS)

    experiment = Experiment(trial, writer, experimentValues,drawImage,restImage,drawBackGround)
   
    restDuration=20

    drawImage(introductionImage1)
    drawImage(introductionImage2)
    

    experiment(designValues,restDuration)
    # self.darwBackground()
    drawImage(finishImage)


    print("Result saved at {}".format(writerPath))


if __name__ == '__main__':
    main()
