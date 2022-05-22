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
from src.trial import ChaseTrialMujoco, CheckHumanResponse
from src.visualization import InitializeScreen, DrawStateWithRope, DrawImage, DrawBackGround, DrawImageClick, DrawText, DrawFixationPoint, DrawStateWithRope,DrawState
from src.pandasWriter import WriteDataFrameToCSV
from src.loadChaseData import ScaleTrajectory,AdjustDfFPStoTraj,loadFromPickle
def crateVariableProduct(variableDict):
    levelNames = list(variableDict.keys())
    levelValues = list(variableDict.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    productDictList=[]
    productDictList=[{levelName:str(modelIndex.get_level_values(levelName)[modelIndexNumber]) \
            for levelName in levelNames} for modelIndexNumber in range(len(modelIndex))]
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
    manipulatedVariables['damping'] = [0.0,0.5]  # [0.0, 1.0]
    manipulatedVariables['frictionloss'] = [0.0,1.0]  # [0.0, 0.2, 0.4]
    manipulatedVariables['offset'] = [0.0]

    masterForceChaseTrailVariables = manipulatedVariables.copy()
    masterForceCatchTrailVariables = manipulatedVariables.copy()
    masterMassChaseTrailVariables = manipulatedVariables.copy()
    masterMassCatchTrailVariables = manipulatedVariables.copy()

    # masterForceChase
    # masterForceCatch
    # masterMassChase
    # masterMassCatch

    masterForceChaseTrailVariables['hideId'] = [2,3] #0 wolf 1 sheep 2 master 3 4 distractor
    masterForceChaseTrailVariables['masterForce'] = [2.0]
    masterForceChaseTrailVariables['masterMass'] = [1.0]

    masterForceCatchTrailVariables['hideId'] = [1] #0 wolf 1 sheep 2 master 3 4 distractor
    masterForceCatchTrailVariables['masterForce'] = [2.0]
    masterForceCatchTrailVariables['masterMass'] = [1.0]

    masterMassChaseTrailVariables['hideId'] = [2,3] #0 wolf 1 sheep 2 master 3 4 distractor
    masterMassChaseTrailVariables['masterForce'] = [0.0]
    masterMassChaseTrailVariables['masterMass'] = [2.0]

    masterMassCatchTrailVariables['hideId'] = [1] #0 wolf 1 sheep 2 master 3 4 distractor
    masterMassCatchTrailVariables['masterForce'] = [0.0]
    masterMassCatchTrailVariables['masterMass'] = [2.0]

    def reMoveCertainCondition(conditions):
        toDelCondion = []
        for condition in conditions:
            if condition['damping'] == 0.0 and condition['frictionloss'] == 0.0:
                # print(condition)
                toDelCondion.append(condition)
                # conditions.remove(condition)
        for delcon in toDelCondion:
            # print(delcon)
            conditions.remove(delcon)
        return conditions

    masterForceChaseTrailconditions = [dict(list(specificValueParameter)) for specificValueParameter in it.product(*[[(key, value) for value in values] for key, values in masterForceChaseTrailVariables.items()])]
    masterForceChaseTrailconditions = reMoveCertainCondition(masterForceChaseTrailconditions)
    
    masterForceCatchTrailconditions = [dict(list(specificValueParameter)) for specificValueParameter in it.product(*[[(key, value) for value in values] for key, values in masterForceCatchTrailVariables.items()])]
    masterForceCatchTrailconditions = reMoveCertainCondition(masterForceCatchTrailconditions)

    masterMassChaseconditions = [dict(list(specificValueParameter)) for specificValueParameter in it.product(*[[(key, value) for value in values] for key, values in masterMassChaseTrailVariables.items()])]
    masterMassChaseconditions = reMoveCertainCondition(masterMassChaseconditions)

    masterMassCatchconditions = [dict(list(specificValueParameter)) for specificValueParameter in it.product(*[[(key, value) for value in values] for key, values in masterMassCatchTrailVariables.items()])]
    masterMassCatchconditions = reMoveCertainCondition(masterMassCatchconditions)



    # conditionsWithId = list (zip(range(len(chaseTrailconditions + catchTrailconditions + masterMassChaseconditions + offsetMasterconditions)),chaseTrailconditions + catchTrailconditions + masterMassChaseconditions + offsetMasterconditions ))

    conditions = masterForceChaseTrailconditions + masterForceCatchTrailconditions + masterMassChaseconditions + masterMassCatchconditions
    # print(conditions[1]['damping'])
 
    print(len(conditions))
    conditionsWithId = list(zip(range(len(conditions)),conditions))
    conditions = [condition.update({'conditionId': condtionId}) for condtionId,condition in zip(range(len(conditions)),conditions )]

    # print (conditions)
    
    
    trajLengthList=[20,5,20,5]
    # trajLengthList=[1,1,1,1]
    # trajLengthList=[0,5,0,0]
    
    chaseTrailTrajetoryIndexList = range (trajLengthList[0])
    chaseTrailManipulatedVariablesForExp =  co.OrderedDict()
    chaseTrailManipulatedVariablesForExp['conditonId'] = range(len(masterForceChaseTrailconditions))
    chaseTrailManipulatedVariablesForExp['trajetoryIndex'] = chaseTrailTrajetoryIndexList
    chaseTrailProductedValues = it.product(*[[(key, value) for value in values] for key, values in chaseTrailManipulatedVariablesForExp.items()])

    catchTrailTrajetoryIndexList = range (trajLengthList[1])
    catchTrailManipulatedVariablesForExp =  co.OrderedDict()
    catchTrailManipulatedVariablesForExp['conditonId'] = range(len(masterForceChaseTrailconditions),len(masterForceChaseTrailconditions)+len(masterForceCatchTrailconditions))
    catchTrailManipulatedVariablesForExp['trajetoryIndex'] = catchTrailTrajetoryIndexList
    catchTrailProductedValues = it.product(*[[(key, value) for value in values] for key, values in catchTrailManipulatedVariablesForExp.items()])

    hideMasterTrajetoryIndexList = range (trajLengthList[2])
    hideMasterManipulatedVariablesForExp =  co.OrderedDict()
    hideMasterManipulatedVariablesForExp['conditonId'] = range(len(masterForceChaseTrailconditions)+len(masterForceCatchTrailconditions),len(masterForceChaseTrailconditions)+len(masterForceCatchTrailconditions)+len(masterMassChaseconditions))
    hideMasterManipulatedVariablesForExp['trajetoryIndex'] = hideMasterTrajetoryIndexList
    hideMasterProductedValues = it.product(*[[(key, value) for value in values] for key, values in hideMasterManipulatedVariablesForExp.items()])

    offsetMasterTrajetoryIndexList = range (trajLengthList[3])
    offsetMasterManipulatedVariablesForExp =  co.OrderedDict()
    offsetMasterManipulatedVariablesForExp['conditonId'] = range(len(masterForceChaseTrailconditions)+len(masterForceCatchTrailconditions)+len(masterMassChaseconditions),len(masterForceChaseTrailconditions)+len(masterForceCatchTrailconditions)+len(masterMassChaseconditions)+len(masterMassCatchconditions))
    offsetMasterManipulatedVariablesForExp['trajetoryIndex'] = offsetMasterTrajetoryIndexList
    offsetMasterProductedValues = it.product(*[[(key, value) for value in values] for key, values in offsetMasterManipulatedVariablesForExp.items()])

    # print(productedValues)
    exprimentVarableList = [dict(list(specificValueParameter)) for specificValueParameter in chaseTrailProductedValues]+[dict(list(specificValueParameter)) for specificValueParameter in catchTrailProductedValues] +[dict(list(specificValueParameter)) for specificValueParameter in hideMasterProductedValues] +[dict(list(specificValueParameter)) for specificValueParameter in offsetMasterProductedValues]

    [exprimentVarable.update({'condition': conditionsWithId[exprimentVarable['conditonId']][1]}) for exprimentVarable in exprimentVarableList ]
    # exprimentVarableList = [exprimentVarable.update({'condition': conditionsWithId[exprimentVarable['conditonId']][1]}) for exprimentVarable in exprimentVarableList ]
    print(exprimentVarableList)
    print(len(exprimentVarableList))
    numOfBlock = 1
    numOfTrialsPerBlock = 1
    isShuffle = True
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



    scaleTrajectoryInTime = ScaleTrajectoryInTime(interpolateState)

    trajectoriesSaveDirectory ='../PataData/killZone=4.0_distractorKillZone=0.0selectTrajByNov1'
    # trajectoriesSaveDirectory =os.path.join(dataFolder, 'trajectory', modelSaveName)
    trajectorySaveExtension = '.pickle'

    evaluateEpisode = 60000
    ropePunishWeight=0.3
    killZone = 4.0
    killZoneofDistractor = 0.0
    ropeLength = 0.06
    fixedParameters = {'distractorNoise': 1.0,'evaluateEpisode': evaluateEpisode,'ropePunishWeight':ropePunishWeight,'killZone':killZone,'killZoneofDistractor':killZoneofDistractor,'ropeLength':ropeLength}
    selctDict = {1:5,2:20,3:20}
    # fixedParameters = {'distractorNoise': 3.0,'evaluateEpisode': evaluateEpisode, 'evalNum': evalNum}
    generateTrajectoryLoadPath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
    # trajectoryDf = lambda condition: pd.read_pickle(generateTrajectoryLoadPath({'offset':condition['offset'],'hideId':condition['hideId'],'damping': condition['damping'], 'frictionloss': condition['frictionloss'], 'masterForce': condition['masterForce'], 'evalNum': evalNum,}))
    trajectoryDf = lambda condition: loadFromPickle(generateTrajectoryLoadPath({'offset':condition['offset'],'hideId':condition['hideId'],'damping': condition['damping'], 'frictionloss': condition['frictionloss'], 'masterForce': condition['masterForce'],'select':selctDict[condition['hideId']] , 'masterMass':condition['masterMass']}))

    # trajectoryDf = lambda condition: pd.read_pickle(generateTrajectoryLoadPath(condition))
    getTrajectory = lambda trajectoryDf: scaleTrajectoryInTime(scaleTrajectoryInSpace(trajectoryDf))
    # getTrajectory = lambda trajectoryDf: scaleTrajectoryInSpace(trajectoryDf)
    
    print('loading')
    stimulus = {conditionId:getTrajectory(trajectoryDf(condition))  for conditionId,condition in conditionsWithId}
    # print(stimulus[0][0][0])
    # print(stimulus[20][0][0])
    print('loding success')

    # print(stimulus[1][1])
    experimentValues = co.OrderedDict()
    experimentValues["name"] = input("Please enter your name:").capitalize()
    
    
    screenWidth = 800
    screenHeight = 800

    fullScreen = True
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

    
    drawState = DrawState(screen, circleSize, numOfAgent, positionIndex, drawBackGround)
    # drawStateWithRope = DrawStateWithRope(screen, circleSize, numOfAgent, positionIndex, ropeColor, drawBackground)    
   

    writerPath = os.path.join(resultsPath, experimentValues["name"]) + '.csv'
    writer = WriteDataFrameToCSV(writerPath)

    displayFrames = 500
    keysForCheck = {'f': 0, 'j': 1}
    checkHumanResponse = CheckHumanResponse(keysForCheck)
    trial = ChaseTrialMujoco(conditionsWithId,displayFrames, drawState, drawImage, stimulus, checkHumanResponse, colorSpace, numOfAgent, drawFixationPoint, drawText, drawImageClick, clickWolfImage, clickSheepImage, FPS)
    
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
