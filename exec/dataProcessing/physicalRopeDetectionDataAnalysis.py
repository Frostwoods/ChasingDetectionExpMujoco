import os
import sys
import glob
DIRNAME = os.path.dirname(__file__)

dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName))
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from collections import OrderedDict
# matplotlib.style.use('ggplot')

def readParametersFromDf(oneConditionDf):
    indexLevelNames = oneConditionDf.index.names
    parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
    return parameters
if __name__ == '__main__':

  

    # dataPath = os.path.join(DIRNAME,'..','..','ExpResult','exp2')
    # dataPath = os.path.join(DIRNAME,'..','..','..','ExpResult','6.17','results')
    dataPath = os.path.join(DIRNAME,'..','..','..','ExpResult','6.21','phsycalResult')
    # dataPath = os.path.join(DIRNAME,'..','..','..','ExpResult','6.18','detectionResult')
    # dataPath = os.path.join(DIRNAME,'..','..','ExpResult','519','result')
    # dataPath = os.path.join(DIRNAME,'..','..','ExpResult','512')
    # dataPath = 'F:\DeskMirror\DRL\ExpResult\512\result'

    df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))))
    # df.to_csv(os.path.join(dataPath, 'allData.csv'), header=list(df.columns))

    nubOfSubj = len(df["name"].unique())
    sublist = df["name"].unique()
    print(sublist)
    statDF = pd.DataFrame()
    
    df.drop(df.columns[0], axis=1, inplace=True)  
    # df1 = df.loc[df['hideId'].isin([3, 4])]
    df1 = df.loc[df['tiedFirstId']==0]
    # df1 = df
    # targetId=2
    # print(sublist[0])
    # print(sublist[targetId])
    # df2 = df1.loc[df['name'] == sublist[0]]
    # df2 = df1.iloc[range(targetId*120,(targetId+1)*120)]
    df2 = df1.fillna(-1)
    print(df2)
    print(list(df2))
    # df1 = df.loc[df['hideId'].isin([1])]
    # filterList = [1,2,3,4]
    # df2 = df.loc[0:60]
    # df2 = df2.iloc[list(range(0,60))+list(range(120,180))+list(range(240,300))+list(range(360,420))+list(range(480,540))+list(range(600,660)),:]
    # df2 = df2.iloc[list(range(60,120))+list(range(180,240))+list(range(300,360))+list(range(420,480))+list(range(540,600))+list(range(660,720)),:]
    # df2 = df.loc[0:60,120:180,240:300,360:420,480:540,600:660]
    # df.fillna
    # print(df2())
    

    # df2 = df1.loc[df1['offset']==0.0]
    # df2["hit"] = df2.apply(lambda row: 1 if row['response'] == 1 and row['chosenWolfIndex'] == 0.0 and row['chosenSheepIndex'] == 1.0 else 0, axis=1)
    # df2["falseAlarm"] = df2.apply(lambda row:  row['response'] == 1 , axis=1)
    df2["hit"] = df2.apply(lambda row: 1 if  row['chosenActiveIndex'] == 0.0 and row['chosenPassiveIndex'] == 2.0 else 0, axis=1)
    # df2["hit"] = df2.apply(lambda row: 1 if  row['response'] == 1.0  else 0, axis=1)
    df2['accray'] = df2.apply(lambda row: 1 if   (row['tiedFirstId'] == 0.0 and row['response'] == 1.0) or (row['tiedFirstId'] == 2.0 and row['response'] == 0.0)  else 0, axis=1)
    # df2["masterWolf"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 2.0 and row['chosenSheepIndex'] == 0.0 else 0, axis=1)
    # df2["masterDistractor"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 2.0 and row['chosenSheepIndex'] > 2.0 else 0, axis=1)
    # df2["wolfDistractor"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 0.0 and row['chosenSheepIndex'] > 2.0 else 0, axis=1)
    # df2["wolfMaster"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 0.0 and row['chosenSheepIndex'] == 2.0 else 0, axis=1)
    # df2["masterSheep"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 2.0 and row['chosenSheepIndex'] == 1.0 else 0, axis=1)
   
    # df2["tiedPair"]=df2.apply(lambda row: str(row['tiedFirstId'])+'-'+str(row['tiedSencondId']))
    # agentlist=['wolf','sheep','master','distractor']
    # df2["tiedPair"]=df2.apply(lambda row:str( row['tiedFirstId']) +'-' + str(min(row['tiedSencondId'],3)), axis=1)
    # df2["tiedPair"]=df2.apply(lambda row:agentlist [row['tiedFirstId']]+'-' + agentlist[min(row['tiedSencondId'],3)], axis=1)
    df2["tiedPair"]=df2.apply(lambda row:'chasing absent' if row['hideId']==1 else "chasing present",axis=1)
    manipulatedVariables = OrderedDict()
    # df2.to_csv(os.path.join(dataPath, 'allSubData.csv'), header=list(df2.columns))
    # manipulatedVariables['damping'] = [0.0, 0.5]  # [0.0, 1.0]
    # manipulatedVariables['frictionloss'] = [0.0, 1.0]  # [0.0, 0.2, 0.4]
    # manipulatedVariables['masterForce'] = [0.0]  # [0.0, 2.0]
    # manipulatedVariables['offset'] = [-0.5,0.0,0.5,1.0]



    manipulatedVariables['damping'] = [0.5]  # [0.0, 1.0]
    manipulatedVariables['frictionloss'] = [1.0]  # [0.0, 0.2, 0.4]
    manipulatedVariables['masterForce'] = [1.0]  # [0.0, 2.0]
    manipulatedVariables['offset'] = [0.0]
    manipulatedVariables['tiedPairs'] = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]

    
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    # modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    # toSplitFrame = pd.DataFrame(index=modelIndex)
    
    def drawPerformanceLine(dataDf, axForDraw):
        for masterForce, grp in dataDf.groupby('masterForce'):

            # meanSub = grp.groupby('tiedPair')['falseAlarm'].mean()
            # meanSub = grp.groupby('tiedPair')['accray'].mean()
            meanSub = grp.groupby('tiedPair')['hit'].mean()
            # meanSub2 = grp.groupby('tiedPair')['masterWolf'].mean()
            # meanSub3 = grp.groupby('tiedPair')['masterSheep'].mean()
            # meanSub4 = grp.groupby('tiedPair')['masterDistractor'].mean()       
            # meanSub5 = grp.groupby('tiedPair')['wolfMaster'].mean()
            # meanSub6 = grp.groupby('tiedPair')['wolfDistractor'].mean()
            print(meanSub)
            print(meanSub.mean())
            # # print(meanSub,meanSub2)
            # meanSub.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-master',color = 'red')
            # meanSub.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='accracy',color = 'red')
            meanSub.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='Identification',color = 'red')
            # meanSub2.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='master-wolf',bottom = meanSub,color = 'red')
            # meanSub3.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='master-sheep',bottom = meanSub+meanSub2,color = 'yellow')
            # meanSub4.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='master-Distractor',bottom = meanSub+meanSub2+meanSub3,color = 'blue')
            # meanSub5.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-master',bottom = meanSub+meanSub2+meanSub3+meanSub4,color = 'black')
            # meanSub6.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-Distractor',bottom = meanSub+meanSub2+meanSub3+meanSub4+meanSub5,color = 'purple')

            # meanSub.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='falseAlarm',color = 'green')
            # meanSub.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-sheep',color = 'green')
            # meanSub2.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='master-wolf',bottom = meanSub,color = 'red')
            # meanSub3.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='master-sheep',bottom = meanSub+meanSub2,color = 'yellow')
            # meanSub4.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='master-Distractor',bottom = meanSub+meanSub2+meanSub3,color = 'blue')
            # meanSub5.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-master',bottom = meanSub+meanSub2+meanSub3,color = 'black')
            # meanSub6.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-Distractor',bottom = meanSub+meanSub2+meanSub3+meanSub4+meanSub5,color = 'purple')
            # meanSub5.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-master',color = 'black')
            plt.xticks(rotation=0)
            # axForDraw.set_xlim(-0.5, 3.5)

    from src.loadChaseData import loadFromPickle,saveToPickle,GetSavePath   
    from exec.dataProcessing.evaluateChasingAngle import calculateChasingSubtletyBetweenTwoAgents,LoadTrajectories,ComputeStatistics,calculateDistanceBetweenTwoAgents
    def drawPerformanceLine2(trajPara, axForDraw2):
        masterForce = 0.0
        distractorNoise = 3.0
        evaluateEpisode=120000

        manipulatedVariables = OrderedDict()    
        manipulatedVariables['offset'] = [-0.5,0.0,0.5,1.0]
        manipulatedVariables['hideId'] = [3,4]
        productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])

        levelNames = list(manipulatedVariables.keys())
        levelValues = list(manipulatedVariables.values())
        modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
        toSplitFrame = pd.DataFrame(index=modelIndex)

        dataFolder = os.path.join(dirName, '..','..', 'PataData')
        trajectoryDirectory= os.path.join(dataFolder,'offsetMasterTrajNewForExp')
        trajectoryExtension = '.pickle'
        selectNum = 10
        trajectoryFixedParameters = {'evaluateEpisode':evaluateEpisode,'damping':trajPara['damping'],'frictionloss':trajPara['frictionloss'],'masterForce':masterForce,'distractorNoise':distractorNoise,'select':selectNum}

        getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
        wolfId=0
        sheepId=1
        masterId=2
        distractorId=3
        nameList = ['wolf','sheep','master','distra']
        targetPair = [[wolfId,sheepId],[masterId,wolfId],[masterId,sheepId]]
        colorList = ['green','red','yellow']
        fuzzySearchParameterNames = []
        loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
        loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
        for angentPair,colorName in zip(targetPair,colorList):
            measurementFunction = lambda trajectory: calculateDistanceBetweenTwoAgents(trajectory,angentPair[0],angentPair[1])
            computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
            statisticsDfNew = toSplitFrame.groupby(levelNames).apply(computeStatistics)
            statisticsDf3 = statisticsDfNew.reset_index()

            df = statisticsDf3.groupby('offset').mean()
            print(df)
            df.plot(ax=axForDraw2, label='{}-{}'.format(nameList[angentPair[0]],nameList[angentPair[1]]), y='mean',marker='o', logx=False, color=colorName)
            axForDraw2.set_ylim(50, 300)
        print('1')
    fig = plt.figure()
    rowName = 'damping'
    columnName = 'frictionloss'
    numRows = len(manipulatedVariables[rowName])
    numColumns = len(manipulatedVariables[columnName])
    plotCounter = 1

    for rowValue, grp in df2.groupby(rowName):
        # grp.index = grp.index.droplevel('damping')

        for columnValue, group in grp.groupby(columnName):
            # group.index = group.index.droplevel('fps')

            axForDraw = fig.add_subplot(numRows,numColumns,plotCounter)
            # if plotCounter % numColumns == 1:
            #     axForDraw.set_ylabel(rowName+': {}'.format(rowValue))
            # if plotCounter <= numColumns:
            #     axForDraw.set_title(columnName+': {}'.format(columnValue))
            axForDraw.set_ylim(0, 1)
            # plt.ylabel('Distance between optimal and actual next position of sheep')

            drawPerformanceLine(group, axForDraw)
            # axForDraw2 = axForDraw.twinx()
            # trajPara = {rowName:rowValue,columnName:columnValue} 
            # drawPerformanceLine2(trajPara, axForDraw)

            plotCounter += 1
    plt.suptitle('Identification, nubOfSubj={}'.format(nubOfSubj))
    # plt.suptitle('hit, nubOfSubj={}'.format(nubOfSubj))
    # plt.suptitle('accracy, nubOfSubj={}'.format(nubOfSubj))
    
    # plt.suptitle('FalseAlarm, nubOfSubj={}'.format(nubOfSubj))
    # plt.suptitle('Identification,Sub No.{}={}'.format(targetId,sublist[targetId]))

    plt.legend(loc='best')
    plt.show()


