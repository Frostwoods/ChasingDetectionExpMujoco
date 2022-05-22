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
    # dataPath = os.path.join(DIRNAME,'..','..','..','ExpResult','preExpMasterOffset6.11')
    dataPath = os.path.join(DIRNAME,'..','..','..','ExpResult','10.20')
    # dataPath = os.path.join(DIRNAME,'..','..','ExpResult','519','result')
    # dataPath = os.path.join(DIRNAME,'..','..','ExpResult','512')
    # dataPath = 'F:\DeskMirror\DRL\ExpResult\512\result'

    df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))))

    nubOfSubj = len(df["name"].unique())
    sublist = df["name"].unique()
    print(df["name"].unique())
    statDF = pd.DataFrame()
    
    df.drop(df.columns[0], axis=1, inplace=True)  
    df1 = df.loc[df['hideId'].isin([3, 2])]
    df2 = df1
    # df2 = df1.loc[df['name'] == sublist[6]]
    # df2 = df1.loc[df1['offset']==0.0]
    # df2["hit"] = df2.apply(lambda row: 1 if row['response'] == 1 and row['chosenWolfIndex'] == 0.0 and row['chosenSheepIndex'] == 1.0 else 0, axis=1)
    df2["hit"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 0.0 and row['chosenSheepIndex'] == 1.0 else 0, axis=1)
    df2["masterWolf"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 2.0 and row['chosenSheepIndex'] == 0.0 else 0, axis=1)
    df2["masterDistractor"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 2.0 and row['chosenSheepIndex'] > 2.0 else 0, axis=1)
    df2["wolfDistractor"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 0.0 and row['chosenSheepIndex'] > 2.0 else 0, axis=1)
    df2["wolfMaster"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 0.0 and row['chosenSheepIndex'] == 2.0 else 0, axis=1)
    df2["masterSheep"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 2.0 and row['chosenSheepIndex'] == 1.0 else 0, axis=1)


    # targetId=2
    # print(sublist[0])
    # print(sublist[targetId])
    # df2 = df1.loc[df['name'] == sublist[0]]
    # df2 = df1.iloc[range(targetId*120,(targetId+1)*120)]

    manipulatedVariables = OrderedDict()
    manipulatedVariables['damping'] = [0.5]  # [0.0, 1.0]
    manipulatedVariables['frictionloss'] = [1.0]  # [0.0, 0.2, 0.4]
    manipulatedVariables['masterForce'] = [1.0]  # [0.0, 2.0]
    manipulatedVariables['hideId'] = [2,3]  # [0.0, 2.0]
    manipulatedVariables['ropeLength'] = [0.06,0.09,0.12,0.15]

    xAxisVariables = 'ropeLength'
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    
    def drawPerformanceLine(dataDf, axForDraw):
        for masterForce, grp in dataDf.groupby('masterForce'):

            meanSub = grp.groupby(xAxisVariables)['hit'].mean()
            meanSub2 = grp.groupby(xAxisVariables)['masterWolf'].mean()
            meanSub3 = grp.groupby(xAxisVariables)['masterSheep'].mean()
            meanSub4 = grp.groupby(xAxisVariables)['masterDistractor'].mean()       
            meanSub5 = grp.groupby(xAxisVariables)['wolfMaster'].mean()
            meanSub6 = grp.groupby(xAxisVariables)['wolfDistractor'].mean()

            print(meanSub,meanSub2)
            # meanSub.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-sheep',color = 'green')
            # meanSub2.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='master-wolf',bottom = meanSub,color = 'red')
            # meanSub3.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='master-sheep',bottom = meanSub+meanSub2,color = 'yellow')
            # meanSub4.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='master-Distractor',bottom = meanSub+meanSub2+meanSub3,color = 'blue')
            # meanSub5.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-master',bottom = meanSub+meanSub2+meanSub3+meanSub4,color = 'black')
            # meanSub6.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-Distractor',bottom = meanSub+meanSub2+meanSub3+meanSub4+meanSub5,color = 'purple')

            meanSub.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-sheep',color = 'green')
            meanSub2.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='master(distrac1)-wolf',bottom = meanSub,color = 'red')
            meanSub3.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='master(distrac1)-sheep',bottom = meanSub+meanSub2,color = 'yellow')
            # meanSub4.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='master-Distractor',bottom = meanSub+meanSub2+meanSub3,color = 'blue')
            # meanSub5.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-master',bottom = meanSub+meanSub2+meanSub3,color = 'black')
            # meanSub6.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-Distractor',bottom = meanSub+meanSub2+meanSub3+meanSub4+meanSub5,color = 'purple')
            # meanSub5.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-master',color = 'black')

            axForDraw.set_xlim(-0.5, 3.5)

    from src.loadChaseData import loadFromPickle,saveToPickle,GetSavePath   
    from exec.dataProcessing.evaluateChasingAngle import calculateChasingSubtletyBetweenTwoAgents,LoadTrajectories,ComputeStatistics,calculateDistanceBetweenTwoAgents

    fig = plt.figure()
    rowName = 'damping'
    columnName = 'hideId'
    numRows = len(manipulatedVariables[rowName])
    numColumns = len(manipulatedVariables[columnName])
    plotCounter = 1

    for rowValue, grp in df2.groupby(rowName):
        # grp.index = grp.index.droplevel('damping')

        for columnValue, group in grp.groupby(columnName):
            # group.index = group.index.droplevel('fps')

            axForDraw = fig.add_subplot(numRows,numColumns,plotCounter)
            if plotCounter % numColumns == 1:
                axForDraw.set_ylabel(rowName+': {}'.format(rowValue))
            if plotCounter <= numColumns:
                axForDraw.set_title(columnName+': {}'.format(columnValue))
            # axForDraw.set_ylim(0, 1)
            # plt.ylabel('Distance between optimal and actual next position of sheep')

            drawPerformanceLine(group, axForDraw)
            # axForDraw2 = axForDraw.twinx()
            trajPara = {rowName:rowValue,columnName:columnValue} 
            # drawPerformanceLine2(trajPara, axForDraw)

            plotCounter += 1
    plt.suptitle('Identification,nubOfSubj={}'.format(nubOfSubj))
    # plt.suptitle('Distance,Sub={}'.format(sublist[0]))

    plt.legend(loc='best')
    plt.show()


