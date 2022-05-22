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

  


    dataPath = os.path.join(DIRNAME,'..','..','..','ExpResult','Nov3VarMassAndForce')


    df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))))

    nubOfSubj = len(df["name"].unique())
    sublist = df["name"].unique()
    print(df["name"].unique())
    statDF = pd.DataFrame()
    
    df.drop(df.columns[0:1], axis=1, inplace=True) 
    print(df.columns) 
    df1 = df.loc[df['hideId'].isin([3, 2])]
    # df2 = df1
    
    df2 = df1.loc[df1['masterForce'].isin([2.0])]
    # targetId = 2
    # print(sublist[targetId])
    # print(df1)
    # print(df1[df1.index.duplicated()])
    # df2 = df1[df1['name'].str.contains (sublist[targetId])]
    # df2 = df1.loc[df1['offset']==0.0]
    # df2["hit"] = df2.apply(lambda row: 1 if row['response'] == 1 and row['chosenWolfIndex'] == 0.0 and row['chosenSheepIndex'] == 1.0 else 0, axis=1)
    df2["hit"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 0.0 and row['chosenSheepIndex'] == 1.0 else 0, axis=1)
    df2["masterWolf"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 2.0 and row['chosenSheepIndex'] == 0.0 else 0, axis=1)
    df2["masterDistractor"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 2.0 and row['chosenSheepIndex'] > 2.0 else 0, axis=1)
    df2["wolfDistractor"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 0.0 and row['chosenSheepIndex'] > 2.0 else 0, axis=1)
    df2["wolfMaster"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 0.0 and row['chosenSheepIndex'] == 2.0 else 0, axis=1)
    df2["masterSheep"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 2.0 and row['chosenSheepIndex'] == 1.0 else 0, axis=1)
    
    lableList = ['Normal','ReplaceMaster']
    
    # df2["condition"] = df2.apply(lambda row: 3 if  row['hideId'] == 2.0  else (1 if row['offset'] == 0.0 else 2), axis=1)
    df2["condition"] = df2.apply(lambda row: 2 if  row['hideId'] == 2.0  else 1, axis=1)
    df2["conditionLabel"] = df2.apply(lambda row: lableList[row['condition']-1] , axis=1)


    # targetId=2
    # print(sublist[0])
    # print(sublist[targetId])
    # df2 = df1.loc[df['name'] == sublist[0]]
    # df2 = df1.iloc[range(targetId*120,(targetId+1)*120)]

    manipulatedVariables = OrderedDict()
    manipulatedVariables['damping'] = [0.0,0.5]  # [0.0, 1.0]
    manipulatedVariables['frictionloss'] = [0.0,1.0]  # [0.0, 0.2, 0.4]
    manipulatedVariables['masterForce'] = [2.0]  # [0.0, 2.0]
    manipulatedVariables['hideId'] = [2,3]  # [0.0, 2.0]
    # manipulatedVariables['offset'] = [0.0]  # [0.0, 2.0]
    # manipulatedVariables['ropeLength'] = [0.06,0.09,0.12,0.15]

    xAxisVariables = 'conditionLabel'
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)
    
    def drawPerformanceLine(dataDf, axForDraw):
        for masterForce, grp in dataDf.groupby('masterForce'):
            print(grp)
            meanSub = grp.groupby(xAxisVariables)['hit'].mean()
            meanSub2 = grp.groupby(xAxisVariables)['masterWolf'].mean()
            meanSub3 = grp.groupby(xAxisVariables)['masterSheep'].mean()
            # normalGrp = grp.loc[df['hideId'].isin([3])].loc[df['offset'].isin([0.0])]
            # normalGrp = grp[(df['hideId']==3) & df['offset']==0.0]

            # replaceGrp = grp[df['hideId'].isin([3]) & df['offset'].isin([0.5])]
            # hideGrp = grp[df['hideId'].isin([2])]
            # print(normalGrp,replaceGrp,hideGrp)

        
            # toDrawData = [df.values[0][1]+0.5 for df in data]
            # toDrawData2 = [df.values[0][1]+1.5 for df in data]
            # toDrawData3 = 
            # plt.bar(range(len(toDrawData)),toDrawData,tick_label=lableList,color='green',logx=False,label='wolf-sheep')
            # plt.bar(range(len(toDrawData)),toDrawData2,bottom=toDrawData,tick_label=lableList,color='color',label='master(distrac1)-wolf')
            # plt.bar(range(len(toDrawData)),toDrawData3,bottom=toDrawData+toDrawData2,tick_label=lableList,color='yellow',label='master(distrac1)-sheep')
            meanSub.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-sheep',color = 'green',tick_label=lableList)
            meanSub2.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='master(distrac1)-wolf',bottom = meanSub,color = 'red',tick_label=lableList)
            meanSub3.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='master(distrac1)-sheep',bottom = meanSub+meanSub2,color = 'yellow',tick_label=lableList,rot =0)

            axForDraw.set_xlim(-0.5, 1.5)

    from src.loadChaseData import loadFromPickle,saveToPickle,GetSavePath   
    from exec.dataProcessing.evaluateChasingAngle import calculateChasingSubtletyBetweenTwoAgents,LoadTrajectories,ComputeStatistics,calculateDistanceBetweenTwoAgents

    fig = plt.figure()
    rowName = 'damping'
    columnName = 'frictionloss'
    # numRows = len(manipulatedVariables[rowName])
    # numColumns = len(manipulatedVariables[columnName])
    numRows = 1 
    numColumns =3
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
            axForDraw.set_title(columnName+': {}'.format(columnValue)+rowName+': {}'.format(rowValue))
            plotCounter += 1
            axForDraw.set_ylim(0, 1)
            # plt.ylabel('Distance between optimal and actual next position of sheep')
            
            drawPerformanceLine(group, axForDraw)
            # axForDraw2 = axForDraw.twinx()
            trajPara = {rowName:rowValue,columnName:columnValue} 
            # drawPerformanceLine2(trajPara, axForDraw)

            
    plt.suptitle('Identification(MasterForce=2.0MasterMass=1.0),nubOfSubj={}'.format(nubOfSubj))
    # plt.suptitle('Identification,Sub={}'.format(sublist[targetId]))

    plt.legend(loc='best')
    plt.show()


