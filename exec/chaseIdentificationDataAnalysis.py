import pandas as pd
import os
import glob
DIRNAME = os.path.dirname(__file__)
import matplotlib.pyplot as plt
from collections import OrderedDict
# matplotlib.style.use('ggplot')
import numpy as np
from scipy.stats import ttest_ind
def readParametersFromDf(oneConditionDf):
    indexLevelNames = oneConditionDf.index.names
    parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
    return parameters
if __name__ == '__main__':

  

    # dataPath = os.path.join(DIRNAME,'..','..','ExpResult','exp2')
    dataPath = os.path.join(DIRNAME,'..','..','ExpResult','preExpMasterOffset')
    
    # dataPath = os.path.join(DIRNAME,'..','..','ExpResult','519','result')
    # dataPath = os.path.join(DIRNAME,'..','..','ExpResult','512')
    # dataPath = 'F:\DeskMirror\DRL\ExpResult\512\result'
    df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, '*.csv'))))
    # df.to_csv("all.csv")

    # print(df.head(6))
    nubOfSubj = len(df["name"].unique())
    statDF = pd.DataFrame()

    # df = df.loc[df['trail'].isin(range(41))]
    df.drop(df.columns[0], axis=1, inplace=True)  
    df1 = df.loc[df['hideId'].isin([3, 4])]
    df2 = df1
    # df2 = df1.loc[df1['offset']==0.0]
    # df2["hit"] = df2.apply(lambda row: 1 if row['response'] == 1 and row['chosenWolfIndex'] == 0.0 and row['chosenSheepIndex'] == 1.0 else 0, axis=1)
    df2["hit"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 0.0 and row['chosenSheepIndex'] == 1.0 else 0, axis=1)
    df2["masterWolf"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 2.0 and row['chosenSheepIndex'] == 0.0 else 0, axis=1)
    # df2["masterDistractor"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 2.0 and row['chosenSheepIndex'] > 2.0 else 0, axis=1)
    # df2["wolfDistractor"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 0.0 and row['chosenSheepIndex'] > 2.0 else 0, axis=1)
    df2["wolfMaster"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 2.0 and row['chosenSheepIndex'] == 0.0 else 0, axis=1)
    df2["masterSheep"] = df2.apply(lambda row: 1 if  row['chosenWolfIndex'] == 2.0 and row['chosenSheepIndex'] == 1.0 else 0, axis=1)

    # df2["miss"] = df2.apply(lambda row: 1 if row['response'] == 0  else 0, axis=1)
    # df2["hit"] = df2.apply(lambda row: 1 if row['response'] == 1 and row['chosenWolfIndex'] == 0.0 and row['chosenSheepIndex'] == 1.0 else 0, axis=1)


    # df.drop(df.columns[0], axis=1, inplace=True) 
    # df3 = df.loc[(df['hideId'] == 1.0) & (df['offset']==0.0)]
    # df3 = df.loc[(df['hideId'] == 1.0) ]
    # df3["correctRejection"] = df3.apply(lambda row: 1 if row['response'] == 0 else 0, axis=1)
    # print(df3)
    


    manipulatedVariables = OrderedDict()
    # manipulatedVariables['damping'] = [0.0, 0.5]  # [0.0, 1.0]
    # manipulatedVariables['frictionloss'] = [40, 50]  # [0.0, 0.2, 0.4]
    # manipulatedVariables['displayTime'] = [10, 15]  # [0.0, 2.0]
    # manipulatedVariables['offset'] = [-0.5,0.0,0.5,1.0]
    manipulatedVariables['damping'] = [0.0, 0.5]  # [0.0, 1.0]
    manipulatedVariables['frictionloss'] = [0.0, 1.0]  # [0.0, 0.2, 0.4]
    manipulatedVariables['masterForce'] = [0.0]  # [0.0, 2.0]
    manipulatedVariables['offset'] = [-0.5,0.0,0.5,1.0]

    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # def getMean(para,data,index):
    #     # selectSub = data.loc[(df['damping']==para['damping']) & (df['fps']==para['fps']) & (df['displayTime']==para['displayTime']) & (df['offset']==para['offset'])]
    #     # selectSub = data.loc[(data['damping']==para['damping']) & (data['fps']==para['fps']) & (data['displayTime']==para['displayTime']) ]
    #     selectSub = data.loc[(data['damping']==para['damping']) & (data['frictionloss']==para['frictionloss']) & (data['displayTime']==para['displayTime']) ]
    #     return selectSub[index].mean()
    # mesureMentFromDf = lambda df: getMean(readParametersFromDf(df),df2)
    # statisticsDf = toSplitFrame.groupby(levelNames).apply(mesureMentFromDf)
    # print(statisticsDf)

  

    # fig = plt.figure()
    # rowTitle = ['correctRejection']
    # numRows = nubOfSubj 
    # numColumns = len(rowTitle)
    # plotCounter = 1
    
    
    # for name, grp in df3.groupby('name'):
    #     # grp.index = grp.index.droplevel('name')

    #     # for fps, group in grp.groupby('fps'):
    #         # group.index = group.index.droplevel('fps')
    #     for measureMent in rowTitle:
    #         axForDraw = fig.add_subplot(numRows,numColumns,plotCounter)
    #         if plotCounter % numColumns == 0:
    #             axForDraw.set_ylabel('{}'.format(name[:5]))
    #         if plotCounter <= numColumns:
    #             # axForDraw.set_title('identification')
    #             axForDraw.set_title('{}'.format(measureMent))
    #         axForDraw.set_ylim(0, 1)
    #         # axForDraw.set_ylim(0, 15000)
    #         measureMentFromDf = lambda df: getMean(readParametersFromDf(df),grp,measureMent)
    #         statisticsDf = toSplitFrame.groupby(levelNames).apply(measureMentFromDf)
    #         # plt.ylabel('Distance between optimal and actual next position of sheep')
    #         # print(statisticsDf)
    #         statisticsDf.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,sharex =True,rot =15)
    #         # axForDraw.set_xticklabels(statisticsDf.index, )
    #         # plt.hlines(1/6, -1,8, colors = "r", linestyles = "dashed")
    #         # drawPerformanceLine(statisticsDf, axForDraw)
    #         # trainStepLevels = statisticsDf.index.get_level_values('trainSteps').values
    #         # axForDraw.plot(trainStepLevels, [1.18]*len(trainStepLevels), label='mctsTrainData')
    #         plotCounter += 1
    # # plt.sup(rowTitle[0])
    # # plt.suptitle(rowTitle[0])

    # plt.legend(loc='best')
    # plt.show()


    # for index in modelIndex
    
    
    # plot the results

    def drawPerformanceLine(dataDf, axForDraw):
        for masterForce, grp in dataDf.groupby('masterForce'):
            # grp.index = grp.index.droplevel('displayTime')
            meanSub = grp.groupby('offset')['hit'].mean()
            meanSub2 = grp.groupby('offset')['masterWolf'].mean()
            meanSub3 = grp.groupby('offset')['masterSheep'].mean()
            # meanSub4 = grp.groupby('offset')['masterDistractor'].mean()
            meanSub4 = grp.groupby('offset')['wolfMaster'].mean()


            # meanSub = grp.groupby('offset')['reactionTime'].mean()

            # print(meanSub)
            # meanSub = grp.groupby('offset')['correctRejection'].mean()
            # meanSub.plot(ax=axForDraw, label='masterForce={}'.format(masterForce), y='mean',marker='o', logx=False)
            print(meanSub,meanSub2)

            meanSub.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-sheep',color = 'green')
            meanSub2.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='master-wolf',bottom = meanSub,color = 'red')
            meanSub3.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='master-sheep',bottom = meanSub+meanSub2,color = 'yellow')
            meanSub4.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-master',bottom = meanSub+meanSub2+meanSub3,color = 'blue')
            # meanSub5.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-master',bottom = meanSub+meanSub2+meanSub3+meanSub4,color = 'black')
            # meanSub5.plot(kind = 'bar',ax=axForDraw, y='mean', logx=False,label='wolf-master',color = 'black')

            # plt.bar(displayTime, meanSub, label='displayTime={}'.format(displayTime), align='center')
            # plt.hlines(1/6, -0.5,1.5, colors = "r", linestyles = "dashed")
            axForDraw.set_xlim(-0.5, 3.5)
            # grp['hit'].plot(x='offset',ax=axForDraw, label='displayTime={}'.format(displayTime), y='mean',
                    # marker='o', logx=False)

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
            if plotCounter % numColumns == 1:
                axForDraw.set_ylabel(rowName+': {}'.format(rowValue))
            if plotCounter <= numColumns:
                axForDraw.set_title(columnName+': {}'.format(columnValue))
            axForDraw.set_ylim(0, 1)
            # axForDraw.set_ylim(0, 20000)

            # plt.ylabel('Distance between optimal and actual next position of sheep')
            drawPerformanceLine(group, axForDraw)
            # trainStepLevels = statisticsDf.index.get_level_values('trainSteps').values
            # axForDraw.plot(trainStepLevels, [1.18]*len(trainStepLevels), label='mctsTrainData')
            plotCounter += 1
    plt.suptitle('Identification,nubOfSubj={}'.format(nubOfSubj))

    plt.legend(loc='best')
    plt.show()


