import numpy as np
import random
import collections as co
import pygame as pg
from pygame import time
from pygame.color import THECOLORS
import os
import sys
from subprocess import Popen, PIPE
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class OpenReportTxt(object):
    def __init__(self, txtPath):
        self.txtPath = txtPath
        if not os.path.exists(txtPath):
            txt = open(txtPath, 'w')
            txt.close()

    def __call__(self):
        proc = Popen(['NOTEPAD', self.txtPath])
        proc.wait()
class CheckHumanResponseWithSpace():
    def __init__(self, keysForCheck):
        self.keysForCheck = keysForCheck

    def __call__(self, initialTime, results, pause, circleColorList):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    reactionTime = time.get_ticks() - initialTime
                    results['response'] = self.keysForCheck['space']
                    results['reactionTime'] = str(reactionTime)
                    pause = False
                # if event.key == pg.K_f:
                #     reactionTime = time.get_ticks() - initialTime
                #     results['response'] = self.keysForCheck['f']
                #     results['reactionTime'] = str(reactionTime)
                #     pause = False

                # if event.key == pg.K_j:
                #     reactionTime = time.get_ticks() - initialTime
                #     results['response'] = self.keysForCheck['j']
                #     results['reactionTime'] = str(reactionTime)
                #     pause = False

        pg.display.update()
        return results, pause
class CheckHumanSelectResponse():
    def __init__(self, keysForCheck):
        self.keysForCheck = keysForCheck

    def __call__(self, initialTime, results, pause, circleColorList):
        for event in pg.fastevent.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_LEFT:
                    reactionTime = time.get_ticks() - initialTime
                    results['response'] = self.keysForCheck['back']
                    results['reactionTime'] = str(reactionTime)
                    pause = True
                if event.key == pg.K_f:
                    reactionTime = time.get_ticks() - initialTime
                    results['response'] = self.keysForCheck['f']
                    results['reactionTime'] = str(reactionTime)
                    pause = False

                if event.key == pg.K_j:
                    reactionTime = time.get_ticks() - initialTime
                    results['response'] = self.keysForCheck['j']
                    results['reactionTime'] = str(reactionTime)
                    pause = False

        pg.display.update()
        return results, pause

class CheckHumanResponse():
    def __init__(self, keysForCheck):
        self.keysForCheck = keysForCheck

    def __call__(self, initialTime, results, pause, circleColorList):
        for event in pg.fastevent.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN:
                # if event.key == pg.K_SPACE:
                #     reactionTime = time.get_ticks() - initialTime
                #     results['response'] = self.keysForCheck['f']
                #     results['reactionTime'] = str(reactionTime)
                #     pause = False
                if event.key == pg.K_f:
                    reactionTime = time.get_ticks() - initialTime
                    results['response'] = self.keysForCheck['f']
                    results['reactionTime'] = str(reactionTime)
                    pause = False

                if event.key == pg.K_j:
                    reactionTime = time.get_ticks() - initialTime
                    results['response'] = self.keysForCheck['j']
                    results['reactionTime'] = str(reactionTime)
                    pause = False

        pg.display.update()
        return results, pause
# class SelectChaseTrialMujoco():
#     def __init__(self, conditionsWithId,displayFrames, drawState, drawImage, stimulus, checkHumanResponse, colorSpace, numOfAgent, drawFixationPoint, drawText, drawImageClick, clickWolfImage, clickSheepImage, fps):
#         self.conditionsWithId = conditionsWithId
#         self.displayFrames = displayFrames
#         self.stimulus = stimulus
#         self.drawState = drawState
#         self.drawImage = drawImage
#         self.checkHumanResponse = checkHumanResponse
#         self.colorSpace = colorSpace
#         self.numOfAgent = numOfAgent
#         self.drawFixationPoint = drawFixationPoint
#         self.drawText = drawText
#         self.drawImageClick = drawImageClick
#         self.clickWolfImage = clickWolfImage
#         self.clickSheepImage = clickSheepImage
#         self.fps = fps


#     def __call__(self, condition):
#         results = co.OrderedDict()
#         results["trail"] = ''

#         conditionId = condition['conditonId']
#         conditionPara = self.conditionsWithId[conditionId][1]
#         print(conditionPara,conditionId)
#         results['conditionId'] = conditionId
#         results['damping'] = conditionPara['damping']
#         results['frictionloss'] = conditionPara['frictionloss']
#         results['masterForce'] = conditionPara['masterForce']

#         results['trajetoryIndex'] = condition['trajetoryIndex']

#         results['response'] = ''
#         results['reactionTime'] = ''


#         trajetoryData = self.stimulus[int(conditionId)][int(condition['trajetoryIndex'])]
#         print(len(trajetoryData))
#         random.shuffle(self.colorSpace)
#         circleColorList = self.colorSpace[:self.numOfAgent]

#         pause = True
#         initialTime = time.get_ticks()
#         fpsClock = pg.time.Clock()
#         while pause:
#             pg.mouse.set_visible(False)
#             self.drawFixationPoint()
#             tialTime = time.get_ticks()
#             Time = time.get_ticks()
#             t = 0
#             while t < self.displayFrames:
#                 state = trajetoryData[t]
#                 fpsClock.tick(self.fps)

#                 screen = self.drawState(state, circleColorList)
#                 self.drawText('Please Response Now!', (screen.get_width() / 4, screen.get_height() / 1.2))
#                 # screen = self.drawState(state, condition, circleColorList)
#                 # screen = self.drawStateWithRope(state, condition, self.colorSpace)

#                 results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)
#                 if not pause:
#                     break

#                 if t == self.displayFrames - 1:
#                     print(t,time.get_ticks()-tialTime)
#                     self.drawText('Please Response Now!', (screen.get_width() / 4, screen.get_height() / 1.2))
#                     while pause:
#                         results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)
                
#                 # print(time.get_ticks()-Time)
#                 Time = time.get_ticks()
#                 t = t +1
#                 if results['response'] == -1:
#                     results['response'] =''
#                     t = 0
#                     pg.time.wait(500)

#         return results
class SelectTrialMujoco():
    def __init__(self, conditionsWithId,displayFrames, drawState, drawImage, stimulus, checkHumanResponse, colorSpace, numOfAgent, drawFixationPoint, drawText, drawImageClick, clickWolfImage, clickSheepImage, fps):
        self.conditionsWithId = conditionsWithId
        self.displayFrames = displayFrames
        self.stimulus = stimulus
        self.drawState = drawState
        self.drawImage = drawImage
        self.checkHumanResponse = checkHumanResponse
        self.colorSpace = colorSpace
        self.numOfAgent = numOfAgent
        self.drawFixationPoint = drawFixationPoint
        self.drawText = drawText
        self.drawImageClick = drawImageClick
        self.clickWolfImage = clickWolfImage
        self.clickSheepImage = clickSheepImage
        self.fps = fps


    def __call__(self, condition,trajIndex):
        results = co.OrderedDict()
        results["trail"] = ''

        conditionId = condition['conditionId']
        conditionPara = self.conditionsWithId[conditionId][1]
        # results['conditionId'] = conditionId
        for key, values in conditionPara.items():
            results[key] = values
        # print(conditionPara,conditionId)
        # results['conditionId'] = conditionId
        # results['damping'] = conditionPara['damping']
        # results['frictionloss'] = conditionPara['frictionloss']
        # results['masterForce'] = conditionPara['masterForce']
        # results['offset'] = conditionPara['offset']
        # results['hideId(1:hidesheep;3,4:hideOneDistractor'] = conditionPara['hideId']
        results['trajetoryIndex'] = trajIndex
        results['response'] = ''
       
        trajetoryData = self.stimulus[int(conditionId)][int(trajIndex)]
        # random.shuffle(self.colorSpace)
        circleColorList = self.colorSpace[:self.numOfAgent]

        pause = True
        initialTime = time.get_ticks()
        fpsClock = pg.time.Clock()
        while pause:
            pg.mouse.set_visible(False)
            self.drawFixationPoint()
            tialTime = time.get_ticks()
            Time = time.get_ticks()
            delayTime = 0
            t = 0
            if t == 0 :
                print(condition,'TrajIndex',trajIndex)
            while t < self.displayFrames:
            # for t in range(self.displayFrames):
                state = trajetoryData[t + delayTime]
                fpsClock.tick(self.fps)

                screen = self.drawState(state, circleColorList)
                paraY = screen.get_height() / 8
                # for key, values in conditionPara.items():
                #     self.drawText(key + str(values), (0,paraY ))
                #     paraY = paraY + screen.get_height() / 16
                    
                # screen = self.drawState(state, condition, circleColorList)
                # screen = self.drawStateWithRope(state, condition, self.colorSpace)


                results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)
                if not pause:
                    break

                if t == self.displayFrames - 1:
                    print(t,time.get_ticks()-tialTime)
                    self.drawText('Please Response Now!', (screen.get_width() / 2, screen.get_height() / 1.2))
                    while pause:
                        results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)
                        if results['response'] == -1:
                             break
                # print(time.get_ticks()-Time)
                Time = time.get_ticks()
                t = t +1
                if results['response'] == -1:
                    results['response'] =''
                    t = 0
                    pg.time.wait(500)

        return results,trajetoryData
class ChaseTrialMujocoWithLineRopeForReport():
    def __init__(self, conditionsWithId,displayFrames, drawState, drawImage, stimulus, checkHumanResponse, colorSpace, numOfAgent, drawFixationPoint, drawText, drawImageClick, clickWolfImage, clickSheepImage, fps):
        self.conditionsWithId = conditionsWithId
        self.displayFrames = displayFrames
        self.stimulus = stimulus
        self.drawState = drawState
        self.drawImage = drawImage
        self.checkHumanResponse = checkHumanResponse
        self.colorSpace = colorSpace
        self.numOfAgent = numOfAgent
        self.drawFixationPoint = drawFixationPoint
        self.drawText = drawText
        self.drawImageClick = drawImageClick
        self.clickWolfImage = clickWolfImage
        self.clickSheepImage = clickSheepImage
        self.fps = fps


    def __call__(self, condition):
        results = co.OrderedDict()
        results["trail"] = ''
      
        conditionId = condition['conditonId']
        conditionPara = self.conditionsWithId[conditionId][1]
        agentIdList = list(range(5))
        del agentIdList[conditionPara['hideId']]
        tiedId = conditionPara['tiedPairs']
        results['conditionId'] = conditionId
        results['trajetoryIndex'] = condition['trajetoryIndex']
        agentIdForDraw = conditionPara['agentIdForDraw']
        # for key, values in conditionPara.items():
        #     if key != 'tiedPairs':
        #         results[key] = values
        results['tiedFirstId'] = agentIdList [tiedId[0]]
        results['tiedSencondId'] =agentIdList [tiedId[1]]
        if conditionId<12:
            isMirror = conditionId//8
            rotationAngle =np.mod(conditionId//2,4)*90
        else:
            isMirror = conditionId//16
            rotationAngle = np.mod(conditionId,4)* 90
        # randomSeed = np.mod(conditionId, 8)
        results['rotationAngle'] = rotationAngle    
        results['mirrror'] = isMirror

        print(condition)
        # results['conditionId'] = conditionId
        # results['damping'] = conditionPara['damping']
        # results['frictionloss'] = conditionPara['frictionloss']
        # results['masterForce'] = conditionPara['masterForce']
        # results['offset'] = conditionPara['offset']
        # results['hideId(1:hidesheep;3,4:hideOneDistractor'] = conditionPara['hideId']
        results['trajetoryIndex'] = condition['trajetoryIndex']

        results['response'] = ''
        results['reactionTime'] = ''
        results['chosenWolfIndex'] = ''
        results['chosenSheepIndex'] = ''

        trajetoryData = self.stimulus[int(conditionId)][int(condition['trajetoryIndex'])]
        print(len(trajetoryData))
        # random.shuffle(self.colorSpace)
        circleColorList = [self.colorSpace[id] for id in agentIdList]
      
        pause = True
        initialTime = time.get_ticks()
        fpsClock = pg.time.Clock()
        while pause:
            pg.mouse.set_visible(False)
            self.drawFixationPoint()
            tialTime = time.get_ticks()
            Time = time.get_ticks()
            for t in range(self.displayFrames):
                state = [trajetoryData[t][agent] for agent in agentIdForDraw]
                fpsClock.tick(self.fps)

                screen = self.drawState(state, circleColorList,tiedId)
                # screen = self.drawState(state, condition, circleColorList)
                # screen = self.drawStateWithRope(state, condition, self.colorSpace)

                results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)
                if not pause:
                    break

                if t == self.displayFrames - 1:
                    print(t,time.get_ticks()-tialTime)
                    self.drawText('Please Response Now!', (screen.get_width() / 4, screen.get_height() / 1.2))
                    while pause:
                        results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)
                # print(time.get_ticks()-Time)
                Time = time.get_ticks()
            if results['response'] == 1:
                print('select0',condition['trajetoryIndex'])
                pg.mouse.set_visible(True)
                chosenWolfIndex = self.drawImageClick(self.clickWolfImage, "W", circleColorList)
                chosenSheepIndex = self.drawImageClick(self.clickSheepImage, 'S', circleColorList)
                results['chosenWolfIndex'] = agentIdList[chosenWolfIndex]
                results['chosenSheepIndex'] = agentIdList[chosenSheepIndex]
                pg.time.wait(500)
        print(results)
        return results
class ChaseTrialMujocoWithLineRope():
    def __init__(self, conditionsWithId,displayFrames, drawState, drawImage, stimulus, checkHumanResponse, colorSpace, numOfAgent, drawFixationPoint, drawText, drawImageClick, clickWolfImage, clickSheepImage, fps):
        self.conditionsWithId = conditionsWithId
        self.displayFrames = displayFrames
        self.stimulus = stimulus
        self.drawState = drawState
        self.drawImage = drawImage
        self.checkHumanResponse = checkHumanResponse
        self.colorSpace = colorSpace
        self.numOfAgent = numOfAgent
        self.drawFixationPoint = drawFixationPoint
        self.drawText = drawText
        self.drawImageClick = drawImageClick
        self.clickWolfImage = clickWolfImage
        self.clickSheepImage = clickSheepImage
        self.fps = fps


    def __call__(self, condition):
        results = co.OrderedDict()
        results["trail"] = ''
      
        conditionId = condition['conditonId']
        conditionPara = self.conditionsWithId[conditionId][1]
        agentIdList = list(range(5))
        del agentIdList[conditionPara['hideId']]
        tiedId = conditionPara['tiedPairs']
        results['conditionId'] = conditionId
        results['trajetoryIndex'] = condition['trajetoryIndex']

        for key, values in conditionPara.items():
            if key != 'tiedPairs':
                results[key] = values
        results['tiedFirstId'] = agentIdList [tiedId[0]]
        results['tiedSencondId'] =agentIdList [tiedId[1]]
        if conditionId<12:
            isMirror = conditionId//8
            rotationAngle =np.mod(conditionId//2,4)*90
        else:
            isMirror = conditionId//16
            rotationAngle = np.mod(conditionId,4)* 90
        # randomSeed = np.mod(conditionId, 8)
        results['rotationAngle'] = rotationAngle    
        results['mirrror'] = isMirror

        print(conditionPara,conditionId)
        # results['conditionId'] = conditionId
        # results['damping'] = conditionPara['damping']
        # results['frictionloss'] = conditionPara['frictionloss']
        # results['masterForce'] = conditionPara['masterForce']
        # results['offset'] = conditionPara['offset']
        # results['hideId(1:hidesheep;3,4:hideOneDistractor'] = conditionPara['hideId']
        results['trajetoryIndex'] = condition['trajetoryIndex']

        results['response'] = ''
        results['reactionTime'] = ''
        results['chosenWolfIndex'] = ''
        results['chosenSheepIndex'] = ''

        trajetoryData = self.stimulus[int(conditionId)][int(condition['trajetoryIndex'])]
        print(len(trajetoryData))
        random.shuffle(self.colorSpace)
        circleColorList = self.colorSpace[:self.numOfAgent]
      
        pause = True
        initialTime = time.get_ticks()
        fpsClock = pg.time.Clock()
        while pause:
            pg.mouse.set_visible(False)
            self.drawFixationPoint()
            tialTime = time.get_ticks()
            Time = time.get_ticks()
            for t in range(self.displayFrames):
                state = trajetoryData[t]
                fpsClock.tick(self.fps)

                screen = self.drawState(state, circleColorList,tiedId)
                # screen = self.drawState(state, condition, circleColorList)
                # screen = self.drawStateWithRope(state, condition, self.colorSpace)

                results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)
                if not pause:
                    break

                if t == self.displayFrames - 1:
                    print(t,time.get_ticks()-tialTime)
                    self.drawText('Please Response Now!', (screen.get_width() / 4, screen.get_height() / 1.2))
                    while pause:
                        results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)
                # print(time.get_ticks()-Time)
                Time = time.get_ticks()
            if results['response'] == 1:
                pg.mouse.set_visible(True)
                chosenWolfIndex = self.drawImageClick(self.clickWolfImage, "W", circleColorList)
                chosenSheepIndex = self.drawImageClick(self.clickSheepImage, 'S', circleColorList)
                results['chosenWolfIndex'] = agentIdList[chosenWolfIndex]
                results['chosenSheepIndex'] = agentIdList[chosenSheepIndex]
                pg.time.wait(500)
        print(results)
        return results
class ChaseTrialMujocoWithLineRopePhysical():
    def __init__(self, conditionsWithId,displayFrames, drawState, drawImage, stimulus, checkHumanResponse, colorSpace, numOfAgent, drawFixationPoint, drawText, drawImageClick, clickWolfImage, clickSheepImage, fps):
        self.conditionsWithId = conditionsWithId
        self.displayFrames = displayFrames
        self.stimulus = stimulus
        self.drawState = drawState
        self.drawImage = drawImage
        self.checkHumanResponse = checkHumanResponse
        self.colorSpace = colorSpace
        self.numOfAgent = numOfAgent
        self.drawFixationPoint = drawFixationPoint
        self.drawText = drawText
        self.drawImageClick = drawImageClick
        self.clickWolfImage = clickWolfImage
        self.clickSheepImage = clickSheepImage
        self.fps = fps


    def __call__(self, condition):
        results = co.OrderedDict()
        results["trail"] = ''      
        conditionId = condition['conditonId']
        conditionPara = self.conditionsWithId[conditionId][1]
        agentIdList = list(range(5))
        del agentIdList[conditionPara['hideId']]
        if (conditionPara['hideId'] == 4) and (conditionPara['tiedPairs'][1] == 4):
            tiedId = [agentIdList.index(agentId) for agentId in [conditionPara['tiedPairs'][0],3]]
            results['tiedFirstId'] = conditionPara['tiedPairs'][0]
            results['tiedSencondId'] = 3
        else:
            tiedId = [agentIdList.index(agentId) for agentId in conditionPara['tiedPairs']]
            results['tiedFirstId'] = conditionPara['tiedPairs'][0]
            results['tiedSencondId'] = conditionPara['tiedPairs'][1]
        results['conditionId'] = conditionId
        results['trajetoryIndex'] = condition['trajetoryIndex']

        for key, values in conditionPara.items():
            if key != 'tiedPairs':
                results[key] = values

       

        if conditionId<12:
            isMirror = conditionId//8
            rotationAngle =np.mod(conditionId//2,4)*90
        else:
            isMirror = conditionId//16
            rotationAngle = np.mod(conditionId,4)* 90
        # randomSeed = np.mod(conditionId, 8)
        results['rotationAngle'] = rotationAngle    
        results['mirrror'] = isMirror

        print(conditionPara,conditionId)
        # results['conditionId'] = conditionId
        # results['damping'] = conditionPara['damping']
        # results['frictionloss'] = conditionPara['frictionloss']
        # results['masterForce'] = conditionPara['masterForce']
        # results['offset'] = conditionPara['offset']
        # results['hideId(1:hidesheep;3,4:hideOneDistractor'] = conditionPara['hideId']
        results['trajetoryIndex'] = condition['trajetoryIndex']

        results['response'] = ''
        results['reactionTime'] = ''
        results['chosenWolfIndex'] = ''
        results['chosenSheepIndex'] = ''

        trajetoryData = self.stimulus[int(conditionId)][int(condition['trajetoryIndex'])]
        print(len(trajetoryData))
        random.shuffle(self.colorSpace)
        circleColorList = self.colorSpace[:self.numOfAgent]
      
        pause = True
        initialTime = time.get_ticks()
        fpsClock = pg.time.Clock()
        while pause:
            pg.mouse.set_visible(False)
            self.drawFixationPoint()
            tialTime = time.get_ticks()
            Time = time.get_ticks()
            for t in range(self.displayFrames):
                state = trajetoryData[t]
                fpsClock.tick(self.fps)

                screen = self.drawState(state, circleColorList,tiedId)
                # screen = self.drawState(state, condition, circleColorList)
                # screen = self.drawStateWithRope(state, condition, self.colorSpace)

                results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)
                if not pause:
                    break

                if t == self.displayFrames - 1:
                    print(t,time.get_ticks()-tialTime)
                    self.drawText('Please Response Now!', (screen.get_width() / 4, screen.get_height() / 1.2))
                    while pause:
                        results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)
                # print(time.get_ticks()-Time)
                Time = time.get_ticks()
            if results['response'] == 1:
                pg.mouse.set_visible(True)
                chosenWolfIndex = self.drawImageClick(self.clickWolfImage, "A", circleColorList)
                chosenSheepIndex = self.drawImageClick(self.clickSheepImage, 'P', circleColorList)
                results['chosenActiveIndex'] = agentIdList[chosenWolfIndex]
                results['chosenPassiveIndex'] = agentIdList[chosenSheepIndex]
                pg.time.wait(500)
        print(results)
        return results

class ChaseTrialMujoco():
    def __init__(self, conditionsWithId,displayFrames, drawState, drawImage, stimulus, checkHumanResponse, colorSpace, numOfAgent, drawFixationPoint, drawText, drawImageClick, clickWolfImage, clickSheepImage, fps,reactionWindowStart=50):
        self.conditionsWithId = conditionsWithId
        self.displayFrames = displayFrames
        self.stimulus = stimulus
        self.drawState = drawState
        self.drawImage = drawImage
        self.checkHumanResponse = checkHumanResponse
        self.colorSpace = colorSpace
        self.numOfAgent = numOfAgent
        self.drawFixationPoint = drawFixationPoint
        self.drawText = drawText
        self.drawImageClick = drawImageClick
        self.clickWolfImage = clickWolfImage
        self.clickSheepImage = clickSheepImage
        self.fps = fps
        self.reactionWindowStart = reactionWindowStart


    def __call__(self, condition):
        results = co.OrderedDict()
        results["trail"] = ''

        conditionId = condition['conditonId']
        conditionPara = self.conditionsWithId[conditionId][1]
        results['conditionId'] = conditionId
        for key, values in conditionPara.items():
            results[key] = values
        print(conditionPara,conditionId)
        # results['conditionId'] = conditionId
        # results['damping'] = conditionPara['damping']
        # results['frictionloss'] = conditionPara['frictionloss']
        # results['masterForce'] = conditionPara['masterForce']
        # results['offset'] = conditionPara['offset']
        # results['hideId(1:hidesheep;3,4:hideOneDistractor'] = conditionPara['hideId']
        results['trajetoryIndex'] = condition['trajetoryIndex']

        results['response'] = ''
        results['reactionTime'] = ''
        results['chosenWolfIndex'] = ''
        results['chosenSheepIndex'] = ''

        trajetoryData = self.stimulus[int(conditionId)][int(condition['trajetoryIndex'])]
        print(len(trajetoryData))
        random.shuffle(self.colorSpace)
        circleColorList = self.colorSpace[:self.numOfAgent]

        pause = True
        initialTime = time.get_ticks()
        fpsClock = pg.time.Clock()
        while pause:
            pg.mouse.set_visible(False)
            self.drawFixationPoint()
            tialTime = time.get_ticks()
            Time = time.get_ticks()
            for t in range(self.displayFrames):
                state = trajetoryData[t]
                fpsClock.tick(self.fps)

                screen = self.drawState(state, circleColorList)
                # screen = self.drawState(state, condition, circleColorList)
                # screen = self.drawStateWithRope(state, condition, self.colorSpace)
                if t > self.reactionWindowStart:
                    results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)
                if not pause:
                    break

                if t == self.displayFrames - 1:
                    print(t,time.get_ticks()-tialTime)
                    self.drawText('Please Response Now!', (screen.get_width() / 4, screen.get_height() / 1.2))
                    while pause:
                        results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)
                # print(time.get_ticks()-Time)
                Time = time.get_ticks()
            if results['response'] == 1:
                pg.mouse.set_visible(True)
                chosenWolfIndex = self.drawImageClick(self.clickWolfImage, "W", circleColorList)
                chosenSheepIndex = self.drawImageClick(self.clickSheepImage, 'S', circleColorList)
                results['chosenWolfIndex'] = chosenWolfIndex
                results['chosenSheepIndex'] = chosenSheepIndex
                pg.time.wait(500)

        return results


class ChaseTrial():
    def __init__(self, condtionList, displayFrames, drawState, drawImage, stimulus, checkHumanResponse, colorSpace, numOfAgent, drawFixationPoint, drawText, drawImageClick, clickWolfImage, clickSheepImage, fps):
        self.displayFrames = displayFrames
        self.stimulus = stimulus
        self.drawState = drawState
        self.drawImage = drawImage
        self.checkHumanResponse = checkHumanResponse
        self.colorSpace = colorSpace
        self.numOfAgent = numOfAgent
        self.drawFixationPoint = drawFixationPoint
        self.drawText = drawText
        self.drawImageClick = drawImageClick
        self.clickWolfImage = clickWolfImage
        self.clickSheepImage = clickSheepImage
        self.fps = fps

        self.conditionList = condtionList

    def __call__(self, condition):
        results = co.OrderedDict()
        results["trail"] = ''
        results['condition'] = condition['ChaseCondition']
        results['trajetoryIndex'] = condition['TrajIndex']
        results['response'] = ''
        results['reactionTime'] = ''
        results['chosenWolfIndex'] = ''
        results['chosenSheepIndex'] = ''

        trajetoryData = self.stimulus[int(condition['ChaseCondition'])][int(condition['TrajIndex'])]
        random.shuffle(self.colorSpace)
        circleColorList = self.colorSpace[:self.numOfAgent]

        pause = True
        initialTime = time.get_ticks()
        fpsClock = pg.time.Clock()
        while pause:
            pg.mouse.set_visible(False)
            self.drawFixationPoint()
            for t in range(self.displayFrames):
                state = trajetoryData[t]
                fpsClock.tick(self.fps)

                screen = self.drawState(state, circleColorList)
                # screen = self.drawState(state, condition, circleColorList)
                # screen = self.drawStateWithRope(state, condition, self.colorSpace)

                results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)
                if not pause:
                    break

                if t == self.displayFrames - 1:
                    self.drawText('Please Response Now!', (screen.get_width() / 4, screen.get_height() / 1.2))
                    while pause:
                        results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)

            if results['response'] == 1:
                pg.mouse.set_visible(True)
                chosenWolfIndex = self.drawImageClick(self.clickWolfImage, "W", circleColorList)
                chosenSheepIndex = self.drawImageClick(self.clickSheepImage, 'S', circleColorList)
                results['chosenWolfIndex'] = chosenWolfIndex
                results['chosenSheepIndex'] = chosenSheepIndex
                pg.time.wait(500)

        return results


class ChaseTrialWithRope():
    def __init__(self, conditionValues, displayFrames, drawStateWithRope, drawImage, stimulus, checkHumanResponse, colorSpace, numOfAgent, drawFixationPoint, drawText, drawImageClick, clickWolfImage, clickSheepImage, fps):
        self.displayFrames = displayFrames
        self.stimulus = stimulus
        self.drawStateWithRope = drawStateWithRope
        self.drawImage = drawImage
        self.checkHumanResponse = checkHumanResponse
        self.colorSpace = colorSpace
        self.numOfAgent = numOfAgent
        self.drawFixationPoint = drawFixationPoint
        self.drawText = drawText
        self.drawImageClick = drawImageClick
        self.clickWolfImage = clickWolfImage
        self.clickSheepImage = clickSheepImage
        self.fps = fps

        self.conditionValues = conditionValues

    def __call__(self, condition):
        results = co.OrderedDict()
        results["trail"] = ''
        results['condition'] = condition['ChaseCondition']
        results['trajetoryIndex'] = condition['TrajIndex']
        results['response'] = ''
        results['reactionTime'] = ''
        results['chosenWolfIndex'] = ''
        results['chosenSheepIndex'] = ''

        trajetoryData = self.stimulus[int(condition['ChaseCondition'])][int(condition['TrajIndex'])]
        random.shuffle(self.colorSpace)
        circleColorList = self.colorSpace[:self.numOfAgent]

        pause = True
        initialTime = time.get_ticks()
        fpsClock = pg.time.Clock()
        while pause:
            pg.mouse.set_visible(False)
            self.drawFixationPoint()
            for t in range(self.displayFrames):
                state = trajetoryData[t]

                fpsClock.tick(self.fps)

                screen = self.drawStateWithRope(state, self.conditionValues[int(condition['ChaseCondition'])], circleColorList)
                # screen = self.drawState(state, condition, circleColorList)
                # screen = self.drawStateWithRope(state, condition, self.colorSpace)

                results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)
                if not pause:
                    break

                if t == self.displayFrames - 1:
                    self.drawText('Please Response Now!', (screen.get_width() / 4, screen.get_height() / 1.2))

                    while pause:
                        results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)

            if results['response'] == 1:
                pg.mouse.set_visible(True)
                chosenWolfIndex = self.drawImageClick(self.clickWolfImage, "W", circleColorList)
                chosenSheepIndex = self.drawImageClick(self.clickSheepImage, 'S', circleColorList)
                results['chosenWolfIndex(0)'] = chosenWolfIndex
                results['chosenSheepIndex(1)'] = chosenSheepIndex
                pg.time.wait(500)

        return results


class ChaseTrialWithTraj:
    def __init__(self, fps, colorSpace,
                 drawState, saveImage, saveImageDir):

        self.fps = fps
        self.colorSpace = colorSpace
        self.drawState = drawState
        self.saveImage = saveImage
        self.saveImageDir = saveImageDir

    def __call__(self, trajectoryData):
        fpsClock = pg.time.Clock()

        for timeStep in range(len(trajectoryData)):
            state = trajectoryData[timeStep]
            fpsClock.tick(200)
            screen = self.drawState(state, self.colorSpace)

            if self.saveImage == True:
                if not os.path.exists(self.saveImageDir):
                    os.makedirs(self.saveImageDir)
                pg.image.save(screen, self.saveImageDir + '/' + format(timeStep, '04') + ".png")

        return


class ChaseTrialWithRopeTraj:
    def __init__(self, fps, colorSpace, drawStateWithRope, saveImage, saveImageDir):
        self.fps = fps
        self.colorSpace = colorSpace
        self.drawStateWithRope = drawStateWithRope
        self.saveImage = saveImage
        self.saveImageDir = saveImageDir

    def __call__(self, trajectoryData, condition):
        fpsClock = pg.time.Clock()

        for timeStep in range(len(trajectoryData)):
            state = trajectoryData[timeStep]
            fpsClock.tick(200)
            screen = self.drawStateWithRope(state, condition, self.colorSpace)

            if self.saveImage == True:
                if not os.path.exists(self.saveImageDir):
                    os.makedirs(self.saveImageDir)
                pg.image.save(screen, self.saveImageDir + '/' + format(timeStep, '04') + ".png")

        return


class ReportTrial():
    def __init__(self, conditionList, displayFrames, drawState, drawImage, stimulus, colorSpace, numOfAgent, drawFixationPoint, drawText, fps, reportInstrucImage, openReportTxt):
        self.conditionList = conditionList
        self.displayFrames = displayFrames
        self.stimulus = stimulus
        self.drawState = drawState
        self.drawImage = drawImage
        self.colorSpace = colorSpace
        self.numOfAgent = numOfAgent
        self.drawFixationPoint = drawFixationPoint
        self.drawText = drawText
        self.fps = fps
        self.reportInstrucImage = reportInstrucImage
        self.openReportTxt = openReportTxt

    def __call__(self, condition):
        results = co.OrderedDict()
        results["trail"] = ''
        results['condition'] = condition['ChaseConditon']
        results['trajetoryIndex'] = condition['TrajIndex']
        trajetoryData = self.stimulus[int(condition['ChaseConditon'])][int(condition['TrajIndex'])]
        random.shuffle(self.colorSpace)
        circleColorList = self.colorSpace[:self.numOfAgent]

        pause = True
        initialTime = time.get_ticks()
        fpsClock = pg.time.Clock()
        pg.mouse.set_visible(False)
        self.drawFixationPoint()
        for t in range(self.displayFrames):
            state = trajetoryData[t]
            fpsClock.tick(self.fps)

            screen = self.drawState(state, circleColorList)
            # screen = self.drawState(state, condition, circleColorList)
            # screen = self.drawStateWithRope(state, condition, self.colorSpace)

        self.drawImage(self.reportInstrucImage)
        pg.quit()
        self.openReportTxt()
        # self.drawImage(self.reportInstrucImage)
        return results
class ChaseTrialMujocoFps():
    def __init__(self, conditionsWithId,reactionWindowStart, drawState, drawImage, stimulus, checkHumanResponse, colorSpace, numOfAgent, drawFixationPoint, drawText, drawImageClick, clickWolfImage, clickSheepImage):
        self.conditionsWithId = conditionsWithId
        self.reactionWindowStart = reactionWindowStart
        self.stimulus = stimulus
        self.drawState = drawState
        self.drawImage = drawImage
        self.checkHumanResponse = checkHumanResponse
        self.colorSpace = colorSpace
        self.numOfAgent = numOfAgent
        self.drawFixationPoint = drawFixationPoint
        self.drawText = drawText
        self.drawImageClick = drawImageClick
        self.clickWolfImage = clickWolfImage
        self.clickSheepImage = clickSheepImage


    def __call__(self, condition):
        results = co.OrderedDict()
        results["trail"] = ''
        conditionId = condition['conditonId']
        conditionPara = self.conditionsWithId[conditionId][1]
        results['conditionId'] = conditionId
        self.fps = conditionPara['fps']
        self.displayFrames = conditionPara['displayTime'] * self.fps
        
        # randomSeed = np.mod(conditionId, 8)
        # results['rotationAngle'] = np.mod(randomSeed, 4)     
        # results['mirrror'] = np.mod(randomSeed//4,2) 
        # print(conditionPara)

        for key, values in conditionPara.items():
            results[key] = values
        # print(conditionPara,conditionId)
        # results['conditionId'] = conditionId
        # results['damping'] = conditionPara['damping']
        # results['frictionloss'] = conditionPara['frictionloss']
        # results['masterForce'] = conditionPara['masterForce']
        # results['offset'] = conditionPara['offset']
        # results['hideId(1:hidesheep;3,4:hideOneDistractor'] = conditionPara['hideId']
        results['trajetoryIndex'] = condition['trajetoryIndex']
        agentIdList = list(range(5))
        del agentIdList[conditionPara['hideId']]
        
        results['response'] = ''
        results['reactionTime'] = ''
        results['chosenWolfIndex'] = ''
        results['chosenSheepIndex'] = ''

        trajetoryData = self.stimulus[int(conditionId)][int(condition['trajetoryIndex'])]
        # print(len(trajetoryData))
        random.shuffle(self.colorSpace)
        circleColorList = self.colorSpace[:self.numOfAgent]

        pause = True
        initialTime = time.get_ticks()
        fpsClock = pg.time.Clock()
        while pause:
            pg.mouse.set_visible(False)
            self.drawFixationPoint()
            tialTime = time.get_ticks()
            Time = time.get_ticks()
            for t in range(self.displayFrames):
                fpsClock.tick(self.fps)
                state = trajetoryData[t]
                screen = self.drawState(state, circleColorList)
                # screen = self.drawState(state, condition, circleColorList)
                # screen = self.drawStateWithRope(state, condition, self.colorSpace)
                if t > self.reactionWindowStart:
                    results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)
                

                if not pause:
                    break

                if t == self.displayFrames - 1:
                    print(self.fps,self.displayFrames)
                    print(t,time.get_ticks()-tialTime)
                    pause = False
                    reactionTime = time.get_ticks() - initialTime
                    results['response'] = 1
                    results['reactionTime'] = str(reactionTime)
                    # self.drawText('Please Response Now!', (screen.get_width() / 4, screen.get_height() / 1.2))
                    # while pause:
                        # results, pause = self.checkHumanResponse(initialTime, results, pause, circleColorList)
                # print(time.get_ticks()-Time)
                Time = time.get_ticks()
            # if results['response'] == 1:
            pg.mouse.set_visible(True)
            chosenWolfIndex = self.drawImageClick(self.clickWolfImage, "Wolf", circleColorList)
            chosenSheepIndex = self.drawImageClick(self.clickSheepImage, 'Sheep', circleColorList)
            results['chosenWolfIndex'] = agentIdList[chosenWolfIndex]
            results['chosenSheepIndex'] = agentIdList[chosenSheepIndex]
            pg.time.wait(500)

        return results

if __name__ == "__main__":
    resultsPath = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'results')
    txtPath = (os.path.join(resultsPath, 'k' + '.txt'))
    openReportTxt = OpenReportTxt(txtPath)
    openReportTxt()
