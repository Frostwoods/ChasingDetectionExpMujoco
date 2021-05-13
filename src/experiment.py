import numpy as np


class Experiment():
    def __init__(self, trial, writer, experimentValues, darwImage, restImage, darwBackground, hasRest=True):
        self.trial = trial
        self.writer = writer
        self.experimentValues = experimentValues
        self.darwImage = darwImage
        self.restImage = restImage
        self.darwBackground = darwBackground
        self.hasRest = hasRest

    def __call__(self, designValues, restDuration):
        for trialIndex in range(len(designValues)):
            condition = designValues[trialIndex]
            results = self.trial(condition)

            results["trail"] = trialIndex + 1
            response = self.experimentValues.copy()
            response.update(results)
            self.writer(response, trialIndex)
            if np.mod(trialIndex + 1, restDuration) == 0 & self.hasRest:
                self.darwBackground()
                self.darwImage(self.restImage)
        return results


class SelectExperiment():
    def __init__(self, trial, writer,saver, experimentValues, darwImage, restImage, darwBackground, hasRest=True):
        self.trial = trial
        self.writer = writer
        self.saver = saver
        self.experimentValues = experimentValues
        self.darwImage = darwImage
        self.restImage = restImage
        self.darwBackground = darwBackground
        self.hasRest = hasRest

    def __call__(self, totalConditions,targetQuantityList,candicateNum,restoreNum): 
        for condition,targetQuantity in zip(totalConditions[restoreNum:],targetQuantityList[restoreNum:]):
            # print(condition)
            # print(totalConditions)
            slectTrajNum = 0
            trajIndex = 0
            slectTrajList = []
            while slectTrajNum < targetQuantity and trajIndex < candicateNum:
                results,traj = self.trial(condition,trajIndex)

                results["trail"] = trajIndex + 1
                response = self.experimentValues.copy()
                response.update(results)
                self.writer(response, trajIndex)
                trajIndex = trajIndex + 1
                if results['response'] == 1:
                    slectTrajList.append(traj)
                    slectTrajNum = slectTrajNum + 1

                # if np.mod(trialIndex + 1, restDuration) == 0 & self.hasRest:
                #     self.darwBackground()
                #     self.darwImage(self.restImage)
            self.saver(slectTrajList,condition,slectTrajNum)

        return results
