# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:31:06 2021

@author: adria
"""


import os
import numpy as np
import json
import copy
from scipy.integrate import odeint
from modelEquations import drnDamageEquationsSystem
import utils
import pandas as pd
from arm import Arm
import matplotlib.pyplot as plt
from numpy.random import RandomState


class BrainDiffentialEquationModel:
    
    def __init__(self, modelType = 'DRNdamage', configDir = 'bestConfigDir', 
        tStart = 0, tStop = 200, seedNum=1, seedNoise = 0.01, dt = 0.01,
        dmgDRND7 = True, dmgDRND7Start1 = 50, dmgDRND7Mag1 =20, dmgDRND7Start2 = 100,
        dmgDRND7Start3 = 150, dmgDRND7Mag2 =30,dmgDRND7Mag3= 40,
        saveData = True, plotResults = True, savePlot = True, closePlot = True,
        armPlot = True, startArmPlot = 0, tremorPlot = True, randomState =0):
        """
        (self, modelType = 'oscillatory', configDir = 'bestConfigDir',
        tStart = 0, tStop = 300, dt = 0.01, dmgSND8 = True, dmgSND8Start = 75,
        dmgSND8Mag = +70., dmgDAOsc = -80, trmtD7 = True, trmtD7Start1 = 150,
        trmtD7Mag1 = -10, trmtD7Start2 = 200, trmtD7Mag2 = -20, 
        trmtD7Start3 = 250, trmtD7Mag3 = -40, saveData = True, 
        plotResults = True, lines4Subplot = 2, closePlot = True, 
        armPlot = True, startArmPlot = 0, tremorPlot = True)
        
        Simulate brain areas activities
        
        Parameters
        ----------
        modelType : TYPE str, optional
            DESCRIPTION brain model type ('DRNdamage').
            The default is 'DRNdamage'.        
        configDir : TYPE str, optional
            DESCRITION set the configuration folder containing the model params
            The default is 'bestConfigDir'.
        tStart : TYPE int, optional
            DESCRIPTION. simulation time start (s).
            The default is 0.            
        tStop : TYPE int, optional
            DESCRIPTION. simultion timne stop (s). 
            The default is 300.   
        seedNum : TYPE int, optional    
            DESCRIPTION set simulation seed number.
            The default is 2
        seedNoise : TYPE, float
            DESCRIPTION set seed noise
            The default is 0.0\
        dt : TYPE float, optional
            DESCRIPTION. simultion delta time(s). 
            The default is 0.01.           
        dmgDRND7 : TYPE bool, optional
            DESCRIPTION. simulate SN damage (d8)). 
            The default is True.        
        dmgDRND7Start : TYPE int, optional
            DESCRIPTION. SN damage start (s).
            The default is 75.    
        dmgDRND7Mag1 : TYPE float, optional
            DESCRIPTION. SN damage magnitude (%). 
            The default is +20.  
        dmgDRND7Mag2 : TYPE float, optional
            DESCRIPTION. SN damage magnitude (%). 
            The default is +30. 
        dmgDRND7Mag3 : TYPE float, optional
            DESCRIPTION. SN damage magnitude (%). 
            The default is +40.
        saveData : TYPE bool, optional
            DESCRIPTION. save simulation results.
            The default is True.    
        plotResults : TYPE bool, optional
            DESCRIPTION. plot brain areas activities. 
            The default is True.
        lines4Subplot : TYPE int, optional
            DESCRIPTION. Set number of line for subplot.
            The default is 2.
        savePlot : TYPE bool, optional
            DESCRIPTION. Save simulation plot. 
            The default is True.
        closePlot : TYPE bool, optional
            DESCRIPTION. Automatically close simulation plot. 
            The default is True.
        armPlot : TYPE bool, optional
            DESCRIPTION. Plot arm simulation.
            The default is True.
        startArmPlot : TYPE int, optional
            DESCRIPTION. Arm plot start plotting simulation time.
            The default is 0.
        tremorPlot : TYPE bool, optional
            DESCRIPTION. Plot tremor amplitude results.
            The default is true.

        Returns
        -------
        None.

        """
        self.modelType = modelType
        self.configDir = configDir
        self.tStart = tStart
        self.tStop = tStop
        self.seedNum = seedNum
        self.seedNoise = seedNoise
        self.dt = dt
        self.dmgDRND7 = dmgDRND7
        self.dmgDRND7Start1 = dmgDRND7Start1
        self.dmgDRND7Start2 = dmgDRND7Start2
        self.dmgDRND7Start3 = dmgDRND7Start3
        self.dmgDRND7Mag1 = dmgDRND7Mag1
        self.dmgDRND7Mag2 = dmgDRND7Mag2
        self.dmgDRND7Mag3 = dmgDRND7Mag3
        self.saveData = saveData
        self.plotResults = plotResults
        self.savePlot = savePlot
        self.closePlot = closePlot
        self.armPlot = armPlot
        self.startArmPlot = startArmPlot
        self.tremorPlot = tremorPlot
        self.randomState = randomState
        
        
        
        self.healtyRange = 25 , 35
        self.damagedRange1 = 75, 85
        self.damagedRange2 = 125 , 135
        self.damagedRange3 = 175 , 185
        
        self.seedTremorData = np.zeros([self.seedNum,4])
        
        self.seedDA5HTData = np.zeros([self.seedNum,4])
        
        self.random = RandomState(randomState)
        
    
    def makeModelDirs(self):
        """
        Make brain model results directories

        Returns
        -------
        None.

        """
        
        self.cwd = os.getcwd()
        
        # set data directory
        self.resultsDir = os.path.join(
            self.cwd,
            'results'
            )
        if not os.path.exists(self.resultsDir):
            os.makedirs(self.resultsDir)
            
        # set model directory
        self.modelDir = os.path.join(
            self.resultsDir,
            self.modelType
            )
        if not os.path.exists(self.modelDir):
            os.makedirs(self.modelDir)
            
    def getTimeParams(self):
        """
        Get simulation time parameters

        Returns
        -------
        None.

        """
              
        self.nStep = int(self.tStop * (1 / self.dt) + 1)
        self.time = np.linspace(self.tStart, self.tStop, self.nStep)   
        
        
    def getBrainModelParams(self):
        """        
        Get brain parameters for simulation
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        # init brain model parameters dictionary 
        self.brainModelParams = {}
        
        # set configuration files directory
        self.configDir = os.path.join(self.cwd, self.configDir)
        
        
        # load brain areas steady state params
        self.steadyStateParamsPath = os.path.join(
            self.configDir,
            'brainSteadyStateParams.json'
            )
        with open(self.steadyStateParamsPath) as fp:
            self.steadyStateParams = json.load(fp)
        self.brainModelParams.update(self.steadyStateParams)
        
        #load model params
        if self.modelType == 'DRNDamage':
            self.oscillatoryParamsPath = os.path.join(
                self.configDir,
                'oscillatoryModelParams.json'
                )
            with open(self.oscillatoryParamsPath) as fp:
                self.oscillatoryParams = json.load(fp)
            self.brainModelParams.update(self.oscillatoryParams)
        elif self.modelType == 'reed':  
            self.reedParamsPath = os.path.join(
                self.configDir,
                'reedModelParams.json'
                )    
            with open(self.reedParamsPath) as fp:
                self.reedParams = json.load(fp)
            self.brainModelParams.update(self.reedParams)  
            
        self.getModelSteadyStateParamsList()
            
            
    
    
    def getModelSteadyStateParamsList(self):
        """
        Get brain elements list from brain parameters

        Returns
        -------
        None.

        """
        self.modelSteadyStateParamsNameList = []    
        for key in list(self.brainModelParams.keys()): 
            if key[-2:] == '_0':
                self.modelSteadyStateParamsNameList.append(key[:-2])
            
    
    def getSeedParams(self):
        """
        get seeds noised parameters
        """
        self.seedsParamsList = []
        
        self.modelParamValuesList = list(self.brainModelParams.values())
        self.modelParamNamesList = list(self.brainModelParams.keys())
        
        for i in range(self.seedNum):
            seedParams = {}           
            for i , param in enumerate(self.modelParamNamesList):
                seedParams[param] = \
                    self.modelParamValuesList[i] + self.random.normal(
                        0.0, 
                        self.modelParamValuesList[i] * self.seedNoise
                        )
            self.seedsParamsList.append(seedParams)
            
            
    def makeSeedDir(self, seed):
        
        self.seedDir = os.path.join(
            self.modelDir,
            'seed_{}'.format(seed)
            )
        if not os.path.exists(self.seedDir):
            os.makedirs(self.seedDir)
      
        # set brain data directory
        self.brainDir = os.path.join(
            self.seedDir,
            'brainActivities'
            )
        if not os.path.exists(self.brainDir):
            os.makedirs(self.brainDir)
            
        # set plot directory
        self.plotsDir = os.path.join(
            self.seedDir,
            'plots'
            )
        if not os.path.exists(self.plotsDir):
            os.makedirs(self.plotsDir)
            
        # set arm data directory
        self.armDir = os.path.join(
            self.seedDir,
            'armSimulation'
            )
        if not os.path.exists(self.armDir):
            os.makedirs(self.armDir)        
            
    def getBrainElementsInitialActivities(self):
        """
        Get brain areas initial activities

        Returns
        -------
        None.

        """
    
        self.y0 = [
            self.brainModelParams['IP_0'], 
            self.brainModelParams['DP_0'], 
            self.brainModelParams['Thal_0'], 
            self.brainModelParams['M1_0'], 
            self.brainModelParams['DRN_0'],
            self.brainModelParams['DA_0'], 
            self.brainModelParams['5-HT_0'], 
            self.brainModelParams['SNc_0'] 
            ]
    
    def integrateBrainElementsActivities(self):
        """
        Integrate brain areas differential equation system

        Returns
        -------
        None.

        """
         
        if self.modelType == 'DRNDamage':
            self.y = odeint(
                drnDamageEquationsSystem, # differential equation system
                self.y0, # initial state
                self.time, # integration step 
                args = (
                    self.brainModelParams['a1c'], self.brainModelParams['a1da'], 
                    self.brainModelParams['d1'], self.brainModelParams['aOsc'], 
                    self.brainModelParams['fOsc'], self.brainModelParams['daOsc'],
                    self.brainModelParams['a2c'], self.brainModelParams['a2da'],
                    self.brainModelParams['d2'], self.brainModelParams['a3'],
                    self.brainModelParams['a3md'], self.brainModelParams['a3mi'], 
                    self.brainModelParams['d3'], self.brainModelParams['a4th'], 
                    self.brainModelParams['d4'], self.brainModelParams['a5'], 
                    self.brainModelParams['a5cx'], self.brainModelParams['a5sn'],
                    self.brainModelParams['d5'], self.brainModelParams['G'],
                    self.brainModelParams['d6'], self.brainModelParams['a7'], 
                    self.brainModelParams['d7'], self.brainModelParams['a8'], 
                    self.brainModelParams['a8drn'], self.brainModelParams['d8'], 
                    self.dmgDRND7, self.dmgDRND7Start1, self.dmgDRND7Mag1, self.dmgDRND7Start2,
                    self.dmgDRND7Mag2, self.dmgDRND7Start3, self.dmgDRND7Mag3, self.brainModelParams['d5'], self.brainModelParams['d7'] 
                    
                    ),
                )           
            
    def getPlotVLines(self):
        """
        Get model plot labels

        Returns
        -------
        None.

        """
        self.vLinesLabelList = []
        self.vLinesXList = []
        
        # Get damaged model label
        if self.dmgDRND7:

            self.dmgLbl1 = '$\u03C4_{_{5HT}}$ = +20%'
            self.vLinesLabelList.append(self.dmgLbl1)
            self.vLinesXList.append(self.dmgDRND7Start1)
            self.dmgLbl2 = '$\u03C4_{_{5HT}}$ = +30%'
            self.vLinesLabelList.append(self.dmgLbl2)
            self.vLinesXList.append(self.dmgDRND7Start2)
            self.dmgLbl3 = '$\u03C4_{_{5HT}}$ = +40%'
            self.vLinesLabelList.append(self.dmgLbl3)
            self.vLinesXList.append(self.dmgDRND7Start3)

    def plotBrainElementsActivities(self):
        """
        Plot brain elements activities

        Returns
        -------
        None.

        """
        self.getPlotVLines()   
        
        plot1Name = self.modelSteadyStateParamsNameList[5] + '-' + \
                        self.modelSteadyStateParamsNameList[6] + ' ' + \
                        'activities'
        yLabel = 'Concentration [nM]'
        # funcList = [self.y[:,0],self.y[:,1]]    
        # labelList = [self.modelSteadyStateParamsNameList[0],
        #             self.modelSteadyStateParamsNameList[1]]  
        savePlotPath = os.path.join(
            self.plotsDir,
            plot1Name + '.png'
            )
        plotFigure = plt.figure(plot1Name,dpi=300)
        plotAxes = plotFigure.add_subplot(111)
        plotAxes.set_xlabel(
            'Time [sec]', 
            fontsize = 50, 
            fontweight = 'bold'
            )
        plotAxes.set_ylabel(
            yLabel, 
            fontsize = 50,
            fontweight = 'bold'
            ) 
        plotAxes.set_xlim(self.time[0],self.time[-1])
        plotAxes.set_ylim(0,5)
        plt.xticks(fontsize=50) 
        plt.yticks(fontsize=50) 
        vline1 = plotAxes.axvline(
            x = self.vLinesXList[0],
            ls='--',
            label=self.dmgLbl1,
            color= 'black',
            linewidth=10
            )
        vline2 = plotAxes.axvline(
            x = self.vLinesXList[1],
            ls='--',
            label=self.dmgLbl2,
            color= 'lightgrey',
            linewidth=10
            )
        vline3 = plotAxes.axvline(
            x = self.vLinesXList[2],
            ls='--',
            label=self.dmgLbl3,
            color= 'grey',
            linewidth=10
            )

        line1, = plotAxes.plot(self.time, self.y[:,5], label=self.modelSteadyStateParamsNameList[5], linewidth=10, color='pink')
        line2, = plotAxes.plot(self.time, self.y[:,6], label=self.modelSteadyStateParamsNameList[6], linewidth=10, color='brown')
        plotAxes.legend(
            (line1,
             line2,
             vline1,
              vline2,
              vline3,
              ),
            (self.modelSteadyStateParamsNameList[5],
             self.modelSteadyStateParamsNameList[6],
             self.dmgLbl1,
             self.dmgLbl2,
             self.dmgLbl3,
              ),
            loc = 'best', 
            fontsize = 25
            ) 
        plotFigure.savefig(savePlotPath)
        if self.closePlot:
            plt.close()
        





        plot1Name = self.modelSteadyStateParamsNameList[4] + '-' + \
                        self.modelSteadyStateParamsNameList[7] + ' ' + \
                        'activities'
        yLabel = 'Spyke frequency [Hz]'
        # funcList = [self.y[:,0],self.y[:,1]]    
        # labelList = [self.modelSteadyStateParamsNameList[0],
        #             self.modelSteadyStateParamsNameList[1]]  
        savePlotPath = os.path.join(
            self.plotsDir,
            plot1Name + '.png'
            )
        plotFigure = plt.figure(plot1Name,figsize=(32,16))
        plotAxes = plotFigure.add_subplot(111)
        plotAxes.set_xlabel(
            'Time [sec]', 
            fontsize = 50, 
            fontweight = 'bold'
            )
        plotAxes.set_ylabel(
            yLabel, 
            fontsize = 50,
            fontweight = 'bold'
            ) 
        plotAxes.set_xlim(self.time[0],self.time[-1])
        plotAxes.set_ylim(0,5)
        plt.xticks(fontsize=50) 
        plt.yticks(fontsize=50) 
        vline1 = plotAxes.axvline(
            x = self.vLinesXList[0],
            ls='--',
            label=self.dmgLbl1,
            color= 'black',
            linewidth=10
            )
        vline2 = plotAxes.axvline(
            x = self.vLinesXList[1],
            ls='--',
            label=self.dmgLbl2,
            color= 'lightgrey',
            linewidth=10
            )
        vline3 = plotAxes.axvline(
            x = self.vLinesXList[2],
            ls='--',
            label=self.dmgLbl3,
            color= 'grey',
            linewidth=10
            )

        line1, = plotAxes.plot(self.time, self.y[:,4], label=self.modelSteadyStateParamsNameList[4], linewidth=10, color='red')
        line2, = plotAxes.plot(self.time, self.y[:,7], label=self.modelSteadyStateParamsNameList[7], linewidth=10, color='blue')
        plotAxes.legend(
            (line1,
             line2,
             vline1,
              vline2,
              vline3,
              ),
            (self.modelSteadyStateParamsNameList[4],
             self.modelSteadyStateParamsNameList[7],
             self.dmgLbl1,
             self.dmgLbl2,
             self.dmgLbl3,
              ),
            loc = 'best', 
            fontsize = 25
            ) 
        plotFigure.savefig(savePlotPath)
        if self.closePlot:
            plt.close()
        
 
    
 
        plot1Name = self.modelSteadyStateParamsNameList[2] + '-' + \
                        self.modelSteadyStateParamsNameList[3] + ' ' + \
                        'activities'
        yLabel = 'Spyke frequency [Hz]'
        # funcList = [self.y[:,0],self.y[:,1]]    
        # labelList = [self.modelSteadyStateParamsNameList[0],
        #             self.modelSteadyStateParamsNameList[1]]  
        savePlotPath = os.path.join(
            self.plotsDir,
            plot1Name + '.png'
            )
        plotFigure = plt.figure(plot1Name,figsize=(32,16))
        plotAxes = plotFigure.add_subplot(111)
        plotAxes.set_xlabel(
            'Time [sec]', 
            fontsize = 50, 
            fontweight = 'bold'
            )
        plotAxes.set_ylabel(
            yLabel, 
            fontsize = 50,
            fontweight = 'bold'
            ) 
        plotAxes.set_xlim(self.time[0],self.time[-1])
        plotAxes.set_ylim(0,35)
        plt.xticks(fontsize=50) 
        plt.yticks(fontsize=50) 
        vline1 = plotAxes.axvline(
            x = self.vLinesXList[0],
            ls='--',
            label=self.dmgLbl1,
            color= 'black',
            linewidth=10
            )
        vline2 = plotAxes.axvline(
            x = self.vLinesXList[1],
            ls='--',
            label=self.dmgLbl2,
            color= 'lightgrey',
            linewidth=10
            )
        vline3 = plotAxes.axvline(
            x = self.vLinesXList[2],
            ls='--',
            label=self.dmgLbl3,
            color= 'grey',
            linewidth=10
            )

        line1, = plotAxes.plot(self.time, self.y[:,2], label=self.modelSteadyStateParamsNameList[2], linewidth=10, color='purple')
        line2, = plotAxes.plot(self.time, self.y[:,3], label=self.modelSteadyStateParamsNameList[3], linewidth=10, color='orange')
        plotAxes.legend(
            (line1,
             line2,
             vline1,
              vline2,
              vline3,
              ),
       (self.modelSteadyStateParamsNameList[2],
             self.modelSteadyStateParamsNameList[3],
             self.dmgLbl1,
             self.dmgLbl2,
             self.dmgLbl3,
              ),
            loc = 'best', 
            fontsize = 25
            ) 
        plotFigure.savefig(savePlotPath)
        if self.closePlot:
            plt.close()
        
        
        
        
        
        
        plot1Name = self.modelSteadyStateParamsNameList[0] + '-' + \
                        self.modelSteadyStateParamsNameList[1] + ' ' + \
                        'activities'
        yLabel = 'Spyke frequency [Hz]'
        # funcList = [self.y[:,0],self.y[:,1]]    
        # labelList = [self.modelSteadyStateParamsNameList[0],
        #             self.modelSteadyStateParamsNameList[1]]  
        savePlotPath = os.path.join(
            self.plotsDir,
            plot1Name + '.png'
            )
        plotFigure = plt.figure(plot1Name,figsize=(32,16))
        plotAxes = plotFigure.add_subplot(111)
        plotAxes.set_xlabel(
            'Time [sec]', 
            fontsize = 50, 
            fontweight = 'bold'
            )
        plotAxes.set_ylabel(
            yLabel, 
            fontsize = 50,
            fontweight = 'bold'
            ) 
        plotAxes.set_xlim(self.time[0],self.time[-1])
        plotAxes.set_ylim(0,5)
        plt.xticks(fontsize=50) 
        plt.yticks(fontsize=50) 
        vline1 = plotAxes.axvline(
            x = self.vLinesXList[0],
            ls='--',
            label=self.dmgLbl1,
            color= 'black',
            linewidth=10
            )
        vline2 = plotAxes.axvline(
            x = self.vLinesXList[1],
            ls='--',
            label=self.dmgLbl2,
            color= 'lightgrey',
            linewidth=10
            )
        vline3 = plotAxes.axvline(
            x = self.vLinesXList[2],
            ls='--',
            label=self.dmgLbl3,
            color= 'grey',
            linewidth=10
            )

        line1, = plotAxes.plot(self.time, self.y[:,0], label=self.modelSteadyStateParamsNameList[0], linewidth=10, color='lightgreen')
        line2, = plotAxes.plot(self.time, self.y[:,1], label=self.modelSteadyStateParamsNameList[1], linewidth=10, color='green')
        plotAxes.legend(
            (line1,
             line2,
             vline1,
             vline2,
             vline3,
              ),
          (self.modelSteadyStateParamsNameList[0],
             self.modelSteadyStateParamsNameList[1],
             self.dmgLbl1,
             self.dmgLbl2,
             self.dmgLbl3,
              ),
            loc = 'best', 
            fontsize = 25
            ) 
        plotFigure.savefig(savePlotPath)
        if self.closePlot:
            plt.close()
            
     
    def getTremorOscillationAmplitude(self, intervalRange):
        """
        compute tremor oscillation amplitude
        """   
        timeRange = [intervalRange[0]*100,intervalRange[1]*100]
        tremDist = self.euclDist[timeRange[0]:timeRange[1]] 
        dTremDist = np.ediff1d(tremDist, to_begin=np.array([0]))
        signDTremDist = np.sign(dTremDist)       
        diffSignDTremDist = np.ediff1d(signDTremDist)        
        tremDistMin = np.where(diffSignDTremDist > 1)[0]
        oscillationAmplitudeTrem = np.array([])
        # oscillationAmplitudeTremSum = 0
        for i in range(len(tremDistMin) -1):
            oscillationAmplitudeTremSum = 0
            oscillationIntervalTrem = \
                tremDist[tremDistMin[i]:tremDistMin[i+1]]
            oscillationAmplitudeTremSum += \
                np.sum(oscillationIntervalTrem)
            oscillationAmplitudeTrem = \
                np.hstack([oscillationAmplitudeTrem, 
                           oscillationAmplitudeTremSum])
        
        return np.mean(oscillationAmplitudeTrem) , np.std(oscillationAmplitudeTrem)
    
    def saveEvents(self):
        """
        Save simulation events data

        Returns
        -------
        None.

        """
        
        if self.modelType == 'DRNDamage':
            self.eventsData = {
                'd7 augmentation start 1': self.dmgDRND7Start1,
                'd7 augmentation mag 1': self.dmgDRND7Mag2,
                'd7 augmentation start 2': self.dmgDRND7Start2,
                'd7 augmentation mag 2': self.dmgDRND7Mag2,
                'd7 augmentation start 3': self.dmgDRND7Start3,
                'd7 augmentation mag 3': self.dmgDRND7Mag3,
                }

        self.eventsPath = os.path.join(
            self.brainDir, 
            'eventData.json'
            )   
        with open(self.eventsPath, 'w') as fp:
            json.dump(self.eventsData, fp)

    def saveBrainElementsActivities(self):
        """
        Save brain elements activities in csv format

        Returns
        -------
        None.

        """       
        self.saveArray = np.hstack([self.time.reshape(-1,1), self.y])
        self.dataColNames = ['Time'] + self.modelSteadyStateParamsNameList 
        self.activitiesDataframe = pd.DataFrame(
            data = self.saveArray,
            columns = self.dataColNames
            )

        self.dataframeSavePath = os.path.join(
            self.brainDir,
            'brainElementsActivities.csv'
            )    
        self.activitiesDataframe.to_csv(self.dataframeSavePath)
            
            
    def runSim(self):
        
        
        

        self.makeModelDirs()     
        self.getTimeParams()
        self.getBrainModelParams()
        self.getSeedParams()
        for seed, seedParams in enumerate(self.seedsParamsList):
            print('\nSeed {}'.format(seed))
            self.brainModelParams = copy.copy(seedParams)
            self.makeSeedDir(seed)
            self.getBrainElementsInitialActivities()
            self.integrateBrainElementsActivities()
            self.plotBrainElementsActivities()
            # self.getHealtySteadyStateMAPE()
            # if self.saveData:
            #     self.saveEvents()
            #     self.saveBrainElementsActivities()
            
            
                    
            

        
            self.CX = self.y[:,3].copy()
            self.CXHealthyMean = self.CX[
                self.healtyRange[0]*100:self.healtyRange[1]*100
                ].mean()
            self.CXHealthyRange = [0,self.CXHealthyMean*2]        
            self.efPosHistory = np.zeros([self.nStep, 2])
            
            self.DA = self.y[:,5].copy()
            self.DAHealthyMean = self.DA[
                self.healtyRange[0]*100:self.healtyRange[1]*100
                ].mean()
            self.DADamaged1Mean = self.DA[
                self.damagedRange1[0]*100:self.damagedRange1[1]*100
                ].mean()
            self.DADamaged2Mean = self.DA[
                self.damagedRange2[0]*100:self.damagedRange2[1]*100
                ].mean()
            self.DADamaged3Mean = self.DA[
                self.damagedRange3[0]*100:self.damagedRange3[1]*100
                ].mean()
            
            
            
            self.shoulderRange = np.deg2rad(np.array([- 60.0, 150.0]))
            self.elbowRange = np.deg2rad(np.array([   0.0, 180.0]))
            self.arm = Arm(self.shoulderRange, self.elbowRange)
            self.desiredAngles = np.ones(2) * 0.5
            self.Kp1 = 20.0
            self.Kd1 = 1.5
            self.Kp2 = 10.0
            self.Kd2 = 1.0
        
            if self.armPlot == True:
                self.simFig = plt.figure("ARM SIMULATION",dpi=300)
                self.simPlot = self.simFig.add_subplot(111)
                self.simPlot.set_xlim([-1.0,1.0])
                self.simPlot.set_ylim([0.0,1.0])   
                self.text1 = plt.figtext(
                    .02,
                    .72, 
                    "sec = %s" % (0.00),
                    style='italic',
                    bbox={'facecolor':'yellow'}
                    )    
                self.armPlot, = self.simPlot.plot(
                    [0, self.arm.xElbow, self.arm.xEndEf], 
                    [0, self.arm.yElbow, self.arm.yEndEf],
                    'k-', 
                    color = 'black', 
                    linewidth = 5
                    )   
                self.xDesAng = self.arm.L1*np.cos(self.desiredAngles[0]) + \
                            self.arm.L2*np.cos(self.desiredAngles[0]+self.desiredAngles[1])
                self.yDesAng = self.arm.L1*np.sin(self.desiredAngles[0]) + \
                            self.arm.L2*np.sin(self.desiredAngles[0]+self.desiredAngles[1])
                self.desEnd, = self.simPlot.plot(
                    [self.xDesAng],
                    [self.yDesAng], 
                    'o', 
                    color = 'green' ,
                    markersize= 10
                    )
        
            for t in range(self.nStep):
          
                self.desiredAngles[0] = utils.changeRange(
                    self.CX[t], 
                    self.CXHealthyRange[0],
                    self.CXHealthyRange[1],
                    self.shoulderRange[0],
                    self.shoulderRange[1])
                   
                self.desiredAngles[1] = utils.changeRange(
                    self.CX[t],
                    self.CXHealthyRange[0],
                    self.CXHealthyRange[1],
                    self.elbowRange[0],
                    self.elbowRange[1])
                   
                self.Torque = self.arm.PD_controller(
                    [self.desiredAngles[0],self.desiredAngles[1]],
                    self.Kp1 ,
                    self.Kp2,
                    self.Kd1, 
                    self.Kd2
                    ) # compute torques
                
                self.arm.SolveDirectDynamics(self.Torque[0], self.Torque[1]) # move the arm
                self.efPosHistory[t,:] = np.array([self.arm.xEndEf, self.arm.yEndEf])
                
                if self.armPlot and t > self.startArmPlot:
                              
                    self.text1.set_text("sec = {}".format(t/100.0))    
                    self.xDesAng = self.arm.L1*np.cos(self.desiredAngles[0]) +\
                                self.arm.L2*np.cos(self.desiredAngles[0]+self.desiredAngles[1])
                    self.yDesAng = self.arm.L1*np.sin(self.desiredAngles[0]) +\
                                self.arm.L2*np.sin(self.desiredAngles[0]+self.desiredAngles[1])
                    self.desEnd.set_data([self.xDesAng],[self.yDesAng])         
                    self.armPlot.set_data([0,self.arm.xElbow, self.arm.xEndEf],
                                      [0,self.arm.yElbow, self.arm.yEndEf])
                    plt.pause(self.dt)
   
            
            if self.saveData:
                efColNames = ['X','Y']
                self.efDataFrame = pd.DataFrame(
                    data = self.efPosHistory,
                    columns = efColNames
                    )
                
                efPath = os.path.join(
                    self.armDir,
                    'efPos.csv'
                    )                      
                self.efDataFrame.to_csv(efPath)
                
        
            self.efX = self.efPosHistory[:,0].copy()
            self.efY = self.efPosHistory[:,1].copy()                                           
            self.efdX = np.ediff1d(self.efX, to_begin=np.array([0]))
            self.efdY = np.ediff1d(self.efY, to_begin=np.array([0]))    
            self.euclDist = np.hypot(self.efdX,self.efdY)
         
            
        

            self.meanPhys, self.stdPhys = \
                self.getTremorOscillationAmplitude(self.healtyRange) 
            
                
            if self.modelType == 'DRNDamage':  
                self.meanDmg1, self.stdDmg1 = \
                self.getTremorOscillationAmplitude(self.damagedRange1)  
                self.meanDmg2, self.stdDmg2 = \
                self.getTremorOscillationAmplitude(self.damagedRange2)  
                self.meanDmg3, self.stdDmg3 = \
                self.getTremorOscillationAmplitude(self.damagedRange3)  
      
        
            
            
            self.seedTremorData[seed,0] = self.meanPhys
            if self.modelType == 'DRNDamage': 
                self.seedTremorData[seed,1] = self.meanDmg1
                self.seedTremorData[seed,2] = self.meanDmg2
                self.seedTremorData[seed,3] = self.meanDmg3
                
            
            self.seedDA5HTData[seed,0] = self.DAHealthyMean 
            self.seedDA5HTData[seed,1] = self.DADamaged1Mean
            self.seedDA5HTData[seed,2] = self.DADamaged2Mean
            self.seedDA5HTData[seed,3] = self.DADamaged3Mean
   
            
        self.dataColumns = ['healthy','damaged 1','damaged 2','damaged 3']
        self.tremorResults = pd.DataFrame(
            data=self.seedTremorData,
            columns=self.dataColumns
            )
        
        
        
        
    
        if self.tremorPlot:
            
            
            self.healthyLabel = 'HEALTHY'
            self.xTicksLabels = [
                self.healthyLabel,
                'DAMAGE 1',
                'DAMAGE 2',
                'DAMAGE 3',
                ]
            
            
            
            self.oscillationAmpFig = \
                plt.figure(
                    "oscillation amplitude",
                    dpi=300
                    )   
            self.oscillationAmpPlot = self.oscillationAmpFig.add_subplot(111)   
            self.oscillationAmpPlot.set_ylabel(
                "Oscillation amplitude (m)",
                fontsize = 10, 
                fontweight='bold'
                )   
            self.oscillationAmpPlot.set_xlabel(
                "DRN Lesion",
                fontsize = 10, 
                fontweight='bold'
                )   
            self.oscillationAmpPlot.set_ylim(0.0, 0.01)
            
            
            self.oscillationAmpPlot.set_xticks(np.arange(5))
            self.oscillationAmpPlot.set_xticklabels(self.xTicksLabels)
            self.oscillationAmpPlot.tick_params(
                axis='both', 
                which='major',
                labelsize=7
                )
            self.oscillationAmpPlot.tick_params(
                axis='both', 
                which='minor', 
                labelsize=7
                )
            
            
            self.oscillationAmpPlot.errorbar(
                [0],
                self.seedTremorData[:,0].mean(),
                yerr = self.seedTremorData[:,0].std(),
                marker= 's', 
                markersize='7' ,
                mec ='black',
                mfc='white',
                ecolor ='black', 
                fmt = '', 
                linestyle='solid'
                )
            
            self.oscillationAmpPlot.errorbar(
                [1],
                self.seedTremorData[:,1].mean(),
                yerr = self.seedTremorData[:,1].std(),
                marker= 's', 
                markersize='7' ,
                mec ='black',
                mfc='white',
                ecolor ='black', 
                fmt = '', 
                linestyle='solid'
                )
            
            self.oscillationAmpPlot.errorbar(
                [2],
                self.seedTremorData[:,2].mean(),
                yerr = self.seedTremorData[:,2].std(),
                marker= 's', 
                markersize='7' ,
                mec ='black',
                mfc='white',
                ecolor ='black', 
                fmt = '', 
                linestyle='solid'
                )
            
            self.oscillationAmpPlot.errorbar(
                [3],
                self.seedTremorData[:,3].mean(),
                yerr = self.seedTremorData[:,3].std(),
                marker= 's', 
                markersize='7' ,
                mec ='black',
                mfc='white',
                ecolor ='black', 
                fmt = '', 
                linestyle='solid'
                )
            
            self.oscillationAmpPlot.plot(
                list(range(4)),
                [self.seedTremorData[:,0].mean(),
                 self.seedTremorData[:,1].mean(),
                 self.seedTremorData[:,2].mean(),
                 self.seedTremorData[:,3].mean()],
                color = 'black')
            
            if self.saveData:
                
                self.dataColumns = ['healthy','damaged1','damaged2','damaged3']
                self.tremorResults = pd.DataFrame(
                    data=self.seedTremorData,
                    columns=self.dataColumns
                    ) 
                
                tremorPath = os.path.join(
                    self.modelDir,
                    'tremors.csv'
                    )
                self.tremorResults.to_csv(tremorPath)

            
            
            
            self.DAConcFig = \
                plt.figure(
                    "DA concentration reduction",
                    dpi=300
                    )   
            self.DAConcPlot = self.DAConcFig.add_subplot(111)   
            self.DAConcPlot.set_ylabel(
                "DA concentration (nM)",
                fontsize = 10, 
                fontweight='bold'
                )   
            self.DAConcPlot.set_xlabel(
                "DRN Lesion",
                fontsize = 10, 
                fontweight='bold'
                )   
            # self.DAConcPlot.set_ylim(0.0, 3.0)
            
            
            self.DAConcPlot.set_xticks(np.arange(5))
            self.DAConcPlot.set_xticklabels(self.xTicksLabels)
            self.DAConcPlot.tick_params(
                axis='both', 
                which='major',
                labelsize=7,
                # fontweight= 'bold'
                )
            self.DAConcPlot.tick_params(
                axis='both', 
                which='minor', 
                labelsize=7,
                # fontweight= 'bold'
                )
            
            
            self.DAConcPlot.errorbar(
                [0],
                self.seedDA5HTData[:,0].mean(),
                yerr = self.seedDA5HTData[:,0].std(),
                marker= 's', 
                markersize='7' ,
                mec ='black',
                mfc='white',
                ecolor ='black', 
                fmt = '', 
                linestyle='none'
                )
            
            self.DAConcPlot.errorbar(
                [1],
                self.seedDA5HTData[:,1].mean(),
                yerr = self.seedDA5HTData[:,1].std(),
                marker= 's', 
                markersize='7' ,
                mec ='black',
                mfc='white',
                ecolor ='black', 
                fmt = '', 
                linestyle='none'
                )
            
            self.DAConcPlot.errorbar(
                [2],
                self.seedDA5HTData[:,2].mean(),
                yerr = self.seedDA5HTData[:,2].std(),
                marker= 's', 
                markersize='7' ,
                mec ='black',
                mfc='white',
                ecolor ='black', 
                fmt = '', 
                linestyle='none'
                )
            
            self.DAConcPlot.errorbar(
                [3],
                self.seedDA5HTData[:,3].mean(),
                yerr = self.seedDA5HTData[:,3].std(),
                marker= 's', 
                markersize='7' ,
                mec ='black',
                mfc='white',
                ecolor ='black', 
                fmt = '', 
                linestyle='none'
                )
            
            self.DAConcPlot.plot(
                list(range(4)),
                [self.seedDA5HTData[:,0].mean(),
                 self.seedDA5HTData[:,1].mean(),
                 self.seedDA5HTData[:,2].mean(),
                 self.seedDA5HTData[:,3].mean()],
                color = 'black')
            
            if self.saveData:
                
                self.dataColumns = ['healthy','damaged1','damaged2','damaged3']
                self.DAResults = pd.DataFrame(
                    data=self.seedDA5HTData,
                    columns=self.dataColumns
                    ) 
                
                DAPath = os.path.join(
                    self.modelDir,
                    'DAconc.csv'
                    )
                self.DAResults.to_csv(DAPath)
        
        


if __name__ == "__main__":
    
    brain = BrainDiffentialEquationModel(
        modelType='DRNDamage',
        configDir = 'bestConfigDir2',
        seedNum = 20,
        dmgDRND7=True,
        closePlot = True,
        armPlot = False,
        tStop = 200, #250
        dmgDRND7Mag1 = +40.,
        dmgDRND7Mag2 = +60.,
        dmgDRND7Mag3 = +80.,
        tremorPlot = True
        )
    
    brain.runSim()
