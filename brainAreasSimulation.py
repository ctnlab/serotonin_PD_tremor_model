# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:34:02 2020

@author: adria
"""

import os
import json
import copy 
import numpy as np
from scipy.integrate import odeint
from modelEquations import oscillatoryEquationsSystem, reedEquationsSystem
import utils
import pandas as pd
from arm import Arm
import matplotlib.pyplot as plt
from numpy.random import RandomState


class BrainDiffentialEquationModel:
    
    def __init__(self, modelType = 'oscillatory', configDir = 'bestConfigDir',
        tStart = 0, tStop = 250, seedNum=1, seedNoise = 0.01,
        dt = 0.01, randomState=0, dmgSND8 = True, dmgSND8Start = 50, dmgSND8Mag = +80., 
        dmgDAOsc = -60, trmtD7 = True, trmtD7Start1 = 100, trmtD7Mag1 = -10,
        trmtD7Start2 = 150, trmtD7Mag2 = -20,trmtD7Start3 = 200, 
        trmtD7Mag3 = -30, saveData = True, plotResults = True, lines4Subplot = 2,
        savePlot = True, closePlot = True, armPlot = True, startArmPlot = 0,
        tremorPlot = True):
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
            DESCRIPTION brain model type ('reed' or 'oscillatory').
            The default is 'oscillatory'.        
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
        randomState : TYPE int, optional    
            DESCRIPTION set random state.
            The default is 0.
        dmgSND8 : TYPE bool, optional
            DESCRIPTION. simulate SN damage (d8)). 
            The default is True.        
        dmgSND8Start : TYPE int, optional
            DESCRIPTION. SN damage start (s).
            The default is 75.    
        dmgSND8Mag : TYPE float, optional
            DESCRIPTION. SN damage magnitude (%). 
            The default is +70.    
        dmgDAOsc : TYPE float, optional
            DESCRIPTION. damage DA oscillation reduction (%).
            The default is -80  (%).    
        trmtD7 : TYPE bool, optional
            DESCRIPTION. ssri treatment (D7).
            The default is True.   
        trmtD7Start1 : TYPE int, optional
            DESCRIPTION. ssri treatment time start 1 (s). 
            The default is 150.    
        trmtD7Mag1 : TYPE float, optional
            DESCRIPTION. ssri treatment magnitude 1 (%). 
            The default is -10.    
        trmtD7Start2 : TYPE int, optional
            DESCRIPTION. ssri treatment time start 2 (s).
            The default is 200.    
        trmtD7Mag2 : TYPE float, optional
            DESCRIPTION. ssri treatment magnitude 2 (%). 
            The default is -20.    
        trmtD7Start3 : TYPE int, optional
            DESCRIPTION. ssri treatment time start 3 (s). 
            The default is 250.    
        trmtD7Mag3 : TYPE float, optional
            DESCRIPTION. ssri treatment magnitude 3 (%). 
            The default is -40.    
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
        self.dmgSND8 = dmgSND8
        self.dmgSND8Start = dmgSND8Start
        self.dmgSND8Mag = dmgSND8Mag
        self.dmgDAOsc = dmgDAOsc
        self.trmtD7 = trmtD7
        self.trmtD7Start1 = trmtD7Start1
        self.trmtD7Mag1 = trmtD7Mag1
        self.trmtD7Start2 = trmtD7Start2
        self.trmtD7Mag2 = trmtD7Mag2
        self.trmtD7Start3 = trmtD7Start3
        self.trmtD7Mag3 = trmtD7Mag3
        self.saveData = saveData
        self.plotResults = plotResults
        self.lines4Subplot = lines4Subplot
        self.savePlot = savePlot
        self.closePlot = closePlot
        self.armPlot = armPlot
        self.startArmPlot = startArmPlot
        self.tremorPlot = tremorPlot
        
        
        
        self.healtyRange = 40 , 50
        self.damagedRange = 100, 110
        self.trmt1Range = 130 , 140
        self.trmt2Range = 180 , 190
        self.trmt3Range = 230 , 240
        
        if self.modelType =='reed':
            self.seedTremorData = np.zeros([self.seedNum,2])
        elif self.modelType =='oscillatory':
            self.seedTremorData = np.zeros([self.seedNum,5])
            
        self.random = RandomState(randomState)
            

        
        
        
                               
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
        if self.modelType == 'oscillatory':
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
        Get Seeds parameters

        Returns
        -------
        None.

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
                
               
    def getTimeParams(self):
        """
        Get simulation time parameters

        Returns
        -------
        None.

        """
              
        self.nStep = int(self.tStop * (1 / self.dt) + 1)
        self.time = np.linspace(self.tStart, self.tStop, self.nStep)    

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
         
        if self.modelType == 'oscillatory':
            self.y = odeint(
                oscillatoryEquationsSystem, # differential equation system
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
                    self.dmgSND8, self.dmgSND8Start, self.dmgSND8Mag, 
                    self.dmgDAOsc, self.trmtD7, self.trmtD7Start1, 
                    self.trmtD7Mag1, self.trmtD7Start2, self.trmtD7Mag2,
                    self.trmtD7Start3, self.trmtD7Mag3, self.brainModelParams['d7']
                    ),
                )    
        elif self.modelType == 'reed':
            self.y = odeint(
                reedEquationsSystem, # differential equation system
                self.y0, # initial state
                self.time, # integration step 
                args = (
                    self.brainModelParams['a1c'], self.brainModelParams['a1da'], 
                    self.brainModelParams['d1'], self.brainModelParams['a2c'], 
                    self.brainModelParams['a2da'], self.brainModelParams['d2'], 
                    self.brainModelParams['a3'], self.brainModelParams['a3md'], 
                    self.brainModelParams['a3mi'], self.brainModelParams['d3'],
                    self.brainModelParams['a4th'], self.brainModelParams['d4'], 
                    self.brainModelParams['a5'], self.brainModelParams['a5cx'],
                    self.brainModelParams['a5sn'], self.brainModelParams['d5'], 
                    self.brainModelParams['G'], self.brainModelParams['d6'], 
                    self.brainModelParams['a7'], self.brainModelParams['d7'], 
                    self.brainModelParams['a8'], self.brainModelParams['a8drn'], 
                    self.brainModelParams['d8'], self.dmgSND8, self.dmgSND8Start, 
                    self.dmgSND8Mag, 
                    ),
                )
            

        else:
            print('This model does not exists')
        

        
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
        if self.dmgSND8:
            self.dmgLbl ='$\u03C4_{_{SNc}}$  = +25%\n$\u03b1_{IP}$   =  -10%'#.format(self.dmgSND8Mag)
            self.vLinesLabelList.append(self.dmgLbl)
            self.vLinesXList.append(self.dmgSND8Start)
        # Get treated model labels
        if self.trmtD7:
            self.trmtLbl1 = '$\u03C4_{_{5HT}}$ = -20%'
            self.vLinesLabelList.append(self.trmtLbl1)
            self.vLinesXList.append(self.trmtD7Start1)
            self.trmtLbl2 = '$\u03C4_{_{5HT}}$ = -30%'
            self.vLinesLabelList.append(self.trmtLbl2)
            self.vLinesXList.append(self.trmtD7Start2)
            self.trmtLbl3 = '$\u03C4_{_{5HT}}$ = -40%'
            self.vLinesLabelList.append(self.trmtLbl3)
            self.vLinesXList.append(self.trmtD7Start3)


    def multiplotBrainElementsActivities(self):
        """
        """
        self.getPlotVLines()
        multiplotName = 'activities multiplot trmt'
        plotFigure = plt.figure(multiplotName,dpi=300)  
       

        
        
        plotAxes1 = plotFigure.add_subplot(221)
        yLabel = 'Concentration [nM]'
        plotAxes1.yaxis.set_label_coords(-0.14,0.5)
        savePlotPath = os.path.join(
            self.plotsDir,
            multiplotName + '.png'
            )
        plotAxes1.set_xlabel(
            'Time [sec]', 
            fontsize = 6, 
            fontweight = 'bold'
            )
        plotAxes1.set_ylabel(
            yLabel, 
            fontsize = 6,
            fontweight = 'bold'
            )
        plotAxes1.set_xlim(self.time[0],self.time[-1])
        plotAxes1.set_ylim(0,5)
        plt.xticks(fontsize=7) 
        plt.yticks(fontsize=7)
        text1 = plt.figtext(0.04, 0.47, 'A)', fontsize=14, fontweight='bold')
        line1, = plotAxes1.plot(
            self.time, 
            self.y[:,5], 
            label=self.modelSteadyStateParamsNameList[5],
            linewidth=4, 
            color='pink'
            )
        line2, = plotAxes1.plot(
            self.time, 
            self.y[:,6], 
            label=self.modelSteadyStateParamsNameList[6], 
            linewidth=4,
            color='brown'
            )
        
        vline1 = plotAxes1.axvline(
            x = self.vLinesXList[0],
            ls='--',
            label=self.dmgLbl,
            color= 'black',
            linewidth=2
            )
        vline2 = plotAxes1.axvline(
            x = self.vLinesXList[1],
            ls='--',
            label=self.trmtLbl1,
            color= 'grey',
            linewidth=2
            )
        vline3 = plotAxes1.axvline(
            x = self.vLinesXList[2],
            ls='--',
            label=self.trmtLbl2,
            color= 'darkgrey',
            linewidth=2
            )
        vline4 = plotAxes1.axvline(
            x = self.vLinesXList[3],
            ls='--',
            label=self.trmtLbl2,
            color= 'lightgrey',
            linewidth=2
            )

        plotAxes1.legend(
            (line1,line2,),
            (self.modelSteadyStateParamsNameList[5],
             self.modelSteadyStateParamsNameList[6],),
            loc='best',
            fontsize = 6
            )         
        
        
        plotAxes2 = plotFigure.add_subplot(222)
        yLabel = 'Spyke frequency [Hz]'
        plotAxes2.set_xlabel(
            'Time [sec]', 
            fontsize = 6, 
            fontweight = 'bold'
            )
        plotAxes2.set_ylabel(
            yLabel, 
            fontsize = 6,
            fontweight = 'bold'
            )
        plotAxes2.set_xlim(self.time[0],self.time[-1])
        plotAxes2.set_ylim(0,5)
        plt.xticks(fontsize=7) 
        plt.yticks(fontsize=7)
        text2 = plt.figtext(0.48, 0.47, 'B)', fontsize=14, fontweight='bold')
        line3, = plotAxes2.plot(
            self.time, 
            self.y[:,4], 
            label=self.modelSteadyStateParamsNameList[4],
            linewidth=4, 
            color='red'
            )
        line4, = plotAxes2.plot(
            self.time, 
            self.y[:,7], 
            label=self.modelSteadyStateParamsNameList[7], 
            linewidth=4,
            color='blue'
            )
        
        vline1 = plotAxes2.axvline(
            x = self.vLinesXList[0],
            ls='--',
            label=self.dmgLbl,
            color= 'black',
            linewidth=2
            )
        vline2 = plotAxes2.axvline(
            x = self.vLinesXList[1],
            ls='--',
            label=self.trmtLbl1,
            color= 'grey',
            linewidth=2
            )
        vline3 = plotAxes2.axvline(
            x = self.vLinesXList[2],
            ls='--',
            label=self.trmtLbl2,
            color= 'darkgrey',
            linewidth=2
            )
        vline4 = plotAxes2.axvline(
            x = self.vLinesXList[3],
            ls='--',
            label=self.trmtLbl2,
            color= 'lightgrey',
            linewidth=2
            )
        
        plotAxes2.legend(
            (line3,line4,),
            (self.modelSteadyStateParamsNameList[4],
             self.modelSteadyStateParamsNameList[7],),
            loc='best',
            fontsize = 6
            )         
        
        
        plotAxes3 = plotFigure.add_subplot(223)
        yLabel = 'Spyke frequency [Hz]'
        plotAxes3.set_xlabel(
            'Time [sec]', 
            fontsize = 6, 
            fontweight = 'bold'
            )
        plotAxes3.set_ylabel(
            yLabel, 
            fontsize = 6,
            fontweight = 'bold'
            )
        plotAxes3.set_xlim(self.time[0],self.time[-1])
        plotAxes3.set_ylim(0,35)
        plt.xticks(fontsize=7) 
        plt.yticks(fontsize=7)
        text3 = plt.figtext(0.04, 0.02, 'C)', fontsize=14, fontweight='bold')
        line5, = plotAxes3.plot(
            self.time, 
            self.y[:,2], 
            label=self.modelSteadyStateParamsNameList[2],
            linewidth=4, 
            color='purple'
            )
        line6, = plotAxes3.plot(
            self.time, 
            self.y[:,3], 
            label=self.modelSteadyStateParamsNameList[3], 
            linewidth=4,
            color='orange'
            )
        
        vline1 = plotAxes3.axvline(
            x = self.vLinesXList[0],
            ls='--',
            label=self.dmgLbl,
            color= 'black',
            linewidth=2
            )
        vline2 = plotAxes3.axvline(
            x = self.vLinesXList[1],
            ls='--',
            label=self.trmtLbl1,
            color= 'grey',
            linewidth=2
            )
        vline3 = plotAxes3.axvline(
            x = self.vLinesXList[2],
            ls='--',
            label=self.trmtLbl2,
            color= 'darkgrey',
            linewidth=2
            )
        vline4 = plotAxes3.axvline(
            x = self.vLinesXList[3],
            ls='--',
            label=self.trmtLbl2,
            color= 'lightgrey',
            linewidth=2
            )
        plotAxes3.legend(
            (line5,line6,),
            (self.modelSteadyStateParamsNameList[2],
             self.modelSteadyStateParamsNameList[3],),
            loc='best',
            fontsize = 6
            ) 
        
        
        
        plotAxes4 = plotFigure.add_subplot(224)
        yLabel = 'Spyke frequency [Hz]'
        plotAxes4.set_xlabel(
            'Time [sec]', 
            fontsize = 6, 
            fontweight = 'bold'
            )
        plotAxes4.set_ylabel(
            yLabel, 
            fontsize = 6,
            fontweight = 'bold'
            )
        plotAxes4.set_xlim(self.time[0],self.time[-1])
        plotAxes4.set_ylim(0,5)
        plt.xticks(fontsize=5) 
        plt.yticks(fontsize=5)
        text4 = plt.figtext(0.48, 0.02, 'D)', fontsize=14, fontweight='bold')
        line7, = plotAxes4.plot(
            self.time, 
            self.y[:,0], 
            label=self.modelSteadyStateParamsNameList[0],
            linewidth=4, 
            color='lightgreen'
            )
        line8, = plotAxes4.plot(
            self.time, 
            self.y[:,1], 
            label=self.modelSteadyStateParamsNameList[1], 
            linewidth=4,
            color='green'
            )
        
        vline1 = plotAxes4.axvline(
            x = self.vLinesXList[0],
            ls='--',
            label=self.dmgLbl,
            color= 'black',
            linewidth=2
            )
        vline2 = plotAxes4.axvline(
            x = self.vLinesXList[1],
            ls='--',
            label=self.trmtLbl1,
            color= 'grey',
            linewidth=2
            )
        vline3 = plotAxes4.axvline(
            x = self.vLinesXList[2],
            ls='--',
            label=self.trmtLbl2,
            color= 'darkgrey',
            linewidth=2
            )
        vline4 = plotAxes4.axvline(
            x = self.vLinesXList[3],
            ls='--',
            label=self.trmtLbl2,
            color= 'lightgrey',
            linewidth=2
            )
        
        plotAxes4.legend(
            (line7,line8,),
            (self.modelSteadyStateParamsNameList[0],
             self.modelSteadyStateParamsNameList[1],),
            loc='best',
            fontsize = 6
            ) 
        
        plt.subplots_adjust(
            left=None, 
            bottom=None, 
            right=None, 
            top=None, 
            wspace=0.4,
            hspace=0.4
            )
        
        
        
        
        
        
        

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
                        'activities_no_trmt'
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
            fontsize = 15, 
            fontweight = 'bold'
            )
        plotAxes.set_ylabel(
            yLabel, 
            fontsize = 15,
            fontweight = 'bold'
            ) 
        plotAxes.set_xlim(self.time[0],self.time[-1])
        plotAxes.set_ylim(0,5)
        plt.xticks(fontsize=13) 
        plt.yticks(fontsize=13) 

        line1, = plotAxes.plot(self.time, self.y[:,5], label=self.modelSteadyStateParamsNameList[5], linewidth=5, color='pink')
        line2, = plotAxes.plot(self.time, self.y[:,6], label=self.modelSteadyStateParamsNameList[6], linewidth=5, color='brown')
        
        vline1 = plotAxes.axvline(
            x = self.vLinesXList[0],
            ls='--',
            label=self.dmgLbl,
            color= 'black',
            linewidth=2
            )
        # vline2 = plotAxes.axvline(
        #     x = self.vLinesXList[1],
        #     ls='--',
        #     label=self.trmtLbl1,
        #     color= 'grey',
        #     linewidth=2
        #     )
        # vline3 = plotAxes.axvline(
        #     x = self.vLinesXList[2],
        #     ls='--',
        #     label=self.trmtLbl2,
        #     color= 'darkgrey',
        #     linewidth=2
        #     )
        # vline4 = plotAxes.axvline(
        #     x = self.vLinesXList[3],
        #     ls='--',
        #     label=self.trmtLbl2,
        #     color= 'lightgrey',
        #     linewidth=2
        #     )
        

      
        plotAxes.legend(
            (line1,
             line2,
             # vline1,
             # vline2,
             # vline3,
             # vline4
              ),
            (self.modelSteadyStateParamsNameList[5],
             self.modelSteadyStateParamsNameList[6],
              # self.dmgLbl,
              #   self.trmtLbl1,
              #   self.trmtLbl2,
              #   self.trmtLbl3,
              ),
            loc='best',
            # bbox_to_anchor=(+0.5, 1.15),
            # ncol=6,
            fontsize = 13
            ) 
        plotFigure.savefig(savePlotPath)
        if self.closePlot:
            plt.close()
        





        plot1Name = self.modelSteadyStateParamsNameList[4] + '-' + \
                        self.modelSteadyStateParamsNameList[7] + ' ' + \
                        'activities_no_trmt'
        yLabel = 'Spyke frequency [Hz]'
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
            fontsize = 15, 
            fontweight = 'bold'
            )
        plotAxes.set_ylabel(
            yLabel, 
            fontsize = 15,
            fontweight = 'bold'
            ) 
        plotAxes.set_xlim(self.time[0],self.time[-1])
        plotAxes.set_ylim(0,5)
        plt.xticks(fontsize=13) 
        plt.yticks(fontsize=13) 
        


        
        line1, = plotAxes.plot(self.time, self.y[:,4], label=self.modelSteadyStateParamsNameList[4], linewidth=5, color='red')
        line2, = plotAxes.plot(self.time, self.y[:,7], label=self.modelSteadyStateParamsNameList[7], linewidth=5, color='blue')
        
        vline1 = plotAxes.axvline(
            x = self.vLinesXList[0],
            ls='--',
            label=self.dmgLbl,
            color= 'black',
            linewidth=2
            )
        # vline2 = plotAxes.axvline(
        #     x = self.vLinesXList[1],
        #     ls='--',
        #     label=self.trmtLbl1,
        #     color= 'grey',
        #     linewidth=2
        #     )
        # vline3 = plotAxes.axvline(
        #     x = self.vLinesXList[2],
        #     ls='--',
        #     label=self.trmtLbl2,
        #     color= 'darkgrey',
        #     linewidth=2
        #     )
        # vline4 = plotAxes.axvline(
        #     x = self.vLinesXList[3],
        #     ls='--',
        #     label=self.trmtLbl2,
        #     color= 'lightgrey',
        #     linewidth=2
        #     )

        plotAxes.legend(
            (line1,
             line2,
             # vline1,
             # vline2,
             # vline3,
             # vline4
              ),
            (self.modelSteadyStateParamsNameList[4],
             self.modelSteadyStateParamsNameList[7],
              # self.dmgLbl,
              #   self.trmtLbl1,
              #   self.trmtLbl2,
              #   self.trmtLbl3,
              ),
            loc='best',
            # bbox_to_anchor=(+0.5, 1.15),
            # ncol=6,
            fontsize = 13
            ) 
        plotFigure.savefig(savePlotPath)
        if self.closePlot:
            plt.close()
        
 
        
 
    
 
        plot1Name = self.modelSteadyStateParamsNameList[2] + '-' + \
                        self.modelSteadyStateParamsNameList[3] + ' ' + \
                        'activities_no_trmt'
        yLabel = 'Spyke frequency [Hz]'
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
            fontsize = 15, 
            fontweight = 'bold'
            )
        plotAxes.set_ylabel(
            yLabel, 
            fontsize = 15,
            fontweight = 'bold'
            ) 
        plotAxes.set_xlim(self.time[0],self.time[-1])
        plotAxes.set_ylim(0,35)
        plt.xticks(fontsize=13) 
        plt.yticks(fontsize=13) 


        line1, = plotAxes.plot(self.time, self.y[:,2], label=self.modelSteadyStateParamsNameList[2], linewidth=5, color='purple')
        line2, = plotAxes.plot(self.time, self.y[:,3], label=self.modelSteadyStateParamsNameList[3], linewidth=5, color='orange')
        
        
        vline1 = plotAxes.axvline(
            x = self.vLinesXList[0],
            ls='--',
            label=self.dmgLbl,
            color= 'black',
            linewidth=2
            )
        # vline2 = plotAxes.axvline(
        #     x = self.vLinesXList[1],
        #     ls='--',
        #     label=self.trmtLbl1,
        #     color= 'grey',
        #     linewidth=2
        #     )
        # vline3 = plotAxes.axvline(
        #     x = self.vLinesXList[2],
        #     ls='--',
        #     label=self.trmtLbl2,
        #     color= 'darkgrey',
        #     linewidth=2
        #     )
        # vline4 = plotAxes.axvline(
        #     x = self.vLinesXList[3],
        #     ls='--',
        #     label=self.trmtLbl2,
        #     color= 'lightgrey',
        #     linewidth=2
        #     )
   
        plotAxes.legend(
            (line1,
             line2,
             # vline1,
             # vline2,
             # vline3,
             # vline4
               ),
            (self.modelSteadyStateParamsNameList[2],
             self.modelSteadyStateParamsNameList[3],
              # self.dmgLbl,
              #   self.trmtLbl1,
              #   self.trmtLbl2,
              #   self.trmtLbl3,
              ),
            loc='best',
            # bbox_to_anchor=(+0.5, 1.15),
            # ncol=6,
            fontsize = 13
            ) 
        plotFigure.savefig(savePlotPath)
        if self.closePlot:
            plt.close()
        
        
        
        
        
        
        plot1Name = self.modelSteadyStateParamsNameList[0] + '-' + \
                        self.modelSteadyStateParamsNameList[1] + ' ' + \
                        'activities_no_trmt'
        yLabel = 'Spyke frequency [Hz]'
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
            fontsize = 15, 
            fontweight = 'bold'
            )
        plotAxes.set_ylabel(
            yLabel, 
            fontsize = 15,
            fontweight = 'bold'
            ) 
        plotAxes.set_xlim(self.time[0],self.time[-1])
        plotAxes.set_ylim(0,5)
        plt.xticks(fontsize=13) 
        plt.yticks(fontsize=13) 


        line1, = plotAxes.plot(self.time, self.y[:,0], label=self.modelSteadyStateParamsNameList[0], linewidth=5, color='lightgreen')
        line2, = plotAxes.plot(self.time, self.y[:,1], label=self.modelSteadyStateParamsNameList[1], linewidth=5, color='green')

        vline1 = plotAxes.axvline(
            x = self.vLinesXList[0],
            ls='--',
            label=self.dmgLbl,
            color= 'black',
            linewidth=2
            )
        
        
        # vline2 = plotAxes.axvline(
        #     x = self.vLinesXList[1],
        #     ls='--',
        #     label=self.trmtLbl1,
        #     color= 'grey',
        #     linewidth=2
        #     )
        # vline3 = plotAxes.axvline(
        #     x = self.vLinesXList[2],
        #     ls='--',
        #     label=self.trmtLbl2,
        #     color= 'darkgrey',
        #     linewidth=2
        #     )
        # vline4 = plotAxes.axvline(
        #     x = self.vLinesXList[3],
        #     ls='--',
        #     label=self.trmtLbl2,
        #     color= 'lightgrey',
        #     linewidth=2
        #     )

        plotAxes.legend(
            (line1,
             line2,
             # vline1,
             # vline2,
             # vline3,
             # vline4
              ),
            (self.modelSteadyStateParamsNameList[0],
             self.modelSteadyStateParamsNameList[1],
              # self.dmgLbl,
              #   self.trmtLbl1,
              #   self.trmtLbl2,
              #   self.trmtLbl3,
              ),
            loc='best',
            # bbox_to_anchor=(+0.5, 1.15),
            # ncol=6,
            fontsize = 13
            ) 
        plotFigure.savefig(savePlotPath)
        if self.closePlot:
            plt.close()
        
        
        

        
        
        
        
        # for plot in range(0,len(self.modelSteadyStateParamsNameList),2):
            
        #     plotName = self.modelSteadyStateParamsNameList[plot] + '-' + \
        #                 self.modelSteadyStateParamsNameList[plot+1] + ' ' + \
        #                 'activities'
        #     if self.modelSteadyStateParamsNameList[plot] == '5-HT':
        #         yLabel = 'Concentration [nM]'
        #     else:
        #         yLabel = 'Spike frequency [Hz]' 
        #     funcList = [self.y[:,plot],self.y[:,plot+1]]    
        #     labelList = [self.modelSteadyStateParamsNameList[plot],
        #                   self.modelSteadyStateParamsNameList[plot+1]]                     
        #     # print(len(self.vLinesXList))          
        #     if self.savePlot:
        #         savePlotPath = os.path.join(
        #             self.plotsDir,
        #             plotName + '.png'
        #             )
        #         utils.plotData(
        #             figureName=plotName,
        #             xLabel='Time [sec]',
        #             yLabel= yLabel,
        #             x = self.time,
        #             Y = funcList,
        #             labelList=labelList,
        #             vLineLabelsList=self.vLinesLabelList,
        #             vLineXList=self.vLinesXList,
        #             savePlotPath = savePlotPath,
        #             closePlot=self.closePlot,
        #             lineWidth=5
        #             )
        #     else:
        #         utils.plotData(
        #             figureName=plotName,
        #             xLabel='Time [sec]',
        #             yLabel= yLabel,
        #             x = self.time,
        #             Y = funcList,
        #             labelList=labelList,
        #             vLineLabelsList=self.vLinesLabelList,
        #             vLineXList=self.vLinesXList,
        #             closePlot=self.closePlot
        #             )                

                

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
            
        # # create seed dirs    
        # for i in range(self.seedNum):
            
          

    def saveEvents(self):
        """
        Save simulation events data

        Returns
        -------
        None.

        """
        
        if self.modelType == 'oscillatory':
            self.eventsData = {
                'd8 reduction start': self.dmgSND8Start,
                'd8 reduction mag': self.dmgSND8Mag,
                'd7 augmentation start 1': self.trmtD7Start1,
                'd7 augmentation mag 1': self.trmtD7Mag1,
                'd7 augmentation start 2': self.trmtD7Start2,
                'd7 augmentation mag 2': self.trmtD7Mag2,
                'd7 augmentation start 3': self.trmtD7Start3,
                'd7 augmentation mag 3': self.trmtD7Mag3,
                }
        elif self.modelType == 'reed':
             self.eventsData = {
                'd8 reduction start': self.dmgSND8Start,
                'd8 reduction mag': self.dmgSND8Mag,
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
        

    def getHealtySteadyStateMAPE(self):
        """
        Compute mean absolute percentage error between brain experimental data
        and healthy model steady state 

        Returns
        -------
        None.

        """
        
        # get healthy model brain elements steady state
        self.healthySteadyState = self.y[
            self.healtyRange[0]*100:self.healtyRange[1]*100
            :].mean(axis=0)
        # compute difference between model and real brain values of steady state 
        self.steadyStateDiff = self.healthySteadyState - \
             np.array(list(self.steadyStateParams.values()))
        
        #compute the percentage difference
        self.percentDiff = (self.steadyStateDiff / self.healthySteadyState) * 100
        # compute the absolute percentage difference
        self.absPercentDiff = np.abs(self.percentDiff)
        # compute the mean and standard deviation 
        # of the absolute percentage difference
        self.mean = np.mean(self.absPercentDiff)
        self.std = np.std(self.absPercentDiff)
        # print('mean absolute percetage error : {}'.format(self.mean))
        # print('std absolute percetage error : {}'.format(self.std))
        
    def getDAConcentrationPercDiff(self):
        
        self.DAConcentration = self.y[:,5].copy()
        self.healthDAConc = self.DAConcentration[self.healtyRange[0]*100:self.healtyRange[1]*100].mean()
        self.dmgDAConc = self.DAConcentration[self.damagedRange[0]*100:self.damagedRange[1]*100].mean()
        self.healthDmgDiff = self.healthDAConc - self.dmgDAConc
        self.percentDiff = (self.healthDmgDiff / self.healthDAConc) * 100
        self.absPercentDiff = np.abs(self.percentDiff)
        print('damaged model DA difference : {}'.format(self.absPercentDiff))
        
        
        
        
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

        
    
    def runSim(self):
        """
        Run brain areas differential equation model 

        Returns
        -------
        None.

        """

        # Init differential equation brain model
       
    
        self.makeModelDirs()
        
        self.getTimeParams()
        self.getBrainModelParams()
        self.getSeedParams()
        
        for seed, seedParams in enumerate(self.seedsParamsList):
            print('\nSeed {}'.format(seed))
            self.brainModelParams = copy.copy(seedParams)
            
            
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
                
                
            
            
        
            self.getBrainElementsInitialActivities()
            self.integrateBrainElementsActivities()
            # self.plotBrainElementsActivities()
            self.multiplotBrainElementsActivities()
            self.getDAConcentrationPercDiff
            # self.getHealtySteadyStateMAPE()
            if self.saveData:
                self.saveEvents()
                self.saveBrainElementsActivities()
        
        

        
            self.CX = self.y[:,3].copy()
            self.CXHealthyMean = self.CX[
                self.healtyRange[0]*100:self.healtyRange[1]*100
                ].mean()
            self.CXHealthyRange = [0,self.CXHealthyMean*2]        
            self.efPosHistory = np.zeros([self.nStep, 2])
            
            
            
            self.shoulderRange = np.deg2rad(np.array([- 60.0, 150.0]))
            self.elbowRange = np.deg2rad(np.array([   0.0, 180.0]))
            self.arm = Arm(self.shoulderRange, self.elbowRange)
            self.desiredAngles = np.ones(2) * 0.5
            self.Kp1 = 20.0
            self.Kd1 = 1.5
            self.Kp2 = 10.0
            self.Kd2 = 1.0
        
            if self.armPlot == True:
                self.simFig = plt.figure("ARM SIMULATION")
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
            if self.modelType == 'oscillatory':  
                self.meanPark, self.stdPark = \
                self.getTremorOscillationAmplitude(self.damagedRange) 
                if self.trmtD7:                  
                    self.meanTrmt1, self.stdTrmt1 = \
                        self.getTremorOscillationAmplitude(self.trmt1Range)         
                    self.meanTrmt2, self.stdTrmt2 = \
                        self.getTremorOscillationAmplitude(self.trmt2Range)         
                    self.meanTrmt3, self.stdTrmt3 = \
                        self.getTremorOscillationAmplitude(self.trmt3Range) 
            
            
            
            self.seedTremorData[seed,0] = self.meanPhys
            if self.modelType == 'oscillatory': 
                self.seedTremorData[seed,1] = self.meanPark
                if self.trmtD7: 
                    self.seedTremorData[seed,2] = self.meanTrmt1
                    self.seedTremorData[seed,3] = self.meanTrmt2
                    self.seedTremorData[seed,4] = self.meanTrmt3
        
   
            
        self.dataColumns = ['healthy','damaged','trmt1','trmt2','trmt3']
        self.tremorResults = pd.DataFrame(
            data=self.seedTremorData,
            columns=self.dataColumns
            ) 
        
        
        if self.tremorPlot:
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
                "Conditions",
                fontsize = 10, 
                fontweight='bold'
                )    
            self.oscillationAmpPlot.set_ylim(0.0, 0.05)
            self.healthyLabel = 'HEALTHY'
            self.damagedLabel = 'SNc DAMAGE'
            self.trmtLbl1 = 'TRMT 1'
            self.trmtLbl2 = 'TRMT 2'
            self.trmtLbl3 = 'TRMT 3'
            
            # if self.modelType == 'reed':
            #     self.damagedLabel = 'd8 = + {}%'.format(self.dmgSND8Mag)
            # elif self.modelType == 'oscillatory':
            #     self.damagedLabel = 'd8 = + {}%'.format(self.dmgSND8Mag)
                
            
            self.xTicksLabels = [
                self.healthyLabel,
                self.damagedLabel,
                self.trmtLbl1,
                self.trmtLbl2,
                self.trmtLbl3
                ]
            
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
                linestyle='none'
                )
            
            if self.modelType == 'oscillatory':
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
                    linestyle='none'
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
                    linestyle='none'
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
                    linestyle='none'
                    )
                
                self.oscillationAmpPlot.errorbar(
                    [4],
                    self.seedTremorData[:,4].mean(),
                    yerr = self.seedTremorData[:,4].std(),
                    marker= 's', 
                    markersize='7' ,
                    mec ='black',
                    mfc='white',
                    ecolor ='black', 
                    fmt = '', 
                    linestyle='none'
                    )
                
                self.oscillationAmpPlot.plot(
                    list(range(5)),
                    [self.seedTremorData[:,0].mean(),
                     self.seedTremorData[:,1].mean(),
                     self.seedTremorData[:,2].mean(),
                     self.seedTremorData[:,3].mean(),
                     self.seedTremorData[:,4].mean()],
                    color = 'black')
        
  
        if self.saveData:
            tremorPath = os.path.join(
                self.modelDir,
                'tremors.csv'
                )
            self.tremorResults.to_csv(tremorPath)

        

        
        
        
        
if __name__ == "__main__":
    
    brain = BrainDiffentialEquationModel(
        modelType='oscillatory',
        configDir = 'bestConfigDir2',
        seedNum = 1,
        dmgSND8=True,
        trmtD7=True, #false
        closePlot = False,
        armPlot = False,
        tStop = 300, #250
        dmgSND8Mag = +70., #70
        dmgDAOsc = -70, #70
        dmgSND8Start = 75, #50
        trmtD7Start1 = 125,
        trmtD7Mag1 = -20,
        trmtD7Start2 = 175, 
        trmtD7Mag2 = -30,
        trmtD7Start3 = 225, 
        trmtD7Mag3 = -40,
        tremorPlot = True
        )
    
    brain.runSim()
        
               
        
                 