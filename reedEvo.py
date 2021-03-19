# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 21:11:40 2020

@author: adria
"""


import os
import json
import numpy as np
import copy
from modelEquations import reedEquationsSystem2, oscillatoryEquationsSystem2
from scipy.integrate import odeint
from arm import Arm
import utils
import matplotlib.pyplot as plt


# sim config
model = 'oscillatory'
tStart = 0
tStop = 100
dt = 0.01
dmgSND8 = True
dmgSND8Start = 50 
dmgSND8Mag = +70 
dmgSNDAoscMag = -70
PLOTTING = False
start_plotting = 0


# genetic algorithm config
chromoPopNum = 30
mutRate = 0.01
survNum = 1
genNum = 300




shoulderRange = np.deg2rad(np.array([- 60.0, 150.0]))
elbowRange    = np.deg2rad(np.array([   0.0, 180.0]))
arm = Arm(shoulderRange, elbowRange)
Kp1 = 20.0
Kd1 = 1.5
Kp2 = 10.0
Kd2 = 1.0

desiredAngles = np.ones(2) * 0.5
desiredAngles[0] = utils.changeRange(
    desiredAngles[0],
    0.0,
    1.0,
    shoulderRange[0],
    shoulderRange[1]
    )
desiredAngles[1] = utils.changeRange(
    desiredAngles[1],
    0.0,
    1.0,
    elbowRange[0],
    elbowRange[1]
    )



healtyRange = 30 , 40
damagedRange = 90 , 100


# get sim directory
cwd = os.getcwd()
configDir = os.path.join(cwd,'config')

# get first genes
firstGenesDict = {}

reedGenesPath = os.path.join(configDir,'reedParams.json')
with open(reedGenesPath) as fp:
    reedGenesDict = json.load(fp)    
firstGenesDict.update(reedGenesDict)

if model == 'oscillatory':    
    oscGenesPath = os.path.join(configDir,'oscillatoryParams.json')
    with open(oscGenesPath) as fp:
        oscGenesDict = json.load(fp)        
    firstGenesDict.update(oscGenesDict)
        
  
geneNamesList = list(firstGenesDict.keys())
firstGeneValuesList = list(firstGenesDict.values())

for m, gene in enumerate(firstGeneValuesList):
    firstGeneValuesList[m] = \
        firstGeneValuesList[m] + np.random.normal(
            0.0, 
            firstGeneValuesList[m] * 0.0001       
            )



    
#get target values
targetValuesPath = os.path.join(configDir,'brainParams.json')
with open(targetValuesPath) as fp:
    targetValuesDict = json.load(fp)    
targetNamesList = list(targetValuesDict.keys())
targetValuesList = list(targetValuesDict.values())

#get time parameters
nStep = int(tStop * (1 / dt) + 1)
time = np.linspace(tStart, tStop, nStep)  

# get initial state
y0 = copy.deepcopy(targetValuesList)

if model == 'reed':
    y = odeint(
        reedEquationsSystem2, # differential equation system
        y0, # initial state
        time, # integration step 
        args = (firstGeneValuesList, dmgSND8, dmgSND8Start, dmgSND8Mag),
        )
elif model == 'oscillatory':
    y = odeint(
        oscillatoryEquationsSystem2, # differential equation system
        y0, # initial state
        time, # integration step 
        args = (firstGeneValuesList, dmgSND8, dmgSND8Start, dmgSND8Mag, dmgSNDAoscMag),
        )
else:
    print('This model does not exists')






print('first gene fitness')
# get healthy model brain elements steady state
healthySteadyState = np.mean(y[healtyRange[0]*100:healtyRange[1]*100,:],axis=0)
errors = healthySteadyState - np.array(targetValuesList)
percErrors = (errors / healthySteadyState) * 100
absPercErrors = np.abs(percErrors)
meanAPE = np.mean(absPercErrors)
print('mean absolute percentage error : {}'.format(meanAPE))
startFitness = (100 - meanAPE) / 100
print('fitness : {}'.format(startFitness))


# offspringNum = chromoPopNum / survNum
bestTotalFitness = -10.0#copy.copy(startFitness)
bestHealtyOscFitness = -10.0
bestSteadyStateFitness = -10.0
bestDamagedOscFitness = -10.0
totalFitnessHistory = []
steadyStateFitnessHistory = []
healtyOscFitnessHistory = []
damagedOscFitnessHistory = []




# totalFitnessHistory.append(startFitness)


# """
# init simulation plot
# """

# if PLOTTING == True:
#     simFig = plt.figure("ARM SIMULATION")
#     simPlot = simFig.add_subplot(111)
#     simPlot.set_xlim([-1.0,1.0])
#     simPlot.set_ylim([0.0,1.0])   
#     text1 = plt.figtext(
#         .02,
#         .72, 
#         "sec = %s" % (0.00),
#         style='italic',
#         bbox={'facecolor':'yellow'}
#         )    
#     armPlot, = simPlot.plot(
#         [0, arm.xElbow, arm.xEndEf], 
#         [0, arm.yElbow, arm.yEndEf],
#         'k-', 
#         color = 'black', 
#         linewidth = 5
#         )   
#     xDesAng = arm.L1*np.cos(desiredAngles[0]) + \
#                 arm.L2*np.cos(desiredAngles[0]+desiredAngles[1])
#     yDesAng = arm.L1*np.sin(desiredAngles[0]) + \
#                 arm.L2*np.sin(desiredAngles[0]+desiredAngles[1])
#     desEnd, = simPlot.plot(
#         [xDesAng],
#         [yDesAng], 
#         'o', 
#         color = 'green' ,
#         markersize= 10
#         )





for gen in range(genNum):
    
    print('\n**********************NEW GENERATION**************************\n')
    
    # mutRate /= (gen+1)

    # generate new population
    if gen ==0:
        selectedPop = copy.copy(firstGeneValuesList)
    chromoPop = [] 
    for i in range(chromoPopNum):
        newChromo = [] 
        for ii in range(len(selectedPop)):           
            mutatedGene = selectedPop[ii] + \
                np.random.normal(0.0, selectedPop[ii] * mutRate)
            newChromo.append(mutatedGene)
        chromoPop.append(newChromo)


    # get fitness
    popSteadyStateFitness =[]
    popHealtyOscFitness = []
    popDamagedOscFitness = []
    popTotalFitness = []
    for n, chromo in enumerate(chromoPop): 
        print('\n**********************NEW CHROMO**************************\n')
        print('ngen num {} chromo num {} \n'.format(gen, n))
        
        if model == 'reed':
            y = odeint(
                reedEquationsSystem2, # differential equation system
                y0, # initial state
                time, # integration step 
                args = (chromo, dmgSND8, dmgSND8Start, dmgSND8Mag),
                )
        elif model == 'oscillatory':
            y = odeint(
                oscillatoryEquationsSystem2, # differential equation system
                y0, # initial state
                time, # integration step 
                args = (chromo, dmgSND8, dmgSND8Start, dmgSND8Mag, dmgSNDAoscMag),
                )
        else:
            print('This model does not exists')
        
        # get healthy model brain elements steady state
        healthySteadyState = np.mean(y[healtyRange[0]*100:healtyRange[1]*100,:],axis=0)
        
        # healthySteadyState = y[int((dmgSND8Start-tStart)/2 *100),:] 
        errors = healthySteadyState - np.array(targetValuesList)
        percErrors = (errors / healthySteadyState) * 100
        absPercErrors = np.abs(percErrors)
        meanAPE = np.mean(absPercErrors)
        # print('mean absolute percentage error : {}'.format(meanAPE))
        
        
        
        
        
        CX = y[:,3].copy()
        CXHealthyMean = CX[healtyRange[0]*100:healtyRange[1]*100].mean()
        CXHealthyRange = [0,CXHealthyMean*2]
        
        efPosHistory = np.zeros([nStep, 2])
    
        for t in range(nStep):
      
            desiredAngles[0] = utils.changeRange(
                CX[t], #26.3,
                CXHealthyRange[0],
                CXHealthyRange[1],
                shoulderRange[0],
                shoulderRange[1])
               
            desiredAngles[1] = utils.changeRange(
                CX[t],
                CXHealthyRange[0],
                CXHealthyRange[1],
                elbowRange[0],
                elbowRange[1])
               
            Torque = arm.PD_controller(
                [desiredAngles[0],desiredAngles[1]],
                Kp1 ,
                Kp2,
                Kd1, 
                Kd2
                ) # compute torques

            arm.SolveDirectDynamics(Torque[0], Torque[1]) # move the arm
            efPosHistory[t,:] = np.array([arm.xEndEf, arm.yEndEf])
            
            # if PLOTTING == True and t > start_plotting * 100:   
            #     text1.set_text("sec = {}".format(t/100.0))    
            #     xDesAng = arm.L1*np.cos(desiredAngles[0]) +\
            #                 arm.L2*np.cos(desiredAngles[0]+desiredAngles[1])
            #     yDesAng = arm.L1*np.sin(desiredAngles[0]) +\
            #                 arm.L2*np.sin(desiredAngles[0]+desiredAngles[1])
            #     desEnd.set_data([xDesAng],[yDesAng])         
            #     armPlot.set_data([0,arm.xElbow, arm.xEndEf],
            #                       [0,arm.yElbow, arm.yEndEf])
            #     plt.pause(dt)
        
         
        
        
        efX = efPosHistory[:,0].copy()
        efY = efPosHistory[:,1].copy()                                           
        efdX = np.ediff1d(efX, to_begin=np.array([0]))
        efdY = np.ediff1d(efY, to_begin=np.array([0]))    
        euclDist = np.hypot(efdX,efdY)
        
        
        # trem_phys_dist_fig = plt.figure('physiological tremor distance')
        # trem_phys_dist_plot = trem_phys_dist_fig.add_subplot(111) 
        
        
        """
        compute physiological tremor oscillation amplitude
        """   
        tremPhysTime = [healtyRange[0]*100,healtyRange[1]*100]
        tremPhysDist = euclDist[tremPhysTime[0]:tremPhysTime[1]]            
        dTremPhysDist = np.ediff1d(tremPhysDist, to_begin=np.array([0]))
        signDTremPhysDist = np.sign(dTremPhysDist)
        diffSignDTremPhysDist = np.ediff1d(signDTremPhysDist)  
        tremPhysDistMin = np.where(diffSignDTremPhysDist > 1)[0]
        oscillationAmplitudePhysTrem = np.array([])
        oscillationAmplitudePhysTremSum = 0
        for i in range(len(tremPhysDistMin) -1):
            oscillationAmplitudePhysTremSum = 0
            oscillationIntervalPhysTrem = \
                tremPhysDist[tremPhysDistMin[i]:tremPhysDistMin[i+1]]
            oscillationAmplitudePhysTremSum += \
                np.sum(oscillationIntervalPhysTrem)
            oscillationAmplitudePhysTrem = \
                np.hstack([oscillationAmplitudePhysTrem, 
                            oscillationAmplitudePhysTremSum])
        meanPhys = np.mean(oscillationAmplitudePhysTrem)
        stdPhys = np.std(oscillationAmplitudePhysTrem)
        
        if model == 'oscillatory':
            """
            compute parkisonian tremor oscillation amplitude
            """   
            tremParkTime = [damagedRange[0]*100,damagedRange[1]*100]
            tremParkDist = euclDist[tremParkTime[0]:tremParkTime[1]]            
            dTremParkDist = np.ediff1d(tremParkDist, to_begin=np.array([0]))
            signDTremParkDist = np.sign(dTremParkDist)
            diffSignDTremParkDist = np.ediff1d(signDTremParkDist)  
            tremParkDistMin = np.where(diffSignDTremParkDist > 1)[0]
            oscillationAmplitudeParkTrem = np.array([])
            oscillationAmplitudeParkTremSum = 0
            for i in range(len(tremParkDistMin) -1):
                oscillationAmplitudeParkTremSum = 0
                oscillationIntervalParkTrem = \
                    tremParkDist[tremParkDistMin[i]:tremParkDistMin[i+1]]
                oscillationAmplitudeParkTremSum += \
                    np.sum(oscillationIntervalParkTrem)
                oscillationAmplitudeParkTrem = \
                    np.hstack([oscillationAmplitudeParkTrem, 
                                oscillationAmplitudeParkTremSum])
            meanPark = np.mean(oscillationAmplitudeParkTrem)
            stdPark = np.std(oscillationAmplitudeParkTrem)
       
        
       
        
       
        
       
        
        if PLOTTING:
            oscillationAmpFig = \
                plt.figure(
                    "oscillation amplitude",
                    figsize=(32,16)
                    )   
            oscillationAmpPlot = oscillationAmpFig.add_subplot(111)   
            oscillationAmpPlot.set_ylabel(
                "Oscillation amplitude (m)",
                fontsize = 30, 
                fontweight='bold'
                )    
            oscillationAmpPlot.set_ylim(0.0, 1e-2)
            healthyLabel = 'HEALTHY MODEL'
            damagedLabel = 'd8 + {}% = '.format(dmgSND8Mag) # + '/ ' + 'dop_stn = -' + str(DA_weight_reduction) + r'%'
            xTicksLabels = [
                healthyLabel,
                damagedLabel,
                # treat1_label,
                # treat2_label,
                # treat3_label
                ]
    
            oscillationAmpPlot.set_xticks(np.arange(5))
            oscillationAmpPlot.set_xticklabels(xTicksLabels)
            oscillationAmpPlot.tick_params(
                axis='both', 
                which='major',
                labelsize=10
                )
            oscillationAmpPlot.tick_params(
                axis='both', 
                which='minor', 
                labelsize=10
                )
            
            oscillationAmpPlot.scatter(
                1,
                [meanPhys]
                )
            
            oscillationAmpPlot.errorbar(
                [0],
                meanPhys,
                yerr = stdPhys,
                marker= 's', 
                markersize='7' ,
                mec ='black',
                mfc='white',
                ecolor ='black', 
                fmt = '', 
                linestyle='none'
                )
            
            if model == 'oscillatory':
                oscillationAmpPlot.scatter(
                    2,
                    [meanPark]
                    )
                
                oscillationAmpPlot.errorbar(
                    [1],
                    meanPark,
                    yerr = stdPark,
                    marker= 's', 
                    markersize='7' ,
                    mec ='black',
                    mfc='white',
                    ecolor ='black', 
                    fmt = '', 
                    linestyle='none'
                    )




        
        
    
                
        if model == 'oscillatory':
            steadyStateFitness = (100 - meanAPE) / 100
            print('steady state fitness {}\n'.format(steadyStateFitness))
            print('best steady fitness {}\n'.format(bestSteadyStateFitness))
            
            healtyOscFitness = 1 - meanPhys 
            print('healty oscillation fitness {}\n'.format(healtyOscFitness))
            print('best healty oscillation fitness {}\n'.format(bestHealtyOscFitness))
            
            damagedOscFitness = meanPark/0.03 
            if damagedOscFitness > 1:
                damagedOscFitness = 1
            print('damaged oscillation fitness {}\n'.format(damagedOscFitness))
            print('best damaged oscillation fitness {}\n'.format(bestDamagedOscFitness))
            
            totalFitness = copy.copy(steadyStateFitness) + \
                (healtyOscFitness) + \
                (damagedOscFitness)
            print('total fitness {}\n'.format(totalFitness)) 
            print('best total fitness {}\n'.format(bestTotalFitness)) 
            
            
    
        elif model == 'reed':
            steadyStateFitness = (100 - meanAPE) / 100
            totalFitness = copy.copy(steadyStateFitness)
            print('total fitness {}\n'.format(totalFitness))
            print('best total fitness {}\n'.format(bestTotalFitness)) 
        
        
        
        


        
        popSteadyStateFitness.append(steadyStateFitness)
        if model == 'oscillatory':
            popHealtyOscFitness.append(healtyOscFitness)
            popDamagedOscFitness.append(damagedOscFitness)
        popTotalFitness.append(totalFitness)
        
    
        if model == 'oscillatory':
            # select chromosome    
            sortedTotalFitness , sortedPopSteadyStateFitness, \
                sortedPopHealtyOscFitness, sortedPopDamagedOscFitness, sortedPop = \
                zip(*sorted(zip(
                    popTotalFitness,
                    popSteadyStateFitness, 
                    popHealtyOscFitness, 
                    popDamagedOscFitness, 
                    chromoPop
                    ), 
                    reverse=True)
                    )
        elif model == 'reed':
            sortedTotalFitness , sortedPop = \
                zip(*sorted(zip(
                    popTotalFitness,
                    chromoPop
                    ), 
                    reverse=True)
                    )
        
 
    
    
        if sortedTotalFitness[0] > bestTotalFitness:   
            selectedPop = list(sortedPop[:survNum])[0]
            bestY = y.copy()
            bestTotalFitness = copy.copy(sortedTotalFitness[0])
        
            if model == 'oscillatory':
                bestSteadyStateFitness = copy.copy(sortedPopSteadyStateFitness[0])
                bestHealtyOscFitness = copy.copy(sortedPopHealtyOscFitness[0])
                bestDamagedOscFitness = copy.copy(sortedPopDamagedOscFitness[0])
                

        
    # save history        
    if model == 'oscillatory':
        steadyStateFitnessHistory.append(bestSteadyStateFitness)
        healtyOscFitnessHistory.append(bestHealtyOscFitness)
        damagedOscFitnessHistory.append(bestDamagedOscFitness)
    totalFitnessHistory.append(bestTotalFitness)
 
    
    
    

#%%

CXFig = plt.figure("CX")
CXPlot = CXFig.add_subplot(111) 
CXPlot.set_xlim([tStart,tStop*(1/dt)])
CXPlot.set_ylim([0.0,40.0])   

CXPlot.plot(bestY[:,3])

            

#%%
fitnessFig = plt.figure("fitness")


if model == 'oscillatory':
    steadyStateFitnessPlot = fitnessFig.add_subplot(221)
    steadyStateFitnessPlot.plot(steadyStateFitnessHistory, label='steady state fitness')
    steadyStateFitnessPlot.legend(loc='best')
    
    healthyOscFitnessPlot = fitnessFig.add_subplot(222)
    healthyOscFitnessPlot.plot(healtyOscFitnessHistory, label='healty osc fitness')
    healthyOscFitnessPlot.legend(loc='best')
    
    damagedOscFitnessPlot = fitnessFig.add_subplot(223)
    damagedOscFitnessPlot.plot(damagedOscFitnessHistory, label='damaged osc fitness')
    damagedOscFitnessPlot.legend(loc='best')

totalFitnessPlot = fitnessFig.add_subplot(224)
totalFitnessPlot.plot(totalFitnessHistory, label='total fitness')
totalFitnessPlot.legend(loc='best')


#%%

bestConfig = {}

for i , name in enumerate(geneNamesList):
    
    bestConfig[name] = selectedPop[i]
    

#%%
bestConfigDir = os.path.join(os.getcwd(),'bestConfigDir')
if not os.path.exists(bestConfigDir):
    os.makedirs(bestConfigDir)
#%%
    
bestConfigPath = os.path.join(
    bestConfigDir,
    'oscillatoryModel.json'
    )

with open(bestConfigPath, 'w') as fp:
    json.dump(bestConfig, fp)

# print(bestConfig)



