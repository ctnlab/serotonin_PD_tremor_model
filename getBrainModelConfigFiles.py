# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 22:43:08 2020

@author: adria
"""


import json
import os

#%%

"""
create config directory
"""

cwd = os.getcwd()
configDir = os.path.join(cwd, 'config')
if not os.path.exists(configDir):
    os.makedirs(configDir)

#%%
"""
brain areas steady state (experimental values)
"""

brainAreasSteadyState = {
    'MI_0' : 1.88, # Firing rate of striatal spiny neuron in the indirect pathway (Hz) 
    'MD_0' : 1.85, # Firing rate of striatal spiny neuron in the direct pathway (Hz)
    'TH_0' : 17.5, # Firing rate of thalamic neuron (Hz)
    'CX_0' : 26.3, # Firing rate of cortical neuron (Hz)
    'DRN_0' : 1.41, # Firing rate of dorsal raphe nucleus neuron (Hz)
    'DA_0' :  2.72, # Concentration of dopamine in the striatum (nM)
    '5-HT_0' : 0.846, # Concentration of serotonin in the striatum (nM)
    'SN_0' :  4.47 # Firing rate of substantia nigra pars compacta neuron (Hz)
    }

steadyStatePath = os.path.join(
    configDir,
    'brainParams.json'
    )

with open(steadyStatePath, 'w') as fp:
    json.dump(brainAreasSteadyState, fp)


#%%
"""
Reed model adjusted parameters 
"""
   
adjustedParams = {
    'a1c' : 2.333,  # Cortical input to medium spiny neurons in indirect pathway
    'a1da' : 0.167,  # Influence of DA on medium spiny neurons in indirect pathway
    'd1': 1.0, # Decay constant of medium spiny neurons in indirect pathway
    'a2c' : 1.167, # Cortical input to medium spiny neurons in direct pathway
    'a2da' : 0.250,  # Influence of DA on medium spiny neurons indirect pathway
    'd2' : 1.0, # Decay constant of medium spiny neurons in direct pathway
    'a3' : 1.667, # External drive to thalamus
    'a3md' : 3.5, # Excitation of direct pathway on the thalamus
    'a3mi' : 2.0, # Inhibition of indirect pathway on the thalamus
    'd3' : 0.25, # Decay constant of thalamic neurons
    'a4th' : 1.5, # Influence of thalamus on cortex
    'd4' : 1.0, # Decay constant of cortical neurons 
    'a5' : 6.667, # External drive to the DRN                 
    'a5cx' : 0.175, # Inhibition of the DRN neuron by the cortex
    'a5sn' : 0.01, # Excitatory influence of the SNc on the DRN
    'd5' : 1.5, # Decay constant of DRN neurons
    'G' : 0.72, # Influence per nM 5-HT on DA release in the striatum
    'd6' : 1.0, # Decay constant of DA in the striatum
    'a7' : 1.2, # Influence of DRN firing on 5-HT release in the striatum
    'd7' : 2.0, # Decay constant of 5-HT in the striatum 
    'a8' : 58.833, # External drive of the SNc
    'a8drn' : 10., # Inhibition of SNc neurons by the DRN
    'd8' : 10. # Decay constant of SNc neurons
    }

adjustedParamsPath = os.path.join(
    configDir,
    'reedParams.json'
    )

with open(adjustedParamsPath, 'w') as fp:
    json.dump(adjustedParams, fp)
    
#%%
"""
oscillatory parameters
"""

oscParams = {
    'aOsc' : 15, # oscillatory amplitude in indirect pathway
    'fOsc' : 200, # oscillatory frequency  in indirect pathway
    'daOsc': 30, # influence of dopamine on oscillaton amplitude 
    }

oscParamsPath = os.path.join(
    configDir,
    'oscillatoryParams.json'
    )

with open(oscParamsPath, 'w') as fp:
    json.dump(oscParams, fp)




















    

