# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 23:20:31 2020

@author: adria
"""

import numpy as np
import copy

def reedEquationsSystem(x,t, a1c, a1da, d1, a2c, a2da, d2, a3 , a3md, a3mi, d3, 
    a4th, d4, a5, a5cx, a5sn, d5, G, d6, a7, d7, a8, a8drn,d8, dmgSND8 = True, 
    dmgSND8Start = 75, dmgSND8Mag = +70.):
    """
    REED MODEL 
    A differential equations system to simulate the activities of different 
    connected brain areas involved in motor behaviours
    
    Parameters
    ----------
    x : TYPE numpy ndarray
        DESCRIPTION. integrated values array
    t : TYPE float
        DESCRIPTION. simulation timestamp
    a1c : TYPE float
        DESCRIPTION. Cortical input to medium spiny neurons in indirect pathway
    a1da : TYPE float
        DESCRIPTION. Influence of DA on medium spiny neurons in indirect pathway
    d1 : TYPE float
        DESCRIPTION. Decay constant of medium spiny neurons in indirect pathway
    a2c : TYPE float
        DESCRIPTION. Cortical input to medium spiny neurons in direct pathway
    a2da : TYPE float
        DESCRIPTION. Influence of DA on medium spiny neurons indirect pathway
    d2 : TYPE float
        DESCRIPTION. Decay constant of medium spiny neurons in direct pathway
    a3 : TYPE float
        DESCRIPTION. External drive to thalamus
    a3md : TYPE float
        DESCRIPTION. Excitation of direct pathway on the thalamus
    a3mi : TYPE float
        DESCRIPTION. Inhibition of indirect pathway on the thalamus
    d3 : TYPE float
        DESCRIPTION. Decay constant of thalamic neurons
    a4th : TYPE float
        DESCRIPTION. Influence of thalamus on cortex
    d4 : TYPE float
        DESCRIPTION. Decay constant of cortical neurons
    a5 : TYPE float
        DESCRIPTION. External drive to the DRN  
    a5cx : TYPE float
        DESCRIPTION. Excitatory influence of the SNc on the DRN
    a5sn : TYPE float
        DESCRIPTION. Inhibition of the DRN neuron by the cortex
    d5 : TYPE float
        DESCRIPTION. Decay constant of DRN neurons
    G : TYPE float
        DESCRIPTION. Influence per nM 5-HT on DA release in the striatum
    d6 : TYPE float
        DESCRIPTION. Decay constant of DA in the striatum
    a7 : TYPE float
        DESCRIPTION. Influence of DRN firing on 5-HT release in the striatum
    d7 : TYPE float
        DESCRIPTION. Decay constant of 5-HT in the striatum 
    a8 : TYPE float
        DESCRIPTION. External drive of the SNc
    a8drn : TYPE float
        DESCRIPTION. Inhibition of SNc neurons by the DRN
    d8 : TYPE float
        DESCRIPTION. # Decay constant of SNc neurons
    dmgSND8 : TYPE bool, optional
        DESCRIPTION. Simulate SN damage (d8)
        The default is True.
    dmgSND8Start : TYPE int, optional
        DESCRIPTION. SN damage start (s).
        The default is 75. 
    dmgSND8Mag : TYPE float, optional
        DESCRIPTION. SN damage magnitude (%). 
        The default is +70.  

    Returns
    -------
    list
        DESCRIPTION.

    """
    
    # Get brain areas activities           
    MI = x[0] # indirect pathway activity (Hz)
    MD = x[1] # direct path activity (Hz)
    TH = x[2] # thalamus activity (Hz)
    CX = x[3] # M1 activity (Hz) 
    DRN = x[4] # dorsal raphe nucleus activity (Hz)
    DA = x[5] # DA concentration in striatum (nM)
    HT5 = x[6] # 5-HT concentration in striatum (nM)
    SN = x[7] # substantia nigra pars compacta activity (Hz)
    
    # increase d8 to simulate substantia nigra damage 
    if dmgSND8 == True and t >= dmgSND8Start:
        d8 *= 1.0 + (dmgSND8Mag / 100.)
   
    # brain areas diffential equation                
    dMIdt = a1c - a1da * DA - d1 * MI # indirect pathway
    dMDdt = a2c + a2da * DA - d2 * MD # direct pathway
    dTHdt = a3 + a3md * MD - a3mi * MI - d3 * TH # thalamus
    dCXdt = a4th * TH - d4 * CX # M1
    dDRNdt = a5 - a5cx * CX + a5sn * SN - d5 * DRN # dorsal raphe nucleus
    dDAdt = G * HT5 * SN - d6 * DA # striatal dopamine
    d5HTdt = a7  * DRN - d7  * HT5 # striatal serotonine
    dSNdt = a8 - a8drn * DRN - d8 * SN # substantia nigra pars compacta     
    return [dMIdt, dMDdt, dTHdt, dCXdt, dDRNdt, dDAdt, d5HTdt, dSNdt]


def reedEquationsSystem2(x,t,reedParams,dmgSND8,dmgSND8Start,dmgSND8Mag):
    """
    REED MODEL 
    A differential equations system to simulate the activities of different 
    connected brain areas involved in motor behaviours
    
    Parameters
    ----------
    x : TYPE numpy ndarray
        DESCRIPTION. integrated values array
    t : TYPE float
        DESCRIPTION. simulation timestamp
    a1c : TYPE float
        DESCRIPTION. Cortical input to medium spiny neurons in indirect pathway
    a1da : TYPE float
        DESCRIPTION. Influence of DA on medium spiny neurons in indirect pathway
    d1 : TYPE float
        DESCRIPTION. Decay constant of medium spiny neurons in indirect pathway
    a2c : TYPE float
        DESCRIPTION. Cortical input to medium spiny neurons in direct pathway
    a2da : TYPE float
        DESCRIPTION. Influence of DA on medium spiny neurons indirect pathway
    d2 : TYPE float
        DESCRIPTION. Decay constant of medium spiny neurons in direct pathway
    a3 : TYPE float
        DESCRIPTION. External drive to thalamus
    a3md : TYPE float
        DESCRIPTION. Excitation of direct pathway on the thalamus
    a3mi : TYPE float
        DESCRIPTION. Inhibition of indirect pathway on the thalamus
    d3 : TYPE float
        DESCRIPTION. Decay constant of thalamic neurons
    a4th : TYPE float
        DESCRIPTION. Influence of thalamus on cortex
    d4 : TYPE float
        DESCRIPTION. Decay constant of cortical neurons
    a5 : TYPE float
        DESCRIPTION. External drive to the DRN  
    a5cx : TYPE float
        DESCRIPTION. Excitatory influence of the SNc on the DRN
    a5sn : TYPE float
        DESCRIPTION. Inhibition of the DRN neuron by the cortex
    d5 : TYPE float
        DESCRIPTION. Decay constant of DRN neurons
    G : TYPE float
        DESCRIPTION. Influence per nM 5-HT on DA release in the striatum
    d6 : TYPE float
        DESCRIPTION. Decay constant of DA in the striatum
    a7 : TYPE float
        DESCRIPTION. Influence of DRN firing on 5-HT release in the striatum
    d7 : TYPE float
        DESCRIPTION. Decay constant of 5-HT in the striatum 
    a8 : TYPE float
        DESCRIPTION. External drive of the SNc
    a8drn : TYPE float
        DESCRIPTION. Inhibition of SNc neurons by the DRN
    d8 : TYPE float
        DESCRIPTION. # Decay constant of SNc neurons
    dmgSND8 : TYPE bool, optional
        DESCRIPTION. Simulate SN damage (d8)
        The default is True.
    dmgSND8Start : TYPE int, optional
        DESCRIPTION. SN damage start (s).
        The default is 75. 
    dmgSND8Mag : TYPE float, optional
        DESCRIPTION. SN damage magnitude (%). 
        The default is +70.  

    Returns
    -------
    list
        DESCRIPTION.

    """
    
    # Get brain areas activities           
    MI = x[0] # indirect pathway activity (Hz)
    MD = x[1] # direct path activity (Hz)
    TH = x[2] # thalamus activity (Hz)
    CX = x[3] # M1 activity (Hz) 
    DRN = x[4] # dorsal raphe nucleus activity (Hz)
    DA = x[5] # DA concentration in striatum (nM)
    HT5 = x[6] # 5-HT concentration in striatum (nM)
    SN = x[7] # substantia nigra pars compacta activity (Hz)
    
    
    
    d8 = copy.copy(reedParams[22])
    # # increase d8 to simulate substantia nigra damage 
    if dmgSND8 == True:
        if t >= dmgSND8Start:
            # print('before ',d8)
            d8 *= 1.0 + (dmgSND8Mag / 100.)
            # print('after', d8)
    
   
    # brain areas diffential equation                
    dMIdt = reedParams[0] - reedParams[1] * DA - reedParams[2] * MI # indirect pathway
    dMDdt = reedParams[3] + reedParams[4] * DA - reedParams[5] * MD # direct pathway
    dTHdt = reedParams[6] + reedParams[7] * MD - reedParams[8] * MI - reedParams[9]* TH # thalamus
    dCXdt = reedParams[10] * TH - reedParams[11] * CX # M1
    dDRNdt = reedParams[12] - reedParams[13] * CX + reedParams[14] * SN - reedParams[15] * DRN # dorsal raphe nucleus
    dDAdt = reedParams[16] * HT5 * SN - reedParams[17] * DA # striatal dopamine
    d5HTdt = reedParams[18] * DRN - reedParams[19] * HT5 # striatal serotonine
    dSNdt = reedParams[20] - reedParams[21] * DRN - d8 * SN # substantia nigra pars compacta     
    return [dMIdt, dMDdt, dTHdt, dCXdt, dDRNdt, dDAdt, d5HTdt, dSNdt]

def oscillatoryEquationsSystem(x, t, a1c, a1da, d1, aOsc, fOsc, daOsc, a2c, 
    a2da, d2, a3 , a3md, a3mi, d3, a4th, d4, a5, a5cx, a5sn, d5, G, d6, a7, d7, 
    a8, a8drn, d8, dmgSND8, dmgSND8Start, dmgSND8Mag, dmgDAOsc, trmtD7, 
    trmtD7Start1, trmtD7Mag1, trmtD7Start2, trmtD7Mag2, trmtD7Start3, trmtD7Mag3,
    startD7):
    """
    OSCILLATORY MODEL 
    A differential equations system to simulate the activities of different 
    connected brain areas involved in motor behaviours. An oscillatory component
    in the activity of the indirect pathway has been added to the Reed model. 
    The oscillatory component is inhibited by the concentration of striatal DA.

    Parameters
    ----------
    x : TYPE numpy ndarray
        DESCRIPTION. integrated values array
    t : TYPE float
        DESCRIPTION. simulation timestamp
    a1c : TYPE float
        DESCRIPTION. Cortical input to medium spiny neurons in indirect pathway
    a1da : TYPE float
        DESCRIPTION. Influence of DA on medium spiny neurons in indirect pathway
    d1 : TYPE float
        DESCRIPTION. Decay constant of medium spiny neurons in indirect pathway
    aOsc : TYPE float
        DESCRIPTION. Oscillatory amplitude in indirect pathway
    fOsc : TYPE float
        DESCRIPTION. Oscillatory frequency  in indirect pathway
    daOsc : TYPE float
        DESCRIPTION. Influence of DA on the inhibition of the oscillatory component
    a2c : TYPE float
        DESCRIPTION. Cortical input to medium spiny neurons in direct pathway
    a2da : TYPE float
        DESCRIPTION. Influence of DA on medium spiny neurons indirect pathway
    d2 : TYPE float
        DESCRIPTION. Decay constant of medium spiny neurons in direct pathway
    a3 : TYPE float
        DESCRIPTION. External drive to thalamus
    a3md : TYPE float
        DESCRIPTION. Excitation of direct pathway on the thalamus
    a3mi : TYPE float
        DESCRIPTION. Inhibition of indirect pathway on the thalamus
    d3 : TYPE float
        DESCRIPTION. Decay constant of thalamic neurons
    a4th : TYPE float
        DESCRIPTION. Influence of thalamus on cortex
    d4 : TYPE float
        DESCRIPTION. Decay constant of cortical neurons
    a5 : TYPE float
        DESCRIPTION. External drive to the DRN  
    a5cx : TYPE float
        DESCRIPTION. Excitatory influence of the SNc on the DRN
    a5sn : TYPE float
        DESCRIPTION. Inhibition of the DRN neuron by the cortex
    d5 : TYPE float
        DESCRIPTION. Decay constant of DRN neurons
    G : TYPE float
        DESCRIPTION. Influence per nM 5-HT on DA release in the striatum
    d6 : TYPE float
        DESCRIPTION. Decay constant of DA in the striatum
    a7 : TYPE float
        DESCRIPTION. Influence of DRN firing on 5-HT release in the striatum
    d7 : TYPE float
        DESCRIPTION. Decay constant of 5-HT in the striatum 
    a8 : TYPE float
        DESCRIPTION. External drive of the SNc
    a8drn : TYPE float
        DESCRIPTION. Inhibition of SNc neurons by the DRN
    d8 : TYPE float
        DESCRIPTION. # Decay constant of SNc neurons
    dmgSND8 : TYPE bool, optional
        DESCRIPTION. Simulate SN damage (d8)
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

    Returns
    -------
    list
        DESCRIPTION.

    """
         
    # Get brain areas activities           
    MI = x[0] # indirect pathway activity (Hz)
    MD = x[1] # direct path activity (Hz)
    TH = x[2] # thalamus activity (Hz)
    CX = x[3] # M1 activity (Hz) 
    DRN = x[4] # dorsal raphe nucleus activity (Hz)
    DA = x[5] # DA concentration in striatum (nM)
    HT5 = x[6] # 5-HT concentration in striatum (nM)
    SN = x[7] # substantia nigra pars compacta activity (Hz)
     
    # Simulate SN damage (increase d8)
    if dmgSND8 == True and t >= dmgSND8Start:
        # Increase d8 to simulate substantia nigra damage   
        d8 *= 1.0 + (dmgSND8Mag / 100.)     
        # Reduce the influence of DA in the inhibition of the oscillatory component
        daOsc *= 1.0 + (dmgDAOsc / 100) 
    
    # Simulate SSRI treatment (reduce d7)
    if trmtD7 == True:
        # First simulated treatment magnitude
        if t >= trmtD7Start1:              
            d7 = startD7
            d7 *= 1.0 + (trmtD7Mag1 / 100)
        # Second simulated treatment magnitude
        if t >= trmtD7Start2:              
            d7 = startD7
            d7 *= 1.0 + (trmtD7Mag2 / 100)
        # Third simulated treatment magnitude
        if t >= trmtD7Start3:              
            d7 = startD7
            d7 *= 1.0 + (trmtD7Mag3 / 100)
            
    # brain areas diffential equation           
    dMIdt = a1c - a1da * DA - d1 * MI + ((aOsc * np.sin(fOsc*t)) / (daOsc*DA)) # indirect pathway
    dMDdt = a2c + a2da * DA - d2 * MD # direct pathway
    dTHdt = a3 + a3md * MD - a3mi * MI - d3 * TH # Thalamus
    dCXdt = a4th * TH - d4 * CX # M1
    dDRNdt = a5 - a5cx * CX + a5sn * SN - d5 * DRN # dorsal raphe nucleus
    dDAdt = G * HT5 * SN - d6 * DA # striatal dopamine
    d5HTdt = a7  * DRN - d7  * HT5 # striatal serotonine
    dSNdt = a8 - a8drn * DRN - d8 * SN # substantia nigra pars compacta     
    return [dMIdt, dMDdt, dTHdt, dCXdt, dDRNdt, dDAdt, d5HTdt, dSNdt]



def drnDamageEquationsSystem(x, t, a1c, a1da, d1, aOsc, fOsc, daOsc, a2c, 
    a2da, d2, a3 , a3md, a3mi, d3, a4th, d4, a5, a5cx, a5sn, d5, G, d6, a7, d7, 
    a8, a8drn, d8, dmgDRND7, dmgDRND7Start1, dmgSND8Mag1, dmgDRND7Start2, dmgSND8Mag2, 
    dmgDRND7Start3, dmgSND8Mag3, 
    startD5, startD7):
    """
    OSCILLATORY MODEL 
    A differential equations system to simulate the activities of different 
    connected brain areas involved in motor behaviours. An oscillatory component
    in the activity of the indirect pathway has been added to the Reed model. 
    The oscillatory component is inhibited by the concentration of striatal DA.

    Parameters
    ----------
    x : TYPE numpy ndarray
        DESCRIPTION. integrated values array
    t : TYPE float
        DESCRIPTION. simulation timestamp
    a1c : TYPE float
        DESCRIPTION. Cortical input to medium spiny neurons in indirect pathway
    a1da : TYPE float
        DESCRIPTION. Influence of DA on medium spiny neurons in indirect pathway
    d1 : TYPE float
        DESCRIPTION. Decay constant of medium spiny neurons in indirect pathway
    aOsc : TYPE float
        DESCRIPTION. Oscillatory amplitude in indirect pathway
    fOsc : TYPE float
        DESCRIPTION. Oscillatory frequency  in indirect pathway
    daOsc : TYPE float
        DESCRIPTION. Influence of DA on the inhibition of the oscillatory component
    a2c : TYPE float
        DESCRIPTION. Cortical input to medium spiny neurons in direct pathway
    a2da : TYPE float
        DESCRIPTION. Influence of DA on medium spiny neurons indirect pathway
    d2 : TYPE float
        DESCRIPTION. Decay constant of medium spiny neurons in direct pathway
    a3 : TYPE float
        DESCRIPTION. External drive to thalamus
    a3md : TYPE float
        DESCRIPTION. Excitation of direct pathway on the thalamus
    a3mi : TYPE float
        DESCRIPTION. Inhibition of indirect pathway on the thalamus
    d3 : TYPE float
        DESCRIPTION. Decay constant of thalamic neurons
    a4th : TYPE float
        DESCRIPTION. Influence of thalamus on cortex
    d4 : TYPE float
        DESCRIPTION. Decay constant of cortical neurons
    a5 : TYPE float
        DESCRIPTION. External drive to the DRN  
    a5cx : TYPE float
        DESCRIPTION. Excitatory influence of the SNc on the DRN
    a5sn : TYPE float
        DESCRIPTION. Inhibition of the DRN neuron by the cortex
    d5 : TYPE float
        DESCRIPTION. Decay constant of DRN neurons
    G : TYPE float
        DESCRIPTION. Influence per nM 5-HT on DA release in the striatum
    d6 : TYPE float
        DESCRIPTION. Decay constant of DA in the striatum
    a7 : TYPE float
        DESCRIPTION. Influence of DRN firing on 5-HT release in the striatum
    d7 : TYPE float
        DESCRIPTION. Decay constant of 5-HT in the striatum 
    a8 : TYPE float
        DESCRIPTION. External drive of the SNc
    a8drn : TYPE float
        DESCRIPTION. Inhibition of SNc neurons by the DRN
    d8 : TYPE float
        DESCRIPTION. # Decay constant of SNc neurons
    dmgSND8 : TYPE bool, optional
        DESCRIPTION. Simulate SN damage (d8)
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

    Returns
    -------
    list
        DESCRIPTION.

    """
         
    # Get brain areas activities           
    MI = x[0] # indirect pathway activity (Hz)
    MD = x[1] # direct path activity (Hz)
    TH = x[2] # thalamus activity (Hz)
    CX = x[3] # M1 activity (Hz) 
    DRN = x[4] # dorsal raphe nucleus activity (Hz)
    DA = x[5] # DA concentration in striatum (nM)
    HT5 = x[6] # 5-HT concentration in striatum (nM)
    SN = x[7] # substantia nigra pars compacta activity (Hz)
     
    
    # Simulate SSRI treatment (reduce d7)
    if dmgDRND7 == True:
        # First simulated treatment magnitude
        if t >= dmgDRND7Start1:              
            d5 = startD5
            d5 *= 1.0 + (dmgSND8Mag1 / 100.)
            # d7 = startD7
            # d7 *= 1.0 + (dmgSND8Mag1 / 100.)
          
        # Second simulated treatment magnitude
        if t >= dmgDRND7Start2:              
            d5 = startD5
            d5 *= 1.0 + (dmgSND8Mag2 / 100.)
            # d7 = startD7
            # d7 *= 1.0 + (dmgSND8Mag2 / 100.)
        # Third simulated treatment magnitude
        if t >= dmgDRND7Start3:              
            d5 = startD5
            d5 *= 1.0 + (dmgSND8Mag3 / 100.) 
            # d7 = startD7
            # d7 *= 1.0 + (dmgSND8Mag3 / 100.) 
            
    # brain areas diffential equation           
    dMIdt = a1c - a1da * DA - d1 * MI + ((aOsc * np.sin(fOsc*t)) / (daOsc*DA)) # indirect pathway
    dMDdt = a2c + a2da * DA - d2 * MD # direct pathway
    dTHdt = a3 + a3md * MD - a3mi * MI - d3 * TH # Thalamus
    dCXdt = a4th * TH - d4 * CX # M1
    dDRNdt = a5 - a5cx * CX + a5sn * SN - d5 * DRN # dorsal raphe nucleus
    dDAdt = G * HT5 * SN - d6 * DA # striatal dopamine
    d5HTdt = a7  * DRN - d7  * HT5 # striatal serotonine
    dSNdt = a8 - a8drn * DRN - d8 * SN # substantia nigra pars compacta     
    return [dMIdt, dMDdt, dTHdt, dCXdt, dDRNdt, dDAdt, d5HTdt, dSNdt]











def oscillatoryEquationsSystemXXX(x, t, oscParams, dmgSND8 = True, dmgSND8Start = 75, 
       dmgSND8Mag = +70.):
    """
    OSCILLATORY MODEL 
    A differential equations system to simulate the activities of different 
    connected brain areas involved in motor behaviours. An oscillatory component
    in the activity of the indirect pathway has been added to the Reed model. 
    The oscillatory component is inhibited by the concentration of striatal DA.

    Parameters
    ----------
    x : TYPE numpy ndarray
        DESCRIPTION. integrated values array
    t : TYPE float
        DESCRIPTION. simulation timestamp
    a1c : TYPE float
        DESCRIPTION. Cortical input to medium spiny neurons in indirect pathway
    a1da : TYPE float
        DESCRIPTION. Influence of DA on medium spiny neurons in indirect pathway
    d1 : TYPE float
        DESCRIPTION. Decay constant of medium spiny neurons in indirect pathway
    aOsc : TYPE float
        DESCRIPTION. Oscillatory amplitude in indirect pathway
    fOsc : TYPE float
        DESCRIPTION. Oscillatory frequency  in indirect pathway
    daOsc : TYPE float
        DESCRIPTION. Influence of DA on the inhibition of the oscillatory component
    a2c : TYPE float
        DESCRIPTION. Cortical input to medium spiny neurons in direct pathway
    a2da : TYPE float
        DESCRIPTION. Influence of DA on medium spiny neurons indirect pathway
    d2 : TYPE float
        DESCRIPTION. Decay constant of medium spiny neurons in direct pathway
    a3 : TYPE float
        DESCRIPTION. External drive to thalamus
    a3md : TYPE float
        DESCRIPTION. Excitation of direct pathway on the thalamus
    a3mi : TYPE float
        DESCRIPTION. Inhibition of indirect pathway on the thalamus
    d3 : TYPE float
        DESCRIPTION. Decay constant of thalamic neurons
    a4th : TYPE float
        DESCRIPTION. Influence of thalamus on cortex
    d4 : TYPE float
        DESCRIPTION. Decay constant of cortical neurons
    a5 : TYPE float
        DESCRIPTION. External drive to the DRN  
    a5cx : TYPE float
        DESCRIPTION. Excitatory influence of the SNc on the DRN
    a5sn : TYPE float
        DESCRIPTION. Inhibition of the DRN neuron by the cortex
    d5 : TYPE float
        DESCRIPTION. Decay constant of DRN neurons
    G : TYPE float
        DESCRIPTION. Influence per nM 5-HT on DA release in the striatum
    d6 : TYPE float
        DESCRIPTION. Decay constant of DA in the striatum
    a7 : TYPE float
        DESCRIPTION. Influence of DRN firing on 5-HT release in the striatum
    d7 : TYPE float
        DESCRIPTION. Decay constant of 5-HT in the striatum 
    a8 : TYPE float
        DESCRIPTION. External drive of the SNc
    a8drn : TYPE float
        DESCRIPTION. Inhibition of SNc neurons by the DRN
    d8 : TYPE float
        DESCRIPTION. # Decay constant of SNc neurons
    dmgSND8 : TYPE bool, optional
        DESCRIPTION. Simulate SN damage (d8)
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

    Returns
    -------
    list
        DESCRIPTION.

    """
         
    # Get brain areas activities           
    MI = x[0] # indirect pathway activity (Hz)
    MD = x[1] # direct path activity (Hz)
    TH = x[2] # thalamus activity (Hz)
    CX = x[3] # M1 activity (Hz) 
    DRN = x[4] # dorsal raphe nucleus activity (Hz)
    DA = x[5] # DA concentration in striatum (nM)
    HT5 = x[6] # 5-HT concentration in striatum (nM)
    SN = x[7] # substantia nigra pars compacta activity (Hz)
    
    d8 = copy.copy(oscParams[22])
    # Simulate SN damage (increase d8)
    if dmgSND8 == True and t >= dmgSND8Start:
        # print('******************************************************damaged')
        # Increase d8 to simulate substantia nigra damage   
        oscParams[22] *= 1.0 + (dmgSND8Mag / 100.)     
        # Reduce the influence of DA in the inhibition of the oscillatory component
        # oscParams[25] *= 1.0 + (dmgDAOsc / 100) 
    # else:
        # print('******************************************************healthy')
    # Simulate SSRI treatment (reduce d7)
    # if trmtD7 == True:
    #     # First simulated treatment magnitude
    #     if t >= trmtD7Start1:              
    #         oscParams[19] = 2.0
    #         oscParams[19] *= 1.0 + (trmtD7Mag1 / 100)
    #     # Second simulated treatment magnitude
    #     if t >= trmtD7Start2:              
    #         oscParams[19] = 2.0
    #         oscParams[19] *= 1.0 + (trmtD7Mag2 / 100)
    #     # Third simulated treatment magnitude
    #     if t >= trmtD7Start3:              
    #         oscParams[19] = 2.0
    #         oscParams[19] *= 1.0 + (trmtD7Mag3 / 100)
            
    # brain areas diffential equation           
    # brain areas diffential equation                
    dMIdt = oscParams[0] - oscParams[1] * DA - oscParams[2] * MI # indirect pathway
    dMDdt = oscParams[3] + oscParams[4] * DA - oscParams[5] * MD # direct pathway
    dTHdt = oscParams[6] + oscParams[7] * MD - oscParams[8] * MI - oscParams[9]* TH # thalamus
    dCXdt = oscParams[10] * TH - oscParams[11] * CX # M1
    dDRNdt = oscParams[12] - oscParams[13] * CX + oscParams[14] * SN - oscParams[15] * DRN # dorsal raphe nucleus
    dDAdt = oscParams[16] * HT5 * SN - oscParams[17] * DA # striatal dopamine
    d5HTdt = oscParams[18] * DRN - oscParams[19] * HT5 # striatal serotonine
    dSNdt = oscParams[20] - oscParams[21] * DRN - d8 * SN # substantia nigra pars compacta 
    return [dMIdt, dMDdt, dTHdt, dCXdt, dDRNdt, dDAdt, d5HTdt, dSNdt]




def oscillatoryEquationsSystem2(x,t,reedParams,dmgSND8,dmgSND8Start,dmgSND8Mag, dmgDAOsc):
    """
    REED MODEL 
    A differential equations system to simulate the activities of different 
    connected brain areas involved in motor behaviours
    
    Parameters
    ----------
    x : TYPE numpy ndarray
        DESCRIPTION. integrated values array
    t : TYPE float
        DESCRIPTION. simulation timestamp
    a1c : TYPE float
        DESCRIPTION. Cortical input to medium spiny neurons in indirect pathway
    a1da : TYPE float
        DESCRIPTION. Influence of DA on medium spiny neurons in indirect pathway
    d1 : TYPE float
        DESCRIPTION. Decay constant of medium spiny neurons in indirect pathway
    a2c : TYPE float
        DESCRIPTION. Cortical input to medium spiny neurons in direct pathway
    a2da : TYPE float
        DESCRIPTION. Influence of DA on medium spiny neurons indirect pathway
    d2 : TYPE float
        DESCRIPTION. Decay constant of medium spiny neurons in direct pathway
    a3 : TYPE float
        DESCRIPTION. External drive to thalamus
    a3md : TYPE float
        DESCRIPTION. Excitation of direct pathway on the thalamus
    a3mi : TYPE float
        DESCRIPTION. Inhibition of indirect pathway on the thalamus
    d3 : TYPE float
        DESCRIPTION. Decay constant of thalamic neurons
    a4th : TYPE float
        DESCRIPTION. Influence of thalamus on cortex
    d4 : TYPE float
        DESCRIPTION. Decay constant of cortical neurons
    a5 : TYPE float
        DESCRIPTION. External drive to the DRN  
    a5cx : TYPE float
        DESCRIPTION. Excitatory influence of the SNc on the DRN
    a5sn : TYPE float
        DESCRIPTION. Inhibition of the DRN neuron by the cortex
    d5 : TYPE float
        DESCRIPTION. Decay constant of DRN neurons
    G : TYPE float
        DESCRIPTION. Influence per nM 5-HT on DA release in the striatum
    d6 : TYPE float
        DESCRIPTION. Decay constant of DA in the striatum
    a7 : TYPE float
        DESCRIPTION. Influence of DRN firing on 5-HT release in the striatum
    d7 : TYPE float
        DESCRIPTION. Decay constant of 5-HT in the striatum 
    a8 : TYPE float
        DESCRIPTION. External drive of the SNc
    a8drn : TYPE float
        DESCRIPTION. Inhibition of SNc neurons by the DRN
    d8 : TYPE float
        DESCRIPTION. # Decay constant of SNc neurons
    dmgSND8 : TYPE bool, optional
        DESCRIPTION. Simulate SN damage (d8)
        The default is True.
    dmgSND8Start : TYPE int, optional
        DESCRIPTION. SN damage start (s).
        The default is 75. 
    dmgSND8Mag : TYPE float, optional
        DESCRIPTION. SN damage magnitude (%). 
        The default is +70.  

    Returns
    -------
    list
        DESCRIPTION.

    """
    
    # Get brain areas activities           
    MI = x[0] # indirect pathway activity (Hz)
    MD = x[1] # direct path activity (Hz)
    TH = x[2] # thalamus activity (Hz)
    CX = x[3] # M1 activity (Hz) 
    DRN = x[4] # dorsal raphe nucleus activity (Hz)
    DA = x[5] # DA concentration in striatum (nM)
    HT5 = x[6] # 5-HT concentration in striatum (nM)
    SN = x[7] # substantia nigra pars compacta activity (Hz)
    
    
    
    d8 = copy.copy(reedParams[22])
    daOsc = copy.copy(reedParams[25])
    # # increase d8 to simulate substantia nigra damage 
    if dmgSND8 == True:
        if t >= dmgSND8Start:
            # print('before ',d8)
            d8 *= 1.0 + (dmgSND8Mag / 100.)
            # print('after', d8)
            daOsc *= 1.0 + (dmgDAOsc / 100.) 
   
   
    # brain areas diffential equation                
    dMIdt = reedParams[0] - reedParams[1] * DA - reedParams[2] * MI + ((reedParams[23] * np.sin(reedParams[24]*t)) / (daOsc*DA))  # indirect pathway
    dMDdt = reedParams[3] + reedParams[4] * DA - reedParams[5] * MD # direct pathway
    dTHdt = reedParams[6] + reedParams[7] * MD - reedParams[8] * MI - reedParams[9]* TH # thalamus
    dCXdt = reedParams[10] * TH - reedParams[11] * CX # M1
    dDRNdt = reedParams[12] - reedParams[13] * CX + reedParams[14] * SN - reedParams[15] * DRN # dorsal raphe nucleus
    dDAdt = reedParams[16] * HT5 * SN - reedParams[17] * DA # striatal dopamine
    d5HTdt = reedParams[18] * DRN - reedParams[19] * HT5 # striatal serotonine
    dSNdt = reedParams[20] - reedParams[21] * DRN - d8 * SN # substantia nigra pars compacta     
    return [dMIdt, dMDdt, dTHdt, dCXdt, dDRNdt, dDAdt, d5HTdt, dSNdt]










