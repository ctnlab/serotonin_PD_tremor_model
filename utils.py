# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 17:41:51 2020

@author: adria
"""

import matplotlib.pyplot as plt
import numpy as np

def plotData(figureName ='Example plot',figSize=(32, 16),
             xLabel='Independent variable',yLabel='Dependent variable',
             xyLabelsFontSize=30,xyLabelsFontWeight='bold',x=np.linspace(0,10,10),
             xTicks = None, Y=[np.linspace(0,1,10),np.linspace(1,2,10)],
             labelList=['f1(x)','f2(x)'],lineWidth=1,
             vLineLabelsList=['event 1', 'event 2', 'event 3'], 
             vLineXList=[3.0, 6.0, 9.0], vls='--',  
             legendSize=30, savePlotPath=None, closePlot = False):
    """
    (figureName ='Example plot', figSize=(128, 64),xLabel='Independent variable', 
     yLabel='Dependent variable', xyLabelsFontSize=20, xyLabelsFontWeight='bold',
     x=np.linspace(0,10,10), xTicks = None, 
     Y=[np.linspace(0,1,10), np.linspace(1,2,10)],
     labelList=['f1(x)', 'f2(x)'], lineWidth=1, 
     vLineLabelsList=['event 1', 'event 2', 'event 3'], 
     vLineXList=[3.0, 6.0, 9.0], vls='--', legendSize=10, savePlotPath=None)

    Plot data 
    
    Parameters
    ----------
    figureName : TYPE str, optional
        DESCRIPTION. Set data figure name.
        The default is 'Example plot'.
    figSize : TYPE tuple, optional
        DESCRIPTION. Set data figure size.
        The default is (128, 64).
    xLabel : TYPE str, optional
        DESCRIPTION. Set X-axis label. 
        The default is 'Independent variable'.
    yLabel : TYPE str, optional
        DESCRIPTION. Set Y-axis label. 
        The default is 'Dependent variable'.
    xyLabelsFontSize : TYPE int, optional
        DESCRIPTION. Set x-y labels font size 
        The default is 20.
    xyLabelsFontWeight : TYPE str, optional
        DESCRIPTION. Set plot font weight
        The default is 'bold'.
    x : TYPE, numpy ndarray, optional
        DESCRIPTION. Line X-axis coords
        The default is np.linspace(0,1,10).
    xTicks : TYPE, list, optional
        DESCRIPTION. X-ticks list
        The default is None.    
    Y : TYPE, ndarray list, optional
        DESCRIPTION. List of multiple lines Y-axis coords
        The default is [np.linspace(0,1,10),np.linspace(1,2,10)].
    labelList : TYPE list of str, optional
        DESCRIPTION. List of multiple line labels
        The default is ['f1(x)','f2(x)'].
    lineWidth : TYPE int, optional
        DESCRIPTION. Lines width
        The default is 1.
    vLineLabelsList : TYPE list of str , optional
        DESCRIPTION. List of vertical lines labels 
        The default is ['event 1', 'event 2'].
    vLineXList : TYPE list of float, optional
        DESCRIPTION. Vertical lines X-axis coords 
        The default is [0.3,0.6].
    vls : TYPE str, optional
        DESCRIPTION. Vertical line style (same of matplotlib.pyplot)
        The default is '--'.
    legendSize : TYPE int, optional
        DESCRIPTION. Plot legend size
        The default is 10.
    savePlotPath : TYPE str, optional
        DESCRIPTION. plot save path
        The default is None.
    closePlot : TYPE bool, optional
        DESCRIPTION. automatically close simulation plot. 
        The default is True.
                

    Returns
    -------
    None.

    """
    
    plotFigure = plt.figure(figureName,figsize=figSize)
    plotAxes = plotFigure.add_subplot(111)
    plotAxes.set_xlabel(
        xLabel, 
        fontsize = xyLabelsFontSize, 
        fontweight = xyLabelsFontWeight
        )
    plotAxes.set_ylabel(
        yLabel, 
        fontsize = xyLabelsFontSize,
        fontweight = xyLabelsFontWeight
        ) 
    plotAxes.set_xlim(x[0],x[-1])
    if xTicks != None:
        plotAxes.set_xticks(xTicks)
    plt.xticks(fontsize=50)
    YMax = 0.0
    YMin = 0.0
    for n, y in enumerate(Y):
        plotAxes.plot(x,y,label=labelList[n],linewidth=lineWidth)
        if y.max() > YMax:
            YMax = y.max()
        if y.min() < YMin:
            YMin = y.min()
    
    plotAxes.set_ylim(0,5)  
    for n, line in enumerate(vLineLabelsList):
        plotAxes.axvline(
            x = vLineXList[n],
            ls=vls,
            label=vLineLabelsList[n],
            color= (1./(n+1),0,0)
            ) 

    plotAxes.legend(loc = 'best', fontsize = legendSize) 
    if savePlotPath != None:
        plotFigure.savefig(savePlotPath)
    if closePlot:
        plt.close()
            



def absolutePercentageError(target, x):
    """
    Return the absolute percentage error between each element of two vectors
    
    Parameters
    ----------
    target : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    ape = np.abs(((target-x) / target) * 100)
    # print(ape)
    
    return  ape
    

def changeRange(old_value, old_min, old_max, new_min, new_max):
    return (((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min    
    
    


if __name__ == '__main__':
    
    x = np.linspace(0,1, 100000)
    y1 = x**2 + x + 5
    y2 = x**4 - 3*x + 7
    mape = absolutePercentageError(y2,y1)
    
    
    plotData(x = x, Y=[y1,y2,mape], labelList=['y1','y2','mape'])        
    
