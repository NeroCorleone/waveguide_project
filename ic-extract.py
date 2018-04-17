import pandas as pd
import numpy as np
import matplotlib.pyplot as plt#
from scipy.signal import argrelmin, argrelmax
from scipy.interpolate import UnivariateSpline
import csv


headervalues = ['Vg_set', 'Vtg_set', 'B_set', 'Vd_set', 'Vg', 'Vtg', 
                'Vd', 'Id', 'Vac_real_raw', 'Iac_real_raw', 
                'Vac_imag_raw', 'Iac_imag_raw', 'Vac_R_raw', 
                'Iac_R_raw', 'Vac_p_raw', 'Iac_p_raw', 'Rac', 
                'Rac_inphase', 'G_inphase', 'timestamp'
               ]

data = pd.read_csv('data.dat', header=None, delimiter='\t')
data.columns = headervalues

bvalues = np.unique(data['B_set'].values)

data.drop(data[data['B_set'] == bvalues[100]].index[:301], inplace=True)

b_offset = bvalues[185]
i_offset = data[data['B_set'] == b_offset]['Id'].values
rac_offset = np.flip(data[data['B_set'] == b_offset]['Rac_inphase'].values, 0)

offset = np.ones(len(i_offset)) * np.abs(i_offset[argrelmin(rac_offset, order=50)[0][0]])

def cutoff(xindex):
    return int(-2.6052 * xindex + 285.921)

def find_peaks(spline, derivative, current, rightest_max=True):
    all_minima = argrelmin(np.abs(derivative(current)), order=10)[0]
    middle = np.argmin(
                        np.abs(
                                all_minima - zero_current_ix * np.ones(len(all_minima))
                               )
                       )#find minima closest to zero_current_ix
    
    #right max und left max bestimmen, indem zu allen minima rechts die werte berechnet werden 
        #und der index mit dem größte wert ist das rechte max 
    if rightest_max:
        right_max = all_minima[middle:][1]
    else:
        right_max = max(zip(all_minima[middle:], [spline(current[minval]) for minval in all_minima[middle:]]), 
                        key=lambda x: x[1])[0]
    return int(right_max)

def get_params(index):
    if index in range(0, 60) or index in range(140, 201):
        s, co, rightest_max, bound, threshold = 60, 100, True, 0, 0
    if index in range(60, 80) or index in range(120, 140):
        s, co, rightest_max, bound, threshold = 50, 80, False, 10, 15
    if index in range(80, 100):
        s, co, rightest_max, bound, threshold = 50, cutoff(index), True, 40, 15
    if index in range(100, 120):
        s, co, rightest_max, bound, threshold = 50, cutoff(index - 20), True, 40, 15
    
    return (s, co, rightest_max, bound, threshold)

def smoothen_middle(rac, bound, threshold):
    leftbound, rightbound  = zero_current_ix - bound, zero_current_ix + bound
    for rindex, rval in enumerate(rac[leftbound:rightbound]):
        if rac[leftbound + rindex] > threshold or rac[leftbound + rindex] < 0:
            rac[leftbound + rindex] = rac[leftbound + rindex - 1]
    return rac

result = []
#rac_max_values = []

rac_threshold = [data[data['B_set'] == bval]['Rac_inphase'].values[0] for bval in bvalues]

for bindex, bval in enumerate(bvalues):
    print(bindex)
    s, co, rightest_max, bound, threshold = get_params(bindex)
    idrain = data[data['B_set'] == bval]['Id'].values
    rac = np.flip(data[data['B_set'] == bval]['Rac_inphase'].values, 0)[co:-co]
    current = (idrain + offset)[co:-co]
    zero_current_ix = np.argmin(np.abs(current))
    
    threshold = rac_threshold[bindex]
    for rindex, rval in enumerate(rac):
        if rval > threshold or rval < 0:
            rac[rindex] = rac[rindex - 1]
    
    smoothen_middle(rac, bound=bound, threshold=threshold)
    
    spline = UnivariateSpline(current, rac, s=s)
    derivative = spline.derivative()
    
    try:
        right_peak = find_peaks(spline, derivative, current, rightest_max= rightest_max)
    except IndexError:
        result.append(0)
        continue
        
    rac_max = spline(current[right_peak]) * np.ones(len(current))
    rac_min = spline(current[zero_current_ix]) * np.ones(len(current))
    rac_half = (rac_max + rac_min) / 2

    try:
        rac_half_ix = zero_current_ix + argrelmin(np.abs(rac_half - spline(current))[zero_current_ix:], order=30)[0][-1]
    except IndexError:
        result.append(0)
        continue

    current_half = current[rac_half_ix]
    result.append(current_half)


colorplot = []

for bval in bvalues:
    rac = np.flip(data[data['B_set'] == bval]['Rac_inphase'].values, 0)
    colorplot.append(list(rac))

y, x= np.meshgrid(idrain, bvalues)
fig, ax = plt.subplots(figsize=(16, 9))
ax.pcolor(x, y, colorplot,  vmin=0.0, vmax=200)
ax.plot(bvalues, result, color='r', marker='o',)
fig.savefig('ic-extract.eps')
