# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:41:26 2022

@author: astertaylor
"""

import numpy as np
import pandas as pd

cutoff=4.7e13

datfile = open('horizons_results.txt', 'r', errors='replace')

# read in all lines
lines = datfile.readlines()

# close file
datfile.close()

# cut out all empty lines
lines = [f.strip('\n') for f in lines]

# create empty lists for running
nlines = []
times = []
dist = []

# loop over all lines
for dat in lines:
    # if lines begin with a number, read in, otherwise, ignore
    try:
        int(dat[0])
        nlines.append(dat)
    except:
        continue

# skip the first 3 line, since these are headers
nlines = nlines[3:]
for dat in nlines:
    # split up the line at spaces
    dat = dat.split()

    # time is the first number
    times.append(float(dat[0]))

    # distance is the 4th number
    dist.append(float(dat[3]))

times=86400*np.array([times])[0]
dist = 1.49e13*np.array([dist])[0]

# cut out distances/times greater than the cutoff
times = times[np.where(dist <= cutoff)]
dist = dist[np.where(dist <= cutoff)]

data=np.array([times,dist]).T

pd.DataFrame(data,columns=["Time","Distance"]).to_csv("oumuamua_traj.csv",float_format='%.16f')
