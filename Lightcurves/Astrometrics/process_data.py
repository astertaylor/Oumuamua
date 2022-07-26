# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 17:23:44 2022

@author: astertaylor
"""

import numpy as np
import pandas as pd
import glob

for file in glob.glob('*horizons_results.txt'):

    datfile = open(file, 'r', errors='replace')

    # read in all lines
    lines = datfile.readlines()

    # close file
    datfile.close()

    # cut out all empty lines
    lines = [f.strip('\n') for f in lines]

    # create empty lists for running
    nlines = []

    times = []
    sundist = []
    earthdist=[]
    phase=[]

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
        times.append(float(dat[2]))

        sundist.append(float(dat[3]))

        earthdist.append(float(dat[5]))

        phase.append(float(dat[7]))

    times=86400*np.array([times])[0]
    sundist = 1.49e13*np.array([sundist])[0]
    earthdist=1.49e13*np.array([earthdist])[0]
    phase=np.pi/180*np.array([phase])[0]


    data=np.array([times,sundist,earthdist,phase]).T

    pd.DataFrame(data,columns=["Time","SunDist","EarthDist","Phase"]).to_csv(file[:-4]+".csv",float_format='%.16f')
