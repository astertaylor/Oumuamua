# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:41:13 2022

@author: astertaylor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quaternion
from scipy.interpolate import UnivariateSpline


def simple_curve(theta,a,b,c):
    dist=np.abs(np.sin(theta)*np.tan(theta)*a**2/b**2+np.cos(theta))
    beta=np.sqrt(1+a**2/b**2*np.tan(theta)**2)

    return(np.pi*c*dist/beta)


belton=pd.read_csv("BeltonLightcurves.csv")
indcut=50

belton=belton[:indcut]

beltime=belton['Time'].to_numpy()
belmag=belton['Magnitude'].to_numpy()
belsig=belton['msig'].to_numpy()

astro=pd.read_csv("2017-10-25_horizons_results.csv")

asttime=astro['Time'].to_numpy()
phase=astro['Phase'].to_numpy()

phase=UnivariateSpline(asttime,phase)

period = 7.937*3600  # seconds
alpha=phase(beltime)

times=np.linspace(np.min(beltime),np.max(beltime),1000)
thetadat=2*np.pi*(times/period)%period

plt.scatter(beltime,belmag,s=0.5,c='k')

data=2.5*np.log10(simple_curve(thetadat,19,115,111))
data+=15
plt.plot(times,data)

plt.savefig("test.png")