# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:01:27 2022

@author: astertaylor
"""

import numpy as np
import pandas as pd
import quadpy
import quaternion


period=7.937*3600 #seconds

def lightcurve(a,b,c,ux,uy,uz,theta,sun=[1,0,0],obs=[0,1,0]):
    sun=np.quaternion(0,*sun)
    obs=np.quaternion(0,*obs)

    q=np.quaternion(np.cos(theta/2),ux*np.sin(theta/2),
                    uy*np.sin(theta/2),uz*np.sin(theta/2))

    sun=(np.conj(q)*sun*q)[1:3]
    obs=(np.conj(q)*obs*q)[1:3]

    def luminance(x,a,b,c,sun,obs):
