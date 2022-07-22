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

    sun=quaternion.as_float_array(np.conj(q)*sun*q)[1:3]
    obs=quaternion.as_float_array(np.conj(q)*obs*q)[1:3]

    def normal(x,a,b,c):
        norm=1/np.sqrt(x[0]**2/a**4 + x[1]**2/b**4 + x[2]**2/c**4)
        vector=x/[a**2,b**2,c**2]
        return(norm*vector)

    def luminance(x,a,b,c,sun,obs):
        x=x*[a,b,c]
        norm=normal(x,a,b,c)

        return(np.inner(norm,sun)*np.inner(norm,obs))

    scheme=quadpy.u3.get_good_scheme(5)

    scheme.integrate(lambda x: luminance(x,a,b,c,sun,obs),)
