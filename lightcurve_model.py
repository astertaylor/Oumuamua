# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:01:27 2022

@author: astertaylor
"""

import numpy as np
import pandas as pd
import quadpy
import quaternion


period = 7.937*3600  # seconds


def lightcurve(a, b, c, theta, rot=[1,1,0], sun=[1, 0, 0], obs=[0, 1, 0]):
    ux,uy,uz=rot

    sun = np.quaternion(0, *sun)
    obs = np.quaternion(0, *obs)

    q = np.quaternion(np.cos(theta/2), ux*np.sin(theta/2),
                      uy*np.sin(theta/2), uz*np.sin(theta/2))
    q = q/np.abs(q)

    sun = quaternion.as_float_array(np.conj(q)*sun*q)[1:]
    obs = quaternion.as_float_array(np.conj(q)*obs*q)[1:]

    C = np.zeros((3, 3))
    C[0, 0] = 1/a**2
    C[1, 1] = 1/b**2
    C[2, 2] = 1/c**2

    Sl = np.sqrt(np.dot(sun.T, np.dot(C, sun)))
    So = np.sqrt(np.dot(obs.T, np.dot(C, obs)))

    alpha = np.arccos((np.dot(sun.T, np.dot(C, obs))/(Sl*So)))

    S = np.sqrt(Sl**2+So**2+2*Sl*So*np.cos(alpha))

    cosl = (Sl+So*np.cos(alpha))/S
    sinl = (So*np.sin(alpha))/S

    if sinl < 0:
        lam = -np.arccos(cosl) % (2*np.pi)
    else:
        lam = np.arccos(cosl)

    return(1/8*a*b*c*Sl*So/S*
           (np.cos(lam-alpha)+np.cos(lam)+np.sin(lam)*np.sin(lam-alpha)*
            np.log(1/np.tan(lam/2)*np.cos((alpha-lam)/2))))

values=np.linspace(0,5,1000)
times=values*period

theta_list=(values%1)*2*np.pi

values=[]
for theta in theta_list:
    values.append(lightcurve(10,2,3,theta))

data=np.array([times,values]).T

outFrame = pd.DataFrame(data, columns=["Times","Lightcurve"])
outFrame.to_csv('test.csv')
