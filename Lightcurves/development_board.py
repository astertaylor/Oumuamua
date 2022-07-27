# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 11:01:27 2022

@author: astertaylor
"""
import numpy as np
import pandas as pd
import quadpy
import quaternion
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

def SSE(x, y):
    return(np.sum(np.square(x-y)))

def MINwrapper(x, theta, phi, psi, a, b, c, alpha, beta, data):
    betainit, delV = x[0], x[1]

    light = lightcurve(theta, phi, psi, betainit,
                       delV, a, b, c, alpha, beta)


    return(SSE(light,data))


def lightcurve(theta, phi, psi, betainit, delV, a, b, c, alpha, beta):
    beta += betainit

    rot = np.array([np.cos(beta/2), np.sin(beta/2)*np.cos(phi), np.sin(beta/2)
                   * np.sin(phi) * np.cos(psi), np.sin(beta/2)*np.sin(phi)*np.sin(psi)]).T
    rot = quaternion.as_quat_array(rot)

    obs = np.array([np.cos(alpha), np.sin(alpha) *
                   np.cos(theta), np.sin(alpha)*np.sin(theta)]).T

    obs = np.append(np.zeros(obs.shape[0])[:, np.newaxis], obs, axis=1)
    obs = quaternion.as_quat_array(obs)

    obs = rot*obs*np.conj(rot)

    obs = quaternion.as_float_array(obs)[:, 1:]

    sun = np.quaternion(0, 1, 0, 0)
    sun = rot*sun*np.conj(rot)
    sun = quaternion.as_float_array(sun)[:, 1:]

    Ss = np.sqrt(np.sum([1/a**2, 1/b**2, 1/c**2]*np.square(sun), axis=1))
    So = np.sqrt(np.sum([1/a**2, 1/b**2, 1/c**2]*np.square(obs), axis=1))

    palpha = np.arccos(
        np.sum([1/a**2, 1/b**2, 1/c**2]*sun*obs, axis=1)/(Ss*So))

    S = np.sqrt(Ss**2+So**2 + 2*Ss*So*np.cos(palpha))

    cosl = (Ss+So*np.cos(palpha))/S
    sinl = (So*np.sin(palpha))/S

    lam = np.where(sinl < 0, (-np.arccos(cosl) % (2*np.pi)), (np.arccos(cosl)))

    H = delV-2.5*np.log10(a*b*c*Ss*So/S*(np.cos(lam-palpha)+np.cos(lam)+np.sin(lam)
                                         * np.sin(lam-palpha)*np.log(1/np.tan(lam/2) *
                                                                     1/np.tan((palpha-lam)/2))))
    return(H)


def lnL(theta,phi,psi,a,b,c,alpha,beta,data,var):
    from scipy.optimize import minimize
    args=(theta,phi,psi,a,b,c,alpha,beta,data)

    opt=minimize(MINwrapper,np.array([np.pi/2,30]),args=(theta,phi,psi,a,b,c,alpha,beta,data),bounds=[(0,np.pi),(20,40)]).x

    light=lightcurve(theta, phi, psi, opt[0], opt[1], a, b, c, alpha, beta)

    return(-0.5*(np.sum((light-data)**2/var)+2*np.pi*np.sum(var)))

def MCMCwrapper(x,args):
    a, b, c, alpha, beta, data, var = args
    outputs=np.zeros(x.shape[0])

    for i,vec in enumerate(x):
        theta=vec[0];phi=vec[1];psi=vec[2];

        outputs[i]=lnL(theta,phi,psi,a,b,c,alpha,beta,data,var)

    return(outputs)

period=7.397*3600

belton=pd.read_csv("BeltonLightcurves.csv")
astro=pd.read_csv("2017-10-25_horizons_results.csv")

alpha=UnivariateSpline(astro['Time'],astro['Phase'])
times=belton['Time']
data=belton['Magnitude']

alpha=alpha(times)

beta=2*np.pi*((times/period)%period)

args=115,115,19,alpha,beta,data,np.ones_like(data)
x=np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
print(MCMCwrapper(x,args))
