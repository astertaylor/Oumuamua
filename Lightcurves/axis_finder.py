# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:01:27 2022

@author: astertaylor
"""

import numpy as np
import pandas as pd
import quadpy
import quaternion
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt


def sqrtpdf(x,args):
    a=args
    x=np.array([x])
    y=1/np.sqrt(x)
    y[np.where(x<(1/a))]=0
    y[np.where(x>a)]=0
    return(y)


def randDraw(pdf,args,a=100,b=None,dn=1):
    '''
    randomly draws from the distribution pdf
    
    pdf- function object which returns the probability of drawing at x. Note that it need not be normalized.
    args- arguments for the pdf function
    a- lower limit of the points to sample for the inversion. it is assumed that the PDF is essentially 0 below this value
    b- upper limit of the points to sample for the inversion. it is assumed that the PDF is essentially 0 above this value.
        if None, is taken to be the opposite sign of a
    dn- spacing of sampling points. 
    
    '''
    if b is None: b=a;a=-b
    assert(a<b)
    from scipy.integrate import quad
    from scipy.interpolate import interp1d
    import warnings
    import scipy
    warnings.simplefilter('ignore',scipy.integrate.IntegrationWarning) #ignores the rankwarnings from polyfit
    xsamp=np.arange(a,b,dn)
    A=1/quad(pdf,-np.infty,np.infty,args)[0]
    CDF=[]
    for x in xsamp:
        CDF.append(A*quad(pdf,-np.infty,x,args)[0])
    inverse=interp1d(CDF,xsamp)
    return(GenRand(inverse))
    


class GenRand:
    import numpy as np
    def __init__(self, inv):
        self.inv=inv
        
    def __call__(self, size):
        x=np.random.uniform(size=int(size))
        return(self.inv(x))


def gr_indicator(chain, gr_threshold=1.01):
    """

    GR convergence indicator for the nwalkers parallel MCMC chains



    Parameters:

    -----------

    chain: NumPy array of float numbers of shape (nsteps, nwalkers, ndim)

        MCMC sample values for ndim parameters in nwalkers chains after nsteps iterations

    gr_threshold: float (default 1.01)

        chains are considered converged if R_GR < gr_threshold



    Returns:

    --------

    converged: boolean, True if chain has converged (R_GR < gr_threshold), False otherwise

    """

    chain = np.array(chain)
    (nsamp, nchain, ndim) = chain.shape
    sw = np.mean(np.var(chain, axis=0), axis=0)
    chainmean = np.mean(chain, axis=0)
    sv = (nsamp-1)/nsamp * sw + np.var(chainmean, axis=0)
    RGR = sv/sw
    return(np.max(RGR) < gr_threshold)


def MCMC(logfunc, x0, args=None, nwalk=10, nminsteps=10**3, nmaxsteps=10**5, 
         nconv=20, step=1, convergence_func=None, conv_args=None):
    from numpy import matlib

    x0 = np.array(x0)
    dims = np.size(x0)
    x = np.matlib.repmat(np.array(x0), nwalk, 1) + \
        np.random.normal(0, 0.01, (nwalk, dims))
    fnow = logfunc(x, args)
    chain = []
    nsteps = 0
    converged = False
    print_val = True
    sqrt_dist = randDraw(sqrtpdf, 2, a=0, b=2.01, dn=0.01)

    while not converged or (nsteps < nminsteps):
        rand = np.random.randint(1, nwalk, size=nwalk)

        j = (np.arange(nwalk)+rand) % nwalk
        xj = x[j]
        z = step*sqrt_dist(nwalk)
        xtry = xj+np.multiply(np.matlib.repmat(z, dims, 1).T, (x-xj))
        ftry = logfunc(xtry, args)
        logp = (dims-1)*np.log(z)+ftry-fnow
        prob = np.matlib.repmat(logp, dims, 1).T
        rand = np.matlib.repmat(
            np.log(np.random.uniform(size=nwalk)), dims, 1).T
        x = np.where(rand <= prob, xtry, x)
        chain.append(x)
        fnow = logfunc(x, args)
        nsteps += 1

        if nsteps % nconv == 0:
            converged = convergence_func(chain, conv_args)

        if converged and print_val:
            print("Converged at N = %s" % nsteps)
            print_val = False

        if nsteps > nmaxsteps:
            break
        
        print(nsteps)
    return(np.reshape(chain, [-1, dims]))

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
    
    lnL=-0.5*(np.sum((light-data)**2/var)+2*np.pi*np.sum(var))
    
    if theta<0 or theta>=2*np.pi:
        lnL+=-1e15
    if phi<0 or phi>np.pi/2:
        lnL+=-1e15
    if psi<0 or psi>=2*np.pi:
        lnL+=-1e15

    return(lnL)

def MCMCwrapper(x,args):
    a, b, c, alpha, beta, data, var = args
    outputs=np.zeros(x.shape[0])

    for i,vec in enumerate(x):
        theta=vec[0];phi=vec[1];psi=vec[2];

        outputs[i]=lnL(theta,phi,psi,a,b,c,alpha,beta,data,var)

    return(outputs)

belton=pd.read_csv("BeltonLightcurves.csv")
indcut=335

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
beta=2*np.pi*(beltime/period)%period

args=(115,111,19,alpha,beta,belmag,np.square(belsig))

outputs=MCMC(MCMCwrapper,x0=[np.pi,np.pi/2,np.pi],args=args,
             convergence_func=gr_indicator,conv_args=1.01,nminsteps=100,nmaxsteps=1000)
pd.DataFrame(outputs,columns=["Theta","Phi","Psi"]).to_csv("MCMCOutput.csv")
