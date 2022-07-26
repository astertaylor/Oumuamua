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


def sqrtpdf(x, args):

    a = args

    x = np.array([x])

    y = 1/np.sqrt(x)

    y[np.where(x < (1/a))] = 0

    y[np.where(x > a)] = 0

    return(y)


def randDraw(pdf, args, a=100, b=None, dn=1):
    '''

    randomly draws from the distribution pdf



    pdf- function object which returns the probability of drawing at x. Note that it need not be normalized.

    args- arguments for the pdf function

    a- lower limit of the points to sample for the inversion. it is assumed that the PDF is essentially 0 below this value

    b- upper limit of the points to sample for the inversion. it is assumed that the PDF is essentially 0 above this value.

        if None, is taken to be the opposite sign of a

    dn- spacing of sampling points.



    '''

    if b is None:
        b = a
        a = -b

    assert(a < b)

    from scipy.integrate import quad

    from scipy.interpolate import interp1d

    import warnings

    import scipy

    # ignores the rankwarnings from polyfit
    warnings.simplefilter('ignore', scipy.integrate.IntegrationWarning)

    xsamp = np.arange(a, b, dn)

    A = 1/quad(pdf, -np.infty, np.infty, args)[0]

    CDF = []

    for x in xsamp:

        CDF.append(A*quad(pdf, -np.infty, x, args)[0])

    inverse = interp1d(CDF, xsamp)

    return(GenRand(inverse))


class GenRand:
    import numpy as np

    def __init__(self, inv):

        self.inv = inv

    def __call__(self, size):

        x = np.random.uniform(size=int(size))

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


def MCMC(x0, nwalk=10, nminsteps=10**6, nconv=20, step=1, logfunc=None, args=None, convergence_func=None, conv_args=None):

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

        if nsteps > 10000:
            break

    return(np.reshape(chain, [-1, dims]))


def lightcurve(a, b, c, theta, rot=[1, 1, 0], sun=[1, 0, 0], obs=[0, 1, 0]):
    ux, uy, uz = rot

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

    return(a*b*c*Sl*So/S *
           (np.cos(lam-alpha)+np.cos(lam)+np.sin(lam)*np.sin(lam-alpha) *
            np.log(1/np.tan(lam/2)*np.cos((alpha-lam)/2))))


belton=pd.read_csv("BeltonLightcurves.csv")
indcut=335

belton=belton[:indcut]

beltime=belton['Time'].to_numpy()
belmag=belton['Magnitude'].to_numpy()
belsig=belton['msig'].to_numpy()

belmagnorm=belmag-np.mean(belmag)
belmagnorm=belmagnorm/np.max(np.abs(belmagnorm))

astro=pd.read_csv("2017-10-25_horizons_results.csv")

asttime=astro['Time'].to_numpy()
sundist=astro['SunDist'].to_numpy()
earthdist=astro['EarthDist'].to_numpy()
phase=astro['Phase'].to_numpy()

times=np.linspace(asttime[0],asttime[-1],1000)
sundist=UnivariateSpline(asttime,sundist)
earthdist=UnivariateSpline(asttime,earthdist)
phase=UnivariateSpline(asttime,phase)

period = 7.937*3600  # seconds
curve=[]
for t in times:
    theta=2*np.pi*(t%period)/period

    alpha=phase(t)
    light=lightcurve(115,111,19,theta,obs=[np.cos(alpha),np.sin(alpha),0])
    #light*=2e30/(sundist(t)*earthdist(t))**2
    curve.append(-2.5*np.log10(light))

curvenorm=curve-np.mean(curve)
curvenorm=curvenorm/np.max(np.abs(curvenorm))

plt.figure(figsize=(15,10))
plt.scatter(beltime,belmagnorm)
plt.plot(times,curvenorm)
plt.savefig("test.png")
