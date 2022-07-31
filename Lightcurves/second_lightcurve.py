# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:48:24 2022

@author: aster
"""

from matplotlib.gridspec import GridSpec
import numpy as np
import scipy
import quaternion
import matplotlib.pyplot as plt
from tqdm import tqdm


def fixed_lightcurve(theta, a, b, c, rot=[0, 0, 1], obs=[1, 0, 0], N=250):
    ndata = theta.size
    obs = np.quaternion(0, *obs)

    # Generate x,y points
    x = np.linspace(-a, a, N)
    y = np.linspace(-b, b, N)

    X, Y = np.meshgrid(x, y, indexing='ij')

    # Remove the points outside of bounds
    pts_in = np.where(X**2/a**2+Y**2/b**2 <= 1)
    X = X[pts_in]
    Y = Y[pts_in]

    # Generate z points for the x and y points
    z = c*np.sqrt(np.abs(1-X**2/a**2-Y**2/b**2))

    # Get +/- z points
    x = np.append(X, X)
    y = np.append(Y, Y)
    z = np.append(z, -z)

    npts = z.size

    # Combine points
    pts = np.array([x, y, z]).T
    pts = np.tile(pts[:, :, np.newaxis], ndata)

    nrot = np.outer(np.array(rot), np.sin(theta/2))
    nrot = np.append(np.cos(theta[np.newaxis, :]/2), nrot, axis=0)

    q = quaternion.as_quat_array(nrot.T)
    q *= 1/np.abs(q)

    obs = np.conj(q)*obs*q
    obs = quaternion.as_float_array(obs)[:, 1:]

    obs = np.repeat(obs.T[np.newaxis, :, :], npts, axis=0)

    inner = np.sum(pts*obs, axis=1)
    inner = np.repeat(inner[:, np.newaxis, :], 3, axis=1)

    proj = pts-inner*obs
    proj = np.append(np.zeros((npts, ndata))[:, np.newaxis, :], proj, axis=1)

    proj = np.moveaxis(proj, 1, 2)
    proj = quaternion.as_quat_array(proj)

    q = np.repeat(q[np.newaxis, :], npts, axis=0)

    proj = q*proj*np.conj(q)
    proj = quaternion.as_float_array(proj)
    proj = proj[:, :, 2:]

    output = []
    for i in tqdm(range(ndata)):
        hull = scipy.spatial.ConvexHull(proj[:, i, :])
        output.append(hull.volume)

    return(np.array(output))


def simple_curve(theta, a, b, c):
    dist = np.abs(np.sin(theta)*np.tan(theta)*(a**2)+(b**2)*np.cos(theta))
    beta = np.sqrt(b**2+a**2*(np.tan(theta))**2)

    return(np.pi*c*dist/beta)


theta_data = np.linspace(0, 2*np.pi, 500)

simple = (simple_curve(theta_data, 19, 115, 111))
light1 = (fixed_lightcurve(theta_data, 19, 115, 111, N=50))
light2 = (fixed_lightcurve(theta_data, 19, 115, 111, N=100))
light3 = (fixed_lightcurve(theta_data, 19, 115, 111, N=250))

COLOR = 'k'  # '#FFFAF1'
plt.rcParams['font.size'] = 20
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR

plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['xtick.major.size'] = 14  # 12
plt.rcParams['ytick.major.size'] = 14  # 12

plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.minor.size'] = 8

plt.rcParams

plt.rcParams['axes.linewidth'] = 3

plt.rcParams['text.color'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
#plt.rcParams['axes.spines.top'] = False
#plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['axes.edgecolor'] = COLOR
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['legend.facecolor'] = 'none'

plt.figure(figsize=(10, 10))
# plt.plot(theta_data,simple)
plt.plot(theta_data, np.abs(light1-simple)/simple)
plt.plot(theta_data, np.abs(light2-simple)/simple)
plt.plot(theta_data, np.abs(light3-simple)/simple)
# plt.plot(theta_data,light3)
plt.yscale('log')
plt.legend(["$N\\approx 4,000$", "$N\\approx 15,000$", "$N\\approx 100,000$"])
plt.xticks(np.pi*np.linspace(0, 2, 9),
           labels=[0, "$\\pi/4$", "$\\pi/2$", "$3\\pi/4$", "$\\pi$",
                   "$5\\pi/4$", "$3\\pi/2$", "$7\\pi/4$", "$2\\pi$"])
plt.xlabel("$\\theta$")
plt.ylabel("$\\log_{10}(\\frac{|L_1-L_0|}{L_0})$")
plt.savefig("num_lightcurve_comp.pdf", bbox_inches='tight',
            dpi=300)
