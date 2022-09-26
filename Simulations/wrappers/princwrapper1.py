# -*- coding: utf-8 -*-
"""
Created on Wed May 25 19:12:33 2022

@author: aster
"""

import SAMUS
from mpi4py import MPI
import time

start_time = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

rho = 0.5  # density in g/cm^3, assumed constant

size = 35/2  # SIZING, in METERS

period = 7.3975  # in hours

n = 0  # number of mesh refinements

# define the size scale
scaleb = 111/115
scalec = 19/115

# generate the classes
labels = ["pancake1", "pancake2", "pancake2", "pancake3"]
dims = [(scalec, scaleb, 1), (scaleb, scalec, 1),
        (scalec, scaleb, 1), (scaleb, scalec, 1)]
axes = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]]


for i, name in enumerate(labels):
    mu = 10**5  # initialize the momentum
    a, b, c = [size*x for x in dims[i]]  # get the principal axes
    omegavec = axes[i]  # get the rotation axis
    name = name+"_a"+str(size)  # get the name of the size
    STOP = False  # initialize
    while not STOP:  # while convergence not met
        # initalize model
        file = SAMUS.model(name, a, b, c, mu, omegavec, rho, n=n)

        # run model
        frame, div = file.run_model(10, data_name='oumuamua_traj',
                                    period=period, savesteps=False)
        mu *= 10  # update mesh

        MoIs = frame['MoIs'].to_numpy()
        STOP = (MoIs[-1]/MoIs[0] < 1.01) and not (div)  # find if deltaMoI<1.01
        print(MoIs[-1]/MoIs[0])

print("{:.3e}: --- FINISHED --- \n".format((time.time()-start_time)))
