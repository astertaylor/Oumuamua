# -*- coding: utf-8 -*-
"""
Created on Wed May 25 19:12:33 2022

@author: aster
"""

import numpy as np
from SAMUS import *
from mpi4py import MPI
import time

start_time=time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

rho=0.5 #density in g/cm^3, assumed constant. in Oumuamua, estimates are 1500-2800

size=55 #SIZING, in METERS

period=7.937 # in hours

n=0 #number of mesh refinements

scaleb=111/115;scalec=19/115

labels=["pancake1","pancake2","pancake3","pancake4"]
dims=[(1,scaleb,scalec),(scalec,1,scaleb),(1,scaleb,scalec),(1,scalec,scaleb)]
axes=[[0,0,1],[0,0,1],[0,1,0],[0,1,0]]


for i,name in enumerate(labels):
    mu=10**5
    a,b,c=[size*x for x in dims[i]]
    omegavec=axes[i]
    name=name+"_a"+str(size)
    STOP=False   
    while not STOP:
        file=SAMUS(name,a,b,c,mu,omegavec,rho,n=n)
        STOP=file.run_model(10)
        mu*=10

print("%s: --- FINISHED --- \n"%((time.time()-start_time)))
