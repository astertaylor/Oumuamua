# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:56:54 2022

@author: Aster Taylor

This script provides a demonstration of how to add functions to the SAMUS class
which provide data outputs. All functions must either take no inputs or take
'self', and use class variables. This script provides examples of both, and
uses multiple data types.

"""
from SAMUS import *

#defines function which doesn't use class variables
def func():
    return([1,"test"],["ones","string"])

#defines function which uses class variables
def get_viscosity(self):
    return(float(self.mu),"viscosity")

#create class
mod_class=SAMUS("modularity_test",mu=10**4)

#runs simulation for a brief period with only 1 time step per rotation
frame=mod_class.run_model(1,data_name='example_traj.txt',
                          out_funcs=['moment_of_inertia','princ_axes',
                                     func,get_viscosity])

#prints DataFrame as demonstration
print("Test Frame:")
print(frame)
