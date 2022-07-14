"""
Created on Wed May 25 13:57:01 2022

@author: aster
"""
from fenics import *
import numpy as np
import time
import pandas as pd
from scipy.interpolate import UnivariateSpline
from mpi4py import MPI
import quaternion

class SAMUS:
    '''
    Creates a model of an ellipsoidal asteroid and simulates its evolution 
    over the given trajectory.
    
    Attributes
    ----------        
    name : str
        The name of the run. The mesh output file will be name.pvd, the log 
        name.txt, and the moment of inertia file MoIs_name.csv.
    sizecut : float, [cm]
        The maximum allowable body radius. If the body crosses this threshold, 
        the run stops. Set as a ratio of the maximum initial body radius at 
        initialization.
    
    a : float, [m]
        The diameter of the ellipsoid on the x-axis. Twice the semi-axis.
    b : float, [m]
        The diameter of the ellipsoid on the y-axis. Twice the semi-axis.
    c : float, [m]
        The diameter of the ellipsoid on the z-axis. Twice the semi-axis.
    omegavec : list
        A list of length 3, containing the values of the rotation axis. The 
        rotation axis is the unit vector parallel to omegavec.
    mu : float, [dyne cm^-2 s]
        The dynamic viscosity of the body in this model. 
    rho : float, [g cm^-3]
        The density of the body in this model.
    omega : dolfin.function.function.Function, [s^-1]
        A function over the function space V, storing the rotation vector of 
        the asteroid body. Constant over the domain. Stored in a function so 
        that vector math can be performed using FEniCS methods. 
    period : float, [s]
        The rotational period of the asteroid, in seconds.
    
    G : dolfin.function.constant.Constant, [cm^3 g^-1 s^-2]
        A Constant equal to 6.67e-8, the gravitational constant in cgs. 
    M : dolfin.function.constant.Constant, [g]
        A Constant equal to 2e33, the solar mass in grams, used to compute the
        value of the tidal force. 
        
    t : float, [s]
        The current time in the model simulation.
    dt : float, [s]
        The Euler time step of the model simulation.
    ind : integer
        The number of cycles currently run by the model simulation.
    start_time : float, [s]
        The time when the model was initialized, in seconds in UTC.
    end_time :  float, [s]
        The maximum time in the simulation trajectory. The cutoff point of the
        simulation.
    mint : float, [s]
        The initial timestamp of the trajectory data. This is used so that the
        time can be expressed in a percent, but that the output of the 
        simulation can be expressed with time measured from the same origin as
        its input. 
    
    logfile : _io.TextIOWrapper
        File where the run logs are written, '.txt' text file. 
    outfile : dolfin.cpp.io.File
        File where the model mesh and functions are saved, '.pvd' ParaView file.
    
    mesh : dolfin.cpp.mesh.Mesh
        The mesh of the body model, the domain over which all the functions 
        are defined.
        
    V : dolfin.function.functionspace.FunctionSpace
        A vector function space over the mesh, for storing vector functions
        (velocity, forces). Of Continuous Galerkin type, order 2. 
    Q : dolfin.function.functionspace.FunctionSpace
        A scalar function space over the mesh, for storing scalar functions
        (pressure, distance). Of Continuous Galerkin type, order 1. 
    Z : dolfin.function.functionspace.FunctionSpace
        A mixed function space over the mesh, formed from V x Q. Mixed 
        functions are necessary to solve the Navier-Stokes equations, and the 
        solutions are stored in this space.
        
    szscale : float
        The allowable increase in the body size. If any axis is larger than
        szscale times the largest semi-major axis, the simulation stops. 
    solver : dolfin.cpp.fem.NonlinearVariationalSolver
        Nonlinear solver for the Navier-Stokes equation. This is used to 
        compute the solutions to the equations. Various parameters can be 
        changed by users via the built-in methods for the solvers. As a 
        default, this uses a Newton nonlinear method, with a relaxation para-
        meter of 1.0. 
    
    up : dolfin.function.function.Function, [3*(cm s^-1), dyne cm^-2]
        The function for the velocity and the pressure, over the mixed space Z.
        This is where the Navier-Stokes solutions are stored. Equivalent to 
        u x p, as an outer product. 
    u_p_ : dolfin.function.function.Function, [3*(cm s^-1), dyne cm^-2]
        A function over the mixed space Z, which stores the previous solution
        to the Navier-Stokes equations. This is used by the Euler 
        finite-difference solver, which requires the previous solution. 
    u : dolfin.function.function.Function, [cm s^-1]
        A vector function over V, storing the velocity of the model body 
        relative to its center of mass. Used for computing the distortion of 
        the model body, an output of the Navier-Stokes equations.
    p : dolfin.function.function.Function, [dyne cm^-2]
        A scalar function over Q, storing the pressure of the model body. An 
        output of the Navier-Stokes equations.   
    
    ftides : dolfin.function.function.Function, [dyne cm^-3]
        A vector function over V, storing the solar tidal force per unit 
        volume at the current time.
    gravity : dolfin.function.function.Function, [dyne cm^-3]
        A vector function over V, storing the gravitational force per unit 
        volume due to the asteroid body's own mass.
    centrifugal : dolfin.function.function.Function, [dyne cm^-3]
        A vector function over V, storing the centrifugal force per unit 
        volume due to the rotation of the asteroid.
    coriolis : dolfin.function.function.Function, [dyne cm^-3]
        A vector function over V, storing the Coriolis force per unit volume
        due to the model's velocity. Computed using the previous-step's 
        velocity.
    forcing : dolfin.function.function.Function, [dyne cm^-3]
        A vector function over V, storing the sum of the various forces, used 
        in the Navier-Stokes equations. 
        
    gravsolver : dolfin.cpp.fem.NonlinearVariationalSolver
        Nonlinear solver for the Gaussian gravity formulation. This is used
        to compute the solutions to the equations. Various parameters can be 
        changed by users via the built-in methods for the solvers. As a 
        default, this uses a Newton nonlinear method, with a relaxation para-
        meter of 1.0. 
    
    gravgs : dolfin.function.function.Function, [3*(dyne cm^-3), N/A]
        A mixed function over Z, storing the the gravitational force over V 
        and a superfluous function over Q, which is used to allow for a scalar 
        test function. The gravity is computed by solving the Gaussian form.
    gravscale : float
        A scalar constant of 1e-3, created to rescale the gravitational 
        constant G to the SI values. This is necessary to maintain stability 
        in the solutions to the Gaussian form, and then is used to rescale the 
        gravity back to cgs, due to the linearity of the Gaussian form. 
    
    trajectory : scipy.interpolate.fitpack2.UnivariateSpline, [cm]
        A spline fit to the trajectory the simulation is run over. When fed a 
        time, returns the heliocentric distance in cm. 
        
    r : dolfin.function.function.Function, [cm]
        A vector function over V, storing the position vectors of each point.
        This is simply the vector from the origin/center of mass to each point
        in the domain.
    
    umean : dolfin.function.function.Function, [cm s^-1]
        A vector function over V, storing the mean velocity of a certain time
        step, used to normalize the velocity. 
    usum : dolfin.function.function.Function, [cm s^-1]
        A vector function over V, the sum of the velocities over rotation 
        cycles.     
    ucycavg : dolfin.function.function.Function, [cm s^-1]
        A vector function over V, storing the mean velocity over a certain 
        number of rotations. 

    unit : dolfin.function.function.Function
        A scalar function over Q, which is equal to unity at all points. Used
        to perform integrals over the domain, to find the volume of the body.
    
    diverged : bool
        A boolean storing whether or not the simulation has diverged, either 
        due to a failure in the solver or due to the model body crossing the 
        imposed size threshold. 
    savesteps : bool
        Whether or not every timestep should be written to the output file, or
        if it should be just the starting and ending states. If True, writes 
        all timesteps.
    
    nsrot : int
        The number of timesteps in each rotation of the body, used to calculate
        dt. 
    rtol : float
        The tolerance in the trajectory for the trajectory jump. After a 
        certain number of cycles, the simulation takes the average velocity 
        and extrapolates the motion of the body forward a longer time. It stops
        the extrapolation either when the CFL criterion is >1 or when the 
        heliocentric distance changes by a factor of >rtol.
        
    times : list, [s]
        A list of the time values used in the simulation.
    MoIs : list, [g cm^2]
        A list of the moments of inertia over the simulation.
    
    '''
    
    def __init__(self,name,a=115,b=111,c=19,mu=10**7,omegavec=[0,0,1],rho=0.5,
                 szscale=2,n=0):   
        '''
        Sets up the model, creating initial parameters, setting up the mesh, 
        setting up functions, and preparing the solver equations for gravity 
        and Navier-Stokes. 

        Parameters
        ----------
        name : str
            The name of the run, for the output file.
        a : float, optional
            The diameter of the ellipsoid on the x-axis. Twice the semi-axis.
            The default is 115 meters.
        b : float, optional
            The diameter of the ellipsoid on the y-axis. Twice the semi-axis.
            The default is 111 meters.
        c : float, optional
            The diameter of the ellipsoid on the z-axis. Twice the semi-axis.
            The default is 19 meters.
        mu : float, optional
            The viscosity of the body. The default is 10**7 g cm^-1 s^-2.
        omegavec : list, optional
            The direction of the rotation axis, passing through the c.o.m. of 
            the mass. Does not have to be magnitude 1. The default is [0,0,1].
        rho : float, optional
            The density of the body. The default is 0.5 g cm^-3.
        szscale : float, optional
            The allowable increase in the body size. If any axis is larger than
            szscale times the largest semi-major axis, the simulation stops. 
            The default is 2.
        n : int, optional
            How many refinements should be done to the mesh. The refined meshes
            must already be made. The default is 0.

        Returns
        -------
        None.

        '''
        assert(len(omegavec)==3)
        assert(szscale>=1)
        assert(n>=0)
        self.name=name #setting the name
        self.omegavec=omegavec #setting the rotation axis
        self.a=a;self.b=b;self.c=c #setting the principal axes
        self.szscale=szscale #setting the size scale
        
        a*=100;b*=100;c*=100 #converting the axes from meters to cm
        self.sizecut=szscale*np.max([a,b,c])/2 #setting the maximum allowed size
        
        self.mu=Constant(mu) #creating a Constant to avoid slowdowns
        self.t=0;self.ind=0 #initializing the time, and the number of cycles
        self.dt=Constant(1) #set dt to 1 temporarily, for use in the solvers
        self.rho=Constant(rho) #creates a Constant to avoid slowdowns
        
        self.start_time=time.time() #sets the inital time, for diffs
    
        self.logfile=open('logs/{}_{}.txt'.format(self.name,int(np.log10(mu))),
                          'w') #creates log files
        self.logfile.write("%s: Beginning Initialization... \n"% #writes initial time to log file
                           (self.convert_time(time.time()-self.start_time))) 
        
        self.mesh=Mesh('Meshes/3ball%s.xml'%(n)) #read in mesh, with n refinements
        
        #rescales the mesh to the input ellipsoids
        self.mesh.coordinates()[:,0]*=a/2 
        self.mesh.coordinates()[:,1]*=b/2
        self.mesh.coordinates()[:,2]*=c/2
        
        self.set_spline() #computes the spline for the trajectories
        
        self.outfile=File("results/{}_{}.pvd".format(self.name, #sets output file
                                                     int(np.log10(mu))))
        
        #uses Elements to make a mixed function space
        V=VectorElement("CG", self.mesh.ufl_cell(), 2)
        Q=FiniteElement("CG", self.mesh.ufl_cell(), 1)
        self.Z = FunctionSpace(self.mesh, V*Q)
        
        #creates actual function spaces which compose the mixed
        self.V=VectorFunctionSpace(self.mesh,"CG",2)
        self.Q=FunctionSpace(self.mesh,"CG",1)
        
        #creates solution functions from the mixed space
        self.up = Function(self.Z) #solution function
        self.u_p_=Function(self.Z) #function for previous solutions
        
        #gets trial and test functions from the mixed space 
        dup=TrialFunction(self.Z)
        v, q = TestFunctions(self.Z)
        
        #creates the function of the rotation vector
        self.omega=interpolate(Constant(tuple(omegavec)),self.V)
        
        #splits the solution functions
        self.u, self.p = split(self.up)
        u_,p_= split(self.u_p_)
        
        #sets solution functions to 0
        self.up.assign(Constant((0,0,0,0)))
        self.u_p_.assign(Constant((0,0,0,0)))
        
        #creates the functions for storing the forces
        self.ftides=Function(self.V) #tides
        self.gravity=Function(self.V) #gravity
        self.centrifugal=Function(self.V) #centrifugal
        self.coriolis=Function(self.V) #coriolis
        self.forcing=Function(self.V) #total forces
        
        #names the functions for storage
        self.ftides.rename("Tidal Force","Tidal Force")
        self.gravity.rename("Self-Gravity","Gravitational Force")
        self.centrifugal.rename("Centrifugal","Centrifugal Force")
        self.coriolis.rename("Coriolis","Coriolis Force")
        self.forcing.rename("Forcing","Total force on the object")

        A=Constant(1e4/max(mu,1e4)) #creates a constant to ensure solution stability
        
        #creates the solution for the Navier-Stokes equations
        F=(A*self.rho*inner(((self.u-u_)/(self.dt)),v) * dx + #acceleration term
                A*self.mu*inner(grad(self.u),grad(v)) * dx + #viscosity term
                A*self.rho*inner(dot(self.u,nabla_grad(self.u)),v) * dx - #advection term
                A*self.p*div(v) * dx + #pressure term
                q*div(self.u) * dx - #mass continuity equation
                A*inner(self.forcing,v) * dx) #force term
        
        J=derivative(F,self.up,dup) #find the derivative, for speed
        
        #sets up the Navier-Stokes solver
        problem=NonlinearVariationalProblem(F,self.up,J=J)
        self.solver=NonlinearVariationalSolver(problem)
        self.solver.parameters['newton_solver']['relaxation_parameter'] = 1.
        
        #splits solution functions for access (weird FEniCS quirk)
        self.u,self.p=self.up.split()
        u_,p_=self.u_p_.split()
        
        #names the solution functions
        self.u.rename("Velocity","Velocity")
        self.p.rename("Pressure","Pressure")
        
        #COMPUTE FUNCTIONS FOR GRAVITY SOLUTIONS
        self.G=Constant(6.674e-8) #sets gravitational constant, in cgs
        
        #gets solution, trial, and test functions
        self.gravgs = Function(self.Z)
        dgs=TrialFunction(self.Z)
        gravh, gravc = TestFunctions(self.Z)
        gravg, gravs = split(self.gravgs)
        
        #sets a scale to ensure the stability of the solution. this is undone 
        #in the solution, but for unknown reasons O(10^-8) is too large for 
        #the solver to maintain stability
        self.gravscale=1e-3
        
        #computes the scaling constant for the Gaussian gravity form, which is 
        #rescaled by self.gravscale. A Constant, for speed
        gravA=Constant(4*np.pi*float(self.G)*float(self.rho)*self.gravscale)
    
        #creates the equation set for Gaussian gravity
        gravF=(
            #this equation is 0=0, it is here to allow me to mix vector and 
            #scalar solutions
            gravs*div(gravh) * dx + inner(gravg,gravh) * dx + 
            #this equation is the Gaussian form, div(g)=-4 pi G rho. only 
            #relevant equation here. 
            gravc*div(gravg) * dx + gravA*gravc * dx)
                
        gravJ=derivative(gravF,self.gravgs,dgs) #find the derivative, for speed
  
        #sets up the gravitational solver
        gravproblem=NonlinearVariationalProblem(gravF,self.gravgs,J=gravJ)
        self.gravsolver=NonlinearVariationalSolver(gravproblem)
        self.gravsolver.parameters['newton_solver']['relaxation_parameter'] = 1.
        
        self.logfile.write("%s: Initializations Complete \n"% #writes to log
                           (self.convert_time(time.time()-self.start_time)))

    def set_spline(self,data_name='horizons_results.txt'):
        '''
        Computes a spline estimation for the solar distance versus time, for 
        use in computing the distance at non-measured times.

        Parameters
        ----------
        data_name : str, optional
            The name of the file containing the data. The default is
            'horizons_results.txt'.

        Returns
        -------
        None.

        '''
        times,dist=self.read_data(data_name) #reads in data
        self.end_time=times[-1] #finds the ending time for cutoffs
        
        self.trajectory=UnivariateSpline(times,dist) #creates spline
    
    def read_data(self,data_name='horizons_results.txt',cutoff=4.7e13):
        '''
        Reads and returns New Horizons data for time and solar distance. This 
        function is assumed to be reading in New Horizons data, with times in 
        days and distances in AU. 

        Parameters
        ----------
        data_name : str, optional
            The name of the file containing the data. The default is 
            'horizons_results.txt'.
        cutoff : float, optional
            The maximum heliocentric distance we want to consider. Any distance
            beyond this will be cut out of the data. The default is chosen such 
            that the tidal force is always greater than 1e-2. 
            The default is 4.7e13 cm.

        Returns
        -------
        times : numpy.ndarray
            Array of the time difference from the data.
        dist : numpy.ndarray
            Array of the heliocentric difference, less than the cutoff. 

        '''

        datfile=open(data_name,'r',errors='replace') #open file
        lines=datfile.readlines() #reads in all lines
        datfile.close() #closes file
        lines=[f.strip('\n') for f in lines] #cuts out all empty lines
        
        #creates empty lists for running
        nlines=[]
        times=[];dist=[]
        
        for dat in lines: #loops over all lines
            try: #if lines begin with a number, reads in, otherwise, ignores
                np.int(dat[0])
                nlines.append(dat)
            except: continue
        nlines=nlines[3:] #skips the first 3 line, since these are headers
        for dat in nlines:
            dat=dat.split() #splits up the line at spaces
            times.append(np.float(dat[0])) #time is the first number 
            dist.append(np.float(dat[3])) #distance is the 4th number
        times=86400*np.array([times])[0] #time in days, converting to seconds
        dist=1.49e13*np.array([dist])[0]#distance in au, converting to centimeters

        #cut out distances/times greater than the cutoff
        times=times[np.where(dist<=cutoff)] 
        dist=dist[np.where(dist<=cutoff)]
        
        #sets the minimum time, for adding back at the end
        self.mint=np.min(times)
        times=times-self.mint

        return(times,dist)
    
    def convert_time(self,t):
        '''
        Performs a simple time conversion, from seconds to 
        hours:minutes:seconds.
    
        Parameters
        ----------
        t : float
            Time in seconds to be converted.
    
        Returns
        -------
        str : A string of the format "H:M:S", the hours, minutes, and seconds 
            which make up t seconds.
    
        '''
        return(time.strftime("%H:%M:%S", time.gmtime(t)))
    
    def get_outputs(self):
        try: self.times
        except AttributeError: self.times=[]
        try: self.outputs
        except AttributeError: self.outputs=[]
        
        self.times.append(self.t)

        outlist=[]	
        for func in self.out_funcs:
            fv = func()[0]
            if type(fv) is list:
                outlist.extend(fv)
            else:
                outlist.append(fv)
        
        self.outputs.append(outlist)

    def princ_axes(self):
        coords=self.V.tabulate_dof_coordinates()[::3]
        return(list(np.max(coords,axis=0)),["a","b","c"])

    def moment_of_inertia(self):
        '''
        Computes the moment of inertia of the body, which is the integral of 
        the density times the minimum distance to the rotation axis squared. 
        In g cm^2.
    
        Returns
        -------
        None.
    
        '''
        self.get_coords() #update the coordinates of the mesh
        
        # takes the cross product of the position and rotation vectors, giving
        # ||r||*||omega||*sin(theta), with theta the angle between r and omega.
        # note that the minimum distance to the rotation axis is 
        #||r||*sin(theta).
        A1=project(cross(self.r,self.omega),self.V) 
        
        # computes ||omega||^2
        ommag=project(dot(self.omega,self.omega),self.Q)
        
        # computes ||A1||^2, so this is ||r||^2*||omega||^2*sin(theta)^2, which
        # is also d^2*||omega||^2
        A2=project(dot(A1,A1),self.Q)
        
        #dividing out ||omega||^2 to get d^2
        d2=project(abs(A2/ommag),self.Q)
        
        #computes and returns the integral of d^2*rho over the body
        return(assemble(self.rho*d2*dx),"MoIs")
    

    def CFL(self,dt):
        '''
        Computes the CFL criterion, using the velocity self.u and over the 
        time dt.

        Parameters
        ----------
        dt : float
            The finite-element time difference we take the CFL over.

        Returns
        -------
        float : The maximum CFL criterion, which is defined as |u|*dt/dx, 
            where u is the velocity, dt the time difference, and dx the spatial 
            difference. 

        '''
        h = CellDiameter(self.mesh) #gets the diameter of each cell of the mesh, as dx
        dt=Constant(dt) #sets as a Constant for speed

        # Compute the CFL number cell wise
        DG = FunctionSpace(self.mesh, "DG", 0) #sets function space for math purposes
        CFL = project(sqrt(inner(self.u,self.u))*dt/h, DG) #computes the CFL criterion
        return(np.max(CFL.vector()[:])) #returns the maximum CFL criterion over the body
    
    def get_coords(self):
        '''
        Creates a vector (self.r) containing the coordinates of the mesh 
        points.
        
        Parameters
        ----------
        None.

        Returns
        -------
        None.

        '''
        #gets the coordinates from V, skipping every 3 since it's a vector
        meshpts=self.V.tabulate_dof_coordinates()[::3] 
        
        #creates r vector if not already created
        try: self.r
        except: self.r=Function(self.V)   
        
        self.r.vector()[:]=meshpts.flatten() #sets the r vector
        
    def tidal_acc(self,xyz):
        '''
        Computes the magnitude of the tidal stress force. 
    
        Parameters
        ----------
        xyz : numpy.ndarray
            The points at which to compute the tidal acceleration. Has shape (~,3), 
            which are the x,y,z coordinates. Note that the tidal force is 
            assumed to be in the y-direction.
    
        Returns
        -------
        acc : numpy.ndarray
            An array of the tidal acceleration at each of the input points. 
            Only non-zero along acc[:,1], since the force is in the 
            y-direction. acc is in units of cm s^-2.
            
        '''
        assert(xyz.shape[1]==3)
        
        #create solar mass variable, if not already existing
        try: self.M
        except AttributeError: self.M=Constant(2e33) #solar mass in g
        
        y=xyz[:,1] #gets the vector of y-values
        acc=np.zeros_like(xyz) #creates an all-zero force vector
        
        dist=self.trajectory(self.t) #gets the heliocentric distance at the current time
        
        #gets the acceleration as the difference between the acceleration at the 
        #points and the acceleration at the c.o.m.
        acc[:,1]=float(self.G)*float(self.M)*(1/(dist-y)**2-1/(dist)**2)
        return(acc)
    
    def rotate_points(self,pts,theta):
        '''
        Rotates the input point set by some angle theta about self.omega, 
        using quaternions. 
    
        Parameters
        ----------
        pts : numpy.ndarray
            Set of points to rotate, in shape (~,3).
        theta : float
            The angle to rotate the points in pts by, about the axis given by 
            the rotation vector. 
    
        Returns
        -------
        numpy.ndarray : Array of points, rotated by angle theta about the axis 
            defined by self.omega. Has shape (~,3). 
            
        '''
        assert(pts.shape[1]==3)
        
        rot=(self.omega.vector())[:3] #gets the rotation axis
        rot=rot/np.sqrt(np.sum(np.square(rot))) #normalizes the rotation axis
        a,b,c=rot #gets the axes
        
        #computes the quaternion from the rotation axis and angle
        rot=np.quaternion(np.cos(theta/2),np.sin(theta/2)*a, 
                                    np.sin(theta/2)*b,np.sin(theta/2)*c)
        rot=rot/np.abs(rot) #normalizes the quaternion
        
        pts=np.append(np.zeros(pts.shape[0])[:,np.newaxis],pts,axis=1) #adds a column of 0s
        pts=quaternion.as_quat_array(pts) #creates quaternion of pts
        
        out=rot*pts*np.conjugate(rot) #uses quaternions to rotate the points
        return(quaternion.as_float_array(out)[:,1:]) #returns the rotated points
    
    def save_funcs(self,*args):  
        '''
        Saves inputs into a ParaView .pvd file outfile.
    
        Parameters
        ----------
        *args : iterable
            Iterable of the functions to save.
    
        Returns
        -------
        None.
        
        '''
        for func in args:
            #saves all of the functions input
            self.outfile.write(func,self.t) 
                
    def compute_gravity(self):
        '''
        Computes the gravitational force on the body by solving the Gaussian 
        formulation. 

        Returns
        -------
        None.
        
        '''    
        #computes the gravity from the Gauss form, if it fails, marks divergence
        try: self.gravsolver.solve()
        except: 
            print("GRAVITY DIVERGED")
            
            #writes to log
            self.logfile.write("%s: STOPPED DUE TO DIVERGENCE IN GRAVITY \n"%
                               (self.convert_time(time.time()-self.start_time)))
            self.diverged=True #sets diverged to True, this breaks the run
            return
        
        #splits and updates the gravity function with the answers
        #note the gravscale
        gravg,gravs=self.gravgs.split()
        self.gravity.assign(project(gravg/self.gravscale,self.V))
        
    def compute_tides(self):
        '''
        Computes the tidal force vector.

        Returns
        -------
        None.

        '''
        P=self.period #takes the period of the asteroid rotation, assumed given
        phase=2*np.pi*(float(self.t)%P)/P #computes the angle of the asteroid
        
        points=self.V.tabulate_dof_coordinates()[::3] #gets points
        
        #rotates points to the position they should be in, in the solar frame
        points=self.rotate_points(points, -phase) 
        
        #gets the acceleration, multiplies by rho
        force=float(self.rho)*self.tidal_acc(points)
        
        #rotates the force back to the actual points
        force=self.rotate_points(force,phase)
        
        #updates the force function from the array
        self.ftides.vector()[:]=force.flatten()
        
    def compute_centrifugal(self):
        '''
        Computes the centrifugal force, using 
        Fcentri=-rho*(omega x (omega x r)).

        Returns
        -------
        None.

        '''
        self.get_coords() #updates the coordinates
        
        #computes the centrifugal force
        self.centrifugal.assign(project(-1*self.rho*
                        cross(self.omega,cross(self.omega,self.r)),self.V))

    def compute_coriolis(self):
        '''
        Computes the Coriolis force, using Fcori=-2*rho*(omega x u).

        Returns
        -------
        None.

        '''
        #computes the function
        self.coriolis.assign(project(-2*self.rho*cross(self.omega,self.u),self.V))
        
    def move_mesh(self,dt):
        '''
        Moves the mesh, with velocity self.u, over time dt. This uses the 
        built-in ALE functions from FEniCS.

        Parameters
        ----------
        dt : float
            The time difference to move the mesh over. Used to compute the 
            displacement, from the velocity.

        Returns
        -------
        None.

        '''
        #gets the displacement vector from dt*u
        move=project(Constant(dt)*self.u,self.V)
        
        #use ALE to move the mesh, this updates all functions defined on the mesh
        ALE.move(self.mesh,move)
    
    def sum_u(self):
        '''
        Adds up the self.u vectors, to allow for averaging. 
        
        Returns
        -------
        None.

        '''
        try: self.usum.vector()[:]+=self.u.vector()[:] #adds the velocity to the sum
        except AttributeError: 
            self.usum=self.u.copy(deepcopy=True) #initializes the sum
                
    def adjust_u(self):
        '''
        Subtracts the average velocity in each direction, so that the center 
        of mass of the body has no velocity. This ensures that the body sticks
        to the prescribed trajectory. The forces acting on the body should be 
        symmetric, so this is entirely removing numerical noise. 
        
        Due to the instability of the null-velocity c.o.m. (tidal forces grow 
        with distance), this adjustment must be applied at every step.

        Returns
        -------
        None.

        '''
        #computes the volume integrals of the x,y, and z components of u
        ux=assemble(self.u.sub(0)*dx)
        uy=assemble(self.u.sub(1)*dx)
        uz=assemble(self.u.sub(2)*dx)
        
        #creates a function of value 1, which can be integrated.
        try: self.unit
        except AttributeError:
            self.unit=Function(self.Q);self.unit.assign(Constant(1))
        
        Vol=assemble(self.unit*dx) #compute the volume of the body
        
        try:self.umean
        except AttributeError:self.umean=Function(self.Z)
        
        #computes the volume-averaged component means
        self.umean.assign(Constant((ux/Vol,uy/Vol,uz/Vol,0)))
        
        #subtracts the mean from the solution function 
        self.up.assign(self.up-self.umean) 
        
    def size_check(self):
        '''
        Checks the size of the body relative to its initial size, and cuts the 
        simulation if it has crossed the threshold.

        Returns
        -------
        None.

        '''
        #gets the coordinates of the mesh nodes
        coords=self.V.tabulate_dof_coordinates()[::3] 
        
        #computes the magnitude/distance of the points from the c.o.m.
        mags=np.sqrt(np.sum(np.square(coords),axis=1))
        
        if np.max(mags)>self.sizecut: #if the maximum distance is greater than the cut
            self.diverged=True #sets as divergent
            
            #prints and saves logs
            print("--- SIZE BREAKS THRESHOLD ---")
            self.logfile.write("--- SIZE BREAKS THRESHOLD --- \n")
    
    def compute_time_step(self):
        '''
        Solves the Navier-Stokes equations using finite-difference Euler 
        methods at a time step. 

        Returns
        -------
        None.

        '''
        #appends the current time/MoI to the lists
        self.get_outputs()       
 
        #prints
        print("-------------------------")
        print("Now Running Cycle {}, t: {:.3e}, Completed {:.2f}%, CFL: {:.3e}"
              .format(self.ind,self.t,100*self.t/self.end_time,self.CFL(self.dt)))
        
        
        try: self.solver.solve()
        except: 
            print("DIVERGED")
            self.logfile.write("%s: STOPPED DUE TO DIVERGENCE \n"%
                               (self.convert_time(time.time()-self.start_time)))
            self.diverged=True
            return
        
        #if we want to save at steps, save all the functions
        if self.savesteps: 
            self.save_funcs(self.u,self.p,self.ftides,self.gravity,self.centrifugal,
                            self.coriolis,self.forcing)
            
        #write to log
        self.logfile.write(
            "{}: --- Solved Cycle {}, t={:.3e}, Completed {:.2f}%, CFL: {:.3e} --- \n"
            .format(self.convert_time(time.time()-self.start_time),self.ind,
                    self.t,100*self.t/self.end_time,self.CFL(self.dt)))
        
        self.adjust_u() #remove the mean velocity
        self.u_p_.assign(self.up) #assign the current solution to the prior solution
        self.ind+=1 #updates the run index
        
    def update_forces(self):
        '''
        Helper function, calls all of the force-updating functions to avoid 
        clutter. 

        Returns
        -------
        None.

        '''
        self.compute_gravity()
        self.compute_tides()
        self.compute_centrifugal()
        self.compute_coriolis()
        self.forcing.assign(self.ftides+self.gravity+self.centrifugal+self.coriolis)
        
    def compute_rotation(self,ncyc=3):
        '''
        Runs the simulation over a few cycles, taking the average velocity for 
        use in an trajectory jump. This cycle-jump system significantly improves
        efficiency. 

        Parameters
        ----------
        ncyc : int, optional
            The number of full rotations to run and average over before 
            jumping along the trajectory. The default is 3.

        Returns
        -------
        None.

        '''
        nrun=self.nsrot*ncyc #number of Euler steps in ncyc rotations
        
        for i in range(nrun):
            self.update_forces()
            self.size_check()
            self.logfile.write("%s: Forces Computed \n"%
                               (self.convert_time(time.time()-self.start_time)))
            
            if self.diverged: break #stop if diverged
            self.compute_time_step() #solve equations
            self.sum_u() #add the new velocity to the total
            self.move_mesh(self.dt) #move the mesh 

            self.t+=self.dt
        
        if not self.diverged:
            self.ucycavg=self.usum.copy(deepcopy=True) #create a function for the average
            self.ucycavg.assign(self.usum/nrun) #average
            del self.usum #delete the current sum, very important

    def compute_trajectory_step(self):
        '''
        Computes a jump along the trajectory, using the average velocity from the 
        previous cycles. Ensures that the CFL criterion is met, and keeps the 
        change in heliocentric distance to a prespecified tolerance. Assists 
        in improving efficiency of the simulation.

        Returns
        -------
        None.

        '''
        if not self.diverged:
            #gets the heliocentric distance
            dist=self.trajectory(self.t) 
            
            #gets the initial time
            inittime=self.t
            
            #while the distance has changed by less than rtol percent
            while np.abs(self.trajectory(self.t)/dist-1)<self.rtol:
                self.t+=self.nsrot*self.dt #step over a full rotation each time
                if self.CFL(self.t-inittime)>=1: #check CFL criterion
                    #if CFL>1, reverse the time until CFL<1
                    while self.CFL(self.t-inittime)>=1:
                        self.t-=self.dt
                    break
            
            timejump = self.t-inittime #find the total time change
            
            self.u.vector()[:]=self.ucycavg.vector()[:] #set the velocity to the average
            self.move_mesh(timejump) #move the mesh over this displacement
            
            #save the new time/MoI
            self.get_outputs()
            
            #write updates
            print("-------------------------")
            print("{}: Trajectory Jump Completed, Stepped {:.3f} s, t={:.3e}, {:.3e}%"
                .format(self.convert_time(time.time()-self.start_time),timejump,
                        self.t,100*(self.t/self.end_time)))
            print("------------------------- \n")
            self.logfile.write("{}: --- Trajectory Jump Completed, Stepped {:.3f} s, {:.2f}%, CFL: {:.3e}---\n"
                               .format(self.convert_time(time.time()-self.start_time),
                                       timejump,100*(self.t/self.end_time),self.CFL(timejump)))
    
    def run_model(self,nsrot=10,rtol=0.01,period=7.937,savesteps=False,out_funcs=['moment_of_inertia','princ_axes']):
        '''
        Helper function which runs the simulation, to avoid cluttering. 

        Parameters
        ----------
        nsrot : int
            Number of computational steps in each rotation.
        rtol : float, optional
            The allowed tolerance in the heliocentric distance when computing 
            the trajectory jump. The default is 0.01.
        period : float, optional
            The rotational period of the body, in hours. The default is 7.937 
            hours, which is for 'Oumuamua.
        savesteps : boolean, optional
            Whether or not to save the functions and mesh at each computational
            step. This can quickly overwhelm storage if many runs are used. 
            The default is False.

        Returns: 
        -------
        moi_check : boolean
            -True: If the relative moment of inertia changes by less than moitol 
             over the entire trajectory.
            -False: If the simulation diverges or if the relative moment of inertia 
             changes by more than moitol. 
            
        '''
        
        assert(rtol>0)

        for i,func in enumerate(out_funcs):
            if type(func) is str:
                out_funcs[i]=getattr(self,func)	
        
        self.out_funcs=out_funcs	

        #initializes parameters
        self.diverged=False
        self.savesteps=savesteps
        self.rtol=rtol
        self.period=period*3600 #converting from hour to seconds
        self.nsrot=nsrot
        self.dt=self.period/nsrot #in seconds
        
        #computes the magnitude of the rotation vector
        mag=2*np.pi/(self.period) 
        
        #computes the magnitude of the current rotation vector
        vecmag=np.sqrt(np.sum(np.square(self.omega.vector()[:3]))) 
        
        #assigns the rotation vector magnitude
        self.omega.assign(Constant(tuple(self.omega.vector()[:3]*mag/vecmag)))
        
        #runs the simulation, alternating between the rotation cycles and the 
        #trajectory steps until the time reaches the end or the simulation
        #diverges
        while (self.t<=self.end_time) and not (self.diverged):
            self.compute_rotation()
            self.compute_trajectory_step()
            
        #save the functions at the end of the runs
        self.save_funcs(self.u,self.p,self.ftides,self.gravity,self.centrifugal,
                        self.coriolis,self.forcing)
        
        #saves the times and MoIs to a csv
        outnames=["Times"]
        for func in out_funcs:
            fn=func()[1]
            if type(fn) is list:
                outnames.extend(fn)
            else:
                outnames.append(fn)

        out_data=np.insert(np.array(self.outputs),0,np.array(self.times+self.mint),axis=1)
        outFrame=pd.DataFrame(out_data,columns=outnames)
        outFrame.to_csv('logs/Outputs_{}_{}.csv'.format(self.name,int(np.log10(float(self.mu)))))
            
        #writes to the logfile
        self.logfile.write("{}: --- Finished, Run Time: {:.3e} --- \n"
                           .format(self.convert_time(time.time()-self.start_time),
                                   (time.time()-self.start_time)))
        
        self.logfile.close() #closes the log file
        
        return(outFrame)
    
    def reset_model(self,name=None,a=None,b=None,c=None,mu=None,omegavec=None,
                  rho=None,szscale=None,n=0):
        '''
        Resets the model, using the __init__() function. All parameters have the
        same function as in __init__().

        Parameters
        ----------
        Same as __init__(), all optional. If not provided, uses current values.

        Returns
        -------
        None.

        '''
        if name is None: name=self.name
        if a is None: a=self.a
        if b is None: b=self.b
        if c is None: c=self.c
        if mu is None: mu=float(self.mu)
        if omegavec is None: omegavec=self.omegavec
        if rho is None: rho=float(self.rho)
        if szscale is None: szscale=self.szscale
        
        
        assert(len(omegavec)==3)
        assert(szscale>=1)
        assert(n>=0)
        self.name=name #setting the name
        self.omegavec=omegavec #setting the rotation axis
        self.a=a;self.b=b;self.c=c
        
        a*=100;b*=100;c*=100 #converting the axes from meters to cm
        self.sizecut=szscale*np.max([a,b,c])/2 #setting the maximum allowed size
        
        self.mu=Constant(mu) #creating a Constant to avoid slowdowns
        self.t=0;self.ind=0 #initializing the time, and the number of cycles
        self.rho=Constant(rho) #creates a Constant to avoid slowdowns
                    
        nmesh=Mesh('3ball%s.xml'%(n)) #read in mesh, with n refinements
        
        #rescales the mesh to the input ellipsoids
        nmesh.coordinates()[:,0]*=a/2 
        nmesh.coordinates()[:,1]*=b/2
        nmesh.coordinates()[:,2]*=c/2
        
        bmesh=BoundaryMesh(nmesh,'exterior')
        ALE.move(self.mesh,bmesh)
        
        self.set_spline() #computes the spline for the trajectories
        
        self.omega=interpolate(Constant(tuple(omegavec)),self.V)
        
        #sets solution functions to 0
        self.up.assign(Constant((0,0,0,0)))
        self.u_p_.assign(Constant((0,0,0,0)))
