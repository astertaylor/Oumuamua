"""
Created on Wed May 25 13:57:01 2022.

@author: Aster Taylor
"""
from fenics import *  # analysis:ignore
import numpy as np
import time
import pandas as pd
from scipy.interpolate import UnivariateSpline
import quaternion
import os


import importlib.resources as pkg_resources
from testingSAMUS import meshes


class SAMUS:
    """
    Create a model of an ellipsoidal asteroid and simulates its evolution.

    Model the deformation of an ellipsoidal asteroid over a given trajectory,
    for given model parameters. Modeled under tidal deformation, as well as the
    self-gravitational force, the Coriolis force, and the centrifugal force.

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
        File where the model mesh and functions are saved, '.pvd' ParaView
        file.

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

    """

    def __init__(self, name, a=115, b=111, c=19, mu=10**7, omegavec=[0, 0, 1],
                 rho=0.5, szscale=2, n=0):
        """
        Init.

        Set up the model, create initial parameters, set up the mesh,
        set up functions, and prepare the solver equations for gravity
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

        """
        assert(len(omegavec) == 3)
        assert(szscale >= 1)
        assert(n >= 0)

        # set the name
        self.name = name

        # set the rotation axis
        self.omegavec = omegavec

        # set the principal axes
        self.a = a
        self.b = b
        self.c = c

        # set the size scale
        self.szscale = szscale

        # convert the axes from meters to cm
        a *= 100
        b *= 100
        c *= 100

        # set the maximum allowed size
        self.sizecut = szscale*np.max([a, b, c])/2

        # set viscosity, create a Constant to avoid slowdowns
        self.mu = Constant(mu)

        # initialize the time, and the number of cycles
        self.t = 0
        self.ind = 0

        # set dt to 1 temporarily, for use in the solvers
        self.dt = Constant(1)

        # set density, create a Constant to avoid slowdowns
        self.rho = Constant(rho)

        # set the inital time, for logging
        self.start_time = time.time()

        #create log directory
        # get path
        path=os.path.join(os.getcwd(),'logs')
        # try to make directory
        try: os.mkdir(path)
        #if it already exists, continue
        except FileExistsError: pass

        # create log file
        self.logfile = open(
            'logs/{}_{}.txt'.format(self.name, int(np.log10(mu))), 'w')

        # write initial time to log file
        self.logfile.write("%s: Beginning Initialization... \n" %
                           (self.convert_time(time.time()-self.start_time)))

        # read in mesh, with n refinements
        mesh_path=pkg_resources.path(meshes,'3ball%s.xml' % (n))
        self.mesh = Mesh(str(mesh_path))

        # rescale the mesh to the input ellipsoids
        self.mesh.coordinates()[:, 0] *= a/2
        self.mesh.coordinates()[:, 1] *= b/2
        self.mesh.coordinates()[:, 2] *= c/2

        "create results directory"
        path=os.path.join(os.getcwd(),'results')
        # try to make directory
        try: os.mkdir(path)
        #if it already exists, continue
        except FileExistsError: pass

        # create output file
        self.outfile = File(
            "results/{}_{}.pvd".format(self.name, int(np.log10(mu))))

        # use Elements to make a mixed function space
        V = VectorElement("CG", self.mesh.ufl_cell(), 2)
        Q = FiniteElement("CG", self.mesh.ufl_cell(), 1)
        self.Z = FunctionSpace(self.mesh, V*Q)

        # create actual function spaces which compose the mixed
        self.V = VectorFunctionSpace(self.mesh, "CG", 2)
        self.Q = FunctionSpace(self.mesh, "CG", 1)

        # create solution functions from the mixed space
        self.up = Function(self.Z)  # solution function
        self.u_p_ = Function(self.Z)  # function for previous solutions

        # get trial and test functions from the mixed space
        dup = TrialFunction(self.Z)
        v, q = TestFunctions(self.Z)

        # create the function of the rotation vector
        self.omega = interpolate(Constant(tuple(omegavec)), self.V)

        # split the solution functions
        self.u, self.p = split(self.up)
        u_, p_ = split(self.u_p_)

        # set solution functions to 0
        self.up.assign(Constant((0, 0, 0, 0)))
        self.u_p_.assign(Constant((0, 0, 0, 0)))

        # create the functions for storing the forces
        self.ftides = Function(self.V)  # tides
        self.gravity = Function(self.V)  # gravity
        self.centrifugal = Function(self.V)  # centrifugal
        self.coriolis = Function(self.V)  # coriolis
        self.forcing = Function(self.V)  # total forces

        # name the functions for storage
        self.ftides.rename("Tidal Force", "Tidal Force")
        self.gravity.rename("Self-Gravity", "Gravitational Force")
        self.centrifugal.rename("Centrifugal", "Centrifugal Force")
        self.coriolis.rename("Coriolis", "Coriolis Force")
        self.forcing.rename("Forcing", "Total force on the object")

        # create a constant to ensure solution stability
        A = Constant(1e4/max(mu, 1e4))

        # create the solution for the Navier-Stokes equations
        F = (
            # acceleration term
            A*self.rho*inner(((self.u-u_)/(self.dt)), v) * dx +

            # viscosity term
            A*self.mu*inner(grad(self.u), grad(v)) * dx +

            # advection term
            A*self.rho*inner(dot(self.u, nabla_grad(self.u)), v) * dx -

            # pressure term
            A*self.p*div(v) * dx +

            # mass continuity equation
            q*div(self.u) * dx -

            # force term
            A*inner(self.forcing, v) * dx)

        # find the derivative, for speed
        J = derivative(F, self.up, dup)

        # set up the Navier-Stokes solver
        problem = NonlinearVariationalProblem(F, self.up, J=J)
        self.solver = NonlinearVariationalSolver(problem)
        self.solver.parameters['newton_solver']['relaxation_parameter'] = 1.

        # split solution functions for access (weird FEniCS quirk)
        self.u, self.p = self.up.split()
        u_, p_ = self.u_p_.split()

        # name the solution functions
        self.u.rename("Velocity", "Velocity")
        self.p.rename("Pressure", "Pressure")

        # COMPUTE FUNCTIONS FOR GRAVITY SOLUTIONS
        self.G = Constant(6.674e-8)  # sets gravitational constant, in cgs

        # get solution, trial, and test functions
        self.gravgs = Function(self.Z)
        dgs = TrialFunction(self.Z)
        gravh, gravc = TestFunctions(self.Z)
        gravg, gravs = split(self.gravgs)

        # set a scale to ensure the stability of the solution. this is undone
        # in the solution, but for unknown reasons O(10^-8) is too large for
        # the solver to maintain stability
        self.gravscale = 1e-3

        # compute the scaling constant for the Gaussian gravity form, which is
        # rescaled by self.gravscale. A Constant, for speed
        gravA = Constant(4*np.pi*float(self.G)*float(self.rho)*self.gravscale)

        # creates the equation set for Gaussian gravity
        gravF = (
            # this equation is 0=0, used to mix vector and scalar solutions
            gravs*div(gravh) * dx + inner(gravg, gravh) * dx +
            # this equation is the Gaussian form, div(g)=-4 pi G rho
            gravc*div(gravg) * dx + gravA*gravc * dx)

        # find the derivative, for speed
        gravJ = derivative(gravF, self.gravgs, dgs)

        # set up the gravitational solver
        gravproblem = NonlinearVariationalProblem(gravF, self.gravgs, J=gravJ)
        self.gravsolver = NonlinearVariationalSolver(gravproblem)
        self.gravsolver.parameters['newton_solver'
                                   ]['relaxation_parameter'] = 1.

        # write to log
        self.logfile.write("%s: Initializations Complete \n" %
                           (self.convert_time(time.time()-self.start_time)))

    def read_trajectory(self, data_name, cutoff=4.7e13):
        """
        Read and returns New Horizons data for time and solar distance.

        This function is assumed to be reading in New Horizons data, with times
        in days and distances in AU.

        Parameters
        ----------
        data_name : str
            The name of the file containing the data.
        cutoff : float, optional
            The maximum heliocentric distance we want to consider. Any distance
            beyond this will be cut out of the data. The default is chosen such
            that the tidal force is always greater than 1e-2.
            The default is 4.7e13 cm.

        Returns
        -------
        None.

        """
        # open file
        datfile = open(data_name, 'r', errors='replace')

        # read in all lines
        lines = datfile.readlines()

        # close file
        datfile.close()

        # cut out all empty lines
        lines = [f.strip('\n') for f in lines]

        # create empty lists for running
        nlines = []
        times = []
        dist = []

        # loop over all lines
        for dat in lines:
            # if lines begin with a number, read in, otherwise, ignore
            try:
                np.int(dat[0])
                nlines.append(dat)
            except:
                continue

        # skip the first 3 line, since these are headers
        nlines = nlines[3:]
        for dat in nlines:
            # split up the line at spaces
            dat = dat.split()

            # time is the first number
            times.append(np.float(dat[0]))

            # distance is the 4th number
            dist.append(np.float(dat[3]))

        # time in days, convert to seconds
        times = 86400*np.array([times])[0]

        # distance in au, convert to centimeters
        dist = 1.49e13*np.array([dist])[0]

        # cut out distances/times greater than the cutoff
        times = times[np.where(dist <= cutoff)]
        dist = dist[np.where(dist <= cutoff)]

        # set the minimum time, for adding back at the end
        self.mint = np.min(times)
        times = times-self.mint

        # find the ending time for cutoffs
        self.end_time = times[-1]

        # create spline
        self.trajectory = UnivariateSpline(times, dist)

    def convert_time(self, t):
        """
        Perform a simple time conversion, from seconds to hour:minute:second.

        Parameters
        ----------
        t : float
            Time in seconds to be converted.

        Returns
        -------
        str : A string of the format "H:M:S", the hours, minutes, and seconds
            which make up t seconds.

        """
        return(time.strftime("%H:%M:%S", time.gmtime(t)))

    def save_funcs(self, *args):
        """
        Save inputs into a ParaView .pvd file outfile.

        Parameters
        ----------
        *args : iterable
            Iterable of the functions to save.

        Returns
        -------
        None.

        """
        for func in args:
            # save all of the functions input
            self.outfile.write(func, self.t)

    def add_method(self,func):
        """
        Add functions to the class.

        Can have any argument set, including 'self'. If 'self' is included in
        the function definition, the function can access class variables. Keeps
        the docstring of the function intact.

        Parameters
        ----------
        func : function
            A function to add into the class. Is able to take 'self' as an
            input, can also take other values. This is used to allow modular
            single-step outputs.

        Returns
        -------
        None.

        """
        # define a wrapper to add functions which don't have 'self'
        def wrapper(self, *args, **kwargs):
            return func(*args, **kwargs)

        # if function takes no parameters
        if func.__code__.co_argcount==0:

            # add with 'self'
            setattr(SAMUS,func.__name__,wrapper)

        # if function does not take 'self'
        elif not "self" in func.__code__.co_varnames[0]:

            # add with 'self'
            setattr(SAMUS,func.__name__,wrapper)

        # if function takes 'self'
        elif "self" in func.__code__.co_varnames[0]:

            # add unmodified
            setattr(SAMUS,func.__name__,func)

    def get_outputs(self):
        """
        Get the outputs of the output functions and updates the output list.

        Get the values from self.out_funcs, if it's a string, splits it, and
        then store the values in self.outputs. This is used for the writing of
        the per-timestep values. Each function must output a tuple of the
        values and then the name of the values, either in a list or a single
        number.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        # check to see if the outputs exist
        try:
            self.times
        except AttributeError:
            self.times = []
        try:
            self.outputs
        except AttributeError:
            self.outputs = []

        # append the time to the times list
        self.times.append(self.t)

        outlist = []  # initalize

        # run for the output functions
        for func in self.out_funcs:
            # get the function values
            fv = func()[0]

            # if the output is a list, use extend
            if type(fv) is list:
                outlist.extend(fv)

            # otherwise, use append
            else:
                outlist.append(fv)

        self.outputs.append(outlist)

    def princ_axes(self):
        """
        Get the principal axes of the body.

        Compute as the maximum value of the coordinates of the body in the x,
        y, z directions.

        Returns
        -------
        list : A list containing the maximum values in the x,y,z values.
        list : A list containing the strings "a","b","c". This is the name of
            the outputs, used by the modular code to name the outputs.

        """
        # get coordinates of mesh
        coords = self.V.tabulate_dof_coordinates()[::3]

        return(list(np.max(coords, axis=0)), ["a", "b", "c"])

    def moment_of_inertia(self):
        """
        Compute the moment of inertia.

        Compute the moment of inertia of the body, which is the integral of
        the density times the minimum distance to the rotation axis squared.
        In g cm^2.

        Returns
        -------
        float : The moment of inertia of the body.
        str : The string "MoIs", the name of the outputs.

        """
        # update the coordinates of the mesh
        self.get_coords()

        # take the cross product of the position and rotation vectors, giving
        # ||r||*||omega||*sin(theta), with theta the angle between r and omega.
        # note that the minimum distance to the rotation axis is
        # ||r||*sin(theta).
        A1 = project(cross(self.r, self.omega), self.V)

        # compute ||omega||^2
        ommag = project(dot(self.omega, self.omega), self.Q)

        # compute ||A1||^2, so this is ||r||^2*||omega||^2*sin(theta)^2, which
        # is also d^2*||omega||^2
        A2 = project(dot(A1, A1), self.Q)

        # divide out ||omega||^2 to get d^2
        d2 = project(abs(A2/ommag), self.Q)

        # compute and return the integral of d^2*rho over the body
        return(assemble(self.rho*d2*dx), "MoIs")

    def CFL(self, dt):
        """
        Compute the CFL criterion.

        Get the CFL criterion, using the velocity self.u and over the time dt.

        Parameters
        ----------
        dt : float
            The finite-element time difference we take the CFL over.

        Returns
        -------
        float : The maximum CFL criterion, which is defined as |u|*dt/dx,
            where u is the velocity, dt the time difference, and dx the spatial
            difference.

        """
        # get the diameter of each cell of the mesh, as dx
        h = CellDiameter(self.mesh)

        # set as a Constant for speed
        dt = Constant(dt)

        # compute the CFL number cell wise
        # set function space for math purposes
        DG = FunctionSpace(self.mesh, "DG", 0)

        # compute the CFL criterion
        CFL = project(sqrt(inner(self.u, self.u))*dt/h, DG)

        # return the maximum CFL criterion over the body
        return(np.max(CFL.vector()[:]))

    def update_dt(self):
        """
        Update the timestep to ensure CFL<Cmax.

        Returns
        -------
        None.

        """
        if self.CFL(self.dt) >= self.Cmax:
            # find the updated timestep
            ndt = self.dt*self.Cmax/self.CFL(self.dt)

            # update the timestep
            self.dt = ndt

    def size_check(self):
        """
        Check the size of the body.

        Get the size of the body relative to its initial size, and cut the
        simulation if it has crossed the threshold.

        Find the distance from each point to the center of mass, and then
        compare this to the size cutoff. Use self.diverged as the method for
        cutting the simulation, set it to True if this diverges.

        Returns
        -------
        None.

        """
        # get the coordinates of the mesh nodes
        coords = self.V.tabulate_dof_coordinates()[::3]

        # compute the magnitude/distance of the points from the c.o.m.
        mags = np.sqrt(np.sum(np.square(coords), axis=1))

        # if the maximum distance is greater than the cut
        if np.max(mags) > self.sizecut:
            # set as divergent
            self.diverged = True

            # print and save logs
            print("--- SIZE BREAKS THRESHOLD ---")
            self.logfile.write("--- SIZE BREAKS THRESHOLD --- \n")

    def get_coords(self):
        """
        Create a vector (self.r) containing the coordinates of the mesh.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        # get the coordinates from V, skipping every 3 since it's a vector
        meshpts = self.V.tabulate_dof_coordinates()[::3]

        # create r vector if not already created
        try:
            self.r
        except:
            self.r = Function(self.V)

        # set the r vector
        self.r.vector()[:] = meshpts.flatten()

    def rotate_points(self, pts, theta):
        """
        Rotate the input point set by some angle theta about self.omega.

        This function computes this rotation using quaternions.

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

        """
        assert(pts.shape[1] == 3)

        # get the rotation axis
        rot = (self.omega.vector())[:3]

        # normalize the rotation axis
        rot = rot/np.sqrt(np.sum(np.square(rot)))

        # get the axes
        a, b, c = rot

        # compute the quaternion from the rotation axis and angle
        rot = np.quaternion(np.cos(theta/2), np.sin(theta/2)*a,
                            np.sin(theta/2)*b, np.sin(theta/2)*c)

        # normalize the quaternion
        rot = rot/np.abs(rot)

        # add a column of 0s to the pts
        pts = np.append(np.zeros(pts.shape[0])[:, np.newaxis], pts, axis=1)

        # create quaternion of pts
        pts = quaternion.as_quat_array(pts)

        # use quaternions to rotate the points
        out = rot*pts*np.conjugate(rot)

        # return the rotated points
        return(quaternion.as_float_array(out)[:, 1:])

    def tidal_acc(self, xyz):
        """
        Compute the magnitude of the tidal stress force.

        Parameters
        ----------
        xyz : numpy.ndarray
            The points at which to compute the tidal acceleration. Has shape
            (~,3), which are the x,y,z coordinates. Note that the tidal force
            is assumed to be in the y-direction.

        Returns
        -------
        acc : numpy.ndarray
            An array of the tidal acceleration at each of the input points.
            Only non-zero along acc[:,1], since the force is in the
            y-direction. acc is in units of cm s^-2.

        """
        assert(xyz.shape[1] == 3)

        # create solar mass variable, if not already existing
        try:
            self.M
        except AttributeError:
            self.M = Constant(2e33)  # solar mass in g

        # get the vector of y-values
        y = xyz[:, 1]

        # create an all-zero force vector
        acc = np.zeros_like(xyz)

        # get the heliocentric distance at the current time
        dist = self.trajectory(self.t)

        # get the acceleration as the difference between the acceleration at
        # the points and the acceleration at the c.o.m.
        acc[:, 1] = float(self.G)*float(self.M)*(1/(dist-y)**2-1/(dist)**2)
        return(acc)

    def compute_tides(self):
        """
        Compute the tidal force vector.

        Returns
        -------
        None.

        """
        # take the period of the asteroid rotation
        P = self.period

        # compute the angle of the asteroid
        phase = 2*np.pi*(float(self.t) % P)/P

        # get points
        points = self.V.tabulate_dof_coordinates()[::3]

        # rotate points to the position they should be in, in the solar frame
        points = self.rotate_points(points, -phase)

        # get the acceleration, multiplies by rho
        force = float(self.rho)*self.tidal_acc(points)

        # rotate the force back to the actual points
        force = self.rotate_points(force, phase)

        # update the force function from the array
        self.ftides.vector()[:] = force.flatten()

    def compute_gravity(self):
        """
        Compute the gravitational force on the body.

        Solve the Gaussian formulation.

        Returns
        -------
        None.

        """
        # compute the gravity from the Gauss form.
        # if it fails, marks divergence
        try:
            self.gravsolver.solve()
        except:
            print("GRAVITY DIVERGED")

            # write to log
            self.logfile.write("%s: STOPPED DUE TO DIVERGENCE IN GRAVITY \n" %
                               (self.convert_time(time.time() -
                                                  self.start_time)))
            self.diverged = True  # set diverged to True, break the run
            return

        # split and update the gravity function with the answers
        # note the gravscale
        gravg, gravs = self.gravgs.split()

        # assign the result to the gravity function
        self.gravity.assign(project(gravg/self.gravscale, self.V))

    def compute_centrifugal(self):
        """
        Compute the centrifugal force.

        Use Fcentri=-rho*(omega x (omega x r)).

        Returns
        -------
        None.

        """
        # update the coordinates
        self.get_coords()

        # compute the centrifugal force
        self.centrifugal.assign(project(
            -1*self.rho*cross(self.omega, cross(self.omega, self.r)), self.V))

    def compute_coriolis(self):
        """
        Compute the Coriolis force, using Fcori=-2*rho*(omega x u).

        Returns
        -------
        None.

        """
        # compute the Coriolis force
        self.coriolis.assign(
            project(-2*self.rho*cross(self.omega, self.u), self.V))

    def update_forces(self):
        """
        Call all of the force-updating functions.

        This is a helper function to avoid clutter.

        Returns
        -------
        None.

        """
        # update all the functions
        self.compute_gravity()
        self.compute_tides()
        self.compute_centrifugal()
        self.compute_coriolis()

        # add together the forces into the summation function
        self.forcing.assign(self.ftides+self.gravity +
                            self.centrifugal+self.coriolis)

    def move_mesh(self, dt):
        """
        Move the mesh, with velocity self.u, over time dt.

        This uses the built-in ALE functions from FEniCS.

        Parameters
        ----------
        dt : float
            The time difference to move the mesh over. Used to compute the
            displacement, from the velocity.

        Returns
        -------
        None.

        """
        # get the displacement vector from dt*u
        move = project(Constant(dt)*self.u, self.V)

        # use ALE to move the mesh.
        # this updates all functions defined on the mesh
        ALE.move(self.mesh, move)

    def sum_u(self):
        """
        Add up the self.u vectors, to allow for averaging.

        Returns
        -------
        None.

        """
        try:
            # add the velocity to the sum
            self.usum.vector()[:] += self.u.vector()[:]
        except AttributeError:
            # initialize the sum
            self.usum = self.u.copy(deepcopy=True)

    def adjust_u(self):
        """
        Subtract the average velocity in each direction.

        This ensures that the center of mass of the body has no velocity, and
        that the body sticks to the prescribed trajectory. The forces acting on
        the body should be symmetric, so this is entirely removing this sort of
        numerical noise.

        Due to the instability of the null-velocity c.o.m. (tidal forces grow
        with distance), this adjustment must be applied at every step.

        Returns
        -------
        None.

        """
        # compute the volume integrals of the x,y, and z components of u
        ux = assemble(self.u.sub(0)*dx)
        uy = assemble(self.u.sub(1)*dx)
        uz = assemble(self.u.sub(2)*dx)

        # create a function of value 1, which can be integrated.
        try:
            self.unit
        except AttributeError:
            self.unit = Function(self.Q)
            self.unit.assign(Constant(1))

        # compute the volume of the body
        Vol = assemble(self.unit*dx)

        try:
            self.umean
        except AttributeError:
            self.umean = Function(self.Z)

        # compute the volume-averaged component means
        self.umean.assign(Constant((ux/Vol, uy/Vol, uz/Vol, 0)))

        # subtract the mean from the solution function
        self.up.assign(self.up-self.umean)

    def compute_time_step(self):
        """
        Solve the Navier-Stokes equations.

        Solve the Navier-Stokes equations using finite-difference Euler
        methods at a time step, if the solver diverges, then the run stops.
        Save the functions if self.savesteps=True, then writes to the logfile.

        Returns
        -------
        None.

        """
        # append the current time/MoI to the lists
        self.get_outputs()

        print("-------------------------")
        print("Now Running Cycle {}, t: {:.3e}, Completed {:.2f}%, CFL: {:.3e}"
              .format(self.ind, self.t, 100*self.t/self.end_time,
                      self.CFL(self.dt)))

        try:
            self.solver.solve()
        except:
            print("DIVERGED")
            self.logfile.write("%s: STOPPED DUE TO DIVERGENCE \n" %
                               (self.convert_time(time.time()
                                                  - self.start_time)))
            self.diverged = True
            return

        # if we want to save at steps, save all the functions
        if self.savesteps:
            self.save_funcs(self.u, self.p, self.ftides, self.gravity,
                            self.centrifugal, self.coriolis, self.forcing)

        # write to log
        self.logfile.write(
            "{}: --- Solved Cycle {}, t={:.3e}, Completed {:.2f}%,\
                CFL: {:.3e} --- \n".format(
                self.convert_time(time.time()-self.start_time), self.ind,
                self.t, 100*self.t/self.end_time, self.CFL(self.dt)))

        # update the timestep, for if CFL is too large
        self.update_dt()

        # remove the mean velocity
        self.adjust_u()

        # assign the current solution to the prior solution
        self.u_p_.assign(self.up)

        # update the run index
        self.ind += 1

    def compute_rotation(self, ncyc=3):
        """
        Run the simulation over a few cycles.

        Run the simulation, taking the average velocity for
        use in an trajectory jump. This cycle-jump system significantly
        improves efficiency.

        This assumes a linear change over the velocity, which is generally
        valid for a small-magnitude velocity. Might improve...

        Parameters
        ----------
        ncyc : int, optional
            The number of full rotations to run and average over before
            jumping along the trajectory. The default is 3.

        Returns
        -------
        None.

        """
        # number of Euler steps in ncyc rotations
        nrun = self.nsrot*ncyc

        for i in range(nrun):
            self.update_forces()
            self.size_check()
            self.logfile.write("%s: Forces Computed \n" %
                               (self.convert_time(
                                   time.time()-self.start_time)))

            if self.diverged:
                # stop if diverged
                break

            # solve equations
            self.compute_time_step()

            # add the new velocity to the total
            self.sum_u()

            # move the mesh
            self.move_mesh(self.dt)

            # update the time
            self.t += self.dt

        if not self.diverged:
            # create a function for the average
            self.ucycavg = self.usum.copy(deepcopy=True)

            # average the velocity
            self.ucycavg.assign(self.usum/nrun)

            # delete the current sum, necessary for the sum to exist next cycle
            del self.usum

    def compute_trajectory_step(self):
        """
        Compute a jump along the trajectory, using the average velocity.

        Ensures that the CFL criterion is met, and keeps the
        change in heliocentric distance to a prespecified tolerance. Assists
        in improving efficiency of the simulation. This assumes a linear change
        in the heliocentric distance over the jump, which is generally true.
        Uses the average velocity over the several rotation cycles prior, which
        is more accurate for small velocity.

        Returns
        -------
        None.

        """
        if not self.diverged:
            # gets the heliocentric distance
            dist = self.trajectory(self.t)

            # gets the initial time
            inittime = self.t

            # while the distance has changed by less than rtol percent
            while np.abs(self.trajectory(self.t)/dist-1) < self.rtol:

                # step over a full rotation each time
                self.t += self.nsrot*self.dt

                # check CFL criterion
                if self.CFL(self.t-inittime) >= self.Cmax:
                    # if CFL>1, reverse the time until CFL<1
                    while self.CFL(self.t-inittime) >= self.Cmax:
                        self.t -= self.dt
                    break

            timejump = self.t-inittime  # find the total time change

            # set the velocity to the average
            self.u.vector()[:] = self.ucycavg.vector()[:]

            # move the mesh over this displacement
            self.move_mesh(timejump)

            # save the new output data
            self.get_outputs()

            # write updates to the log file
            print("-------------------------")
            print("{}: Trajectory Jump Completed, Stepped {:.3f} s, t={:.3e}, \
                  {:.3e}%".format(self.convert_time(time.time() -
                  self.start_time), timejump, self.t,
                                  100*(self.t/self.end_time)))
            print("------------------------- \n")
            self.logfile.write("{}: --- Trajectory Jump Completed, Stepped \
                               {: .3f} s, {:.2f}%, CFL: {:.3e}---\n".format(
                               self.convert_time(time.time()-self.start_time),
                               timejump, 100*(self.t/self.end_time),
                               self.CFL(timejump)))

    def run_model(self, nsrot=10, rtol=0.01, period=7.937, Cmax=1.,
                  savesteps=False, data_name='horizons_results.txt',
                  out_funcs=['moment_of_inertia', 'princ_axes']):
        """
        Run the simulation, to avoid cluttering. Helper function.

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
        Cmax : float, optional
            The maximum allowable value of the CFL criterion in the simulation.
            This will be used to dynamically shrink the time steps to ensure
            that the solver steps maintain stability. The default is 1, which
            is a generally accepted good value.
        savesteps : boolean, optional
            Whether or not to save the functions and mesh at each computational
            step. This can quickly overwhelm storage if many runs are used.
            The default is False.
        out_funcs : list, optional
            A list of functions, which will be computed at every time step and
            put into the output. Can have either function objects or strings as
            its components.

        Returns
        -------
        outFrame : pandas.DataFrame
            DataFrame of the output data, with times, and all output data
            computed from functions in out_funcs.

        """
        assert(rtol > 0)

        # go through functions for outputs
        for i, func in enumerate(out_funcs):

            # if a string, assume already be in the class and fetch
            if type(func) is str:
                out_funcs[i] = getattr(self, func)

            # if a function, add to class, then fetch
            else:
                self.add_method(func)
                out_funcs[i]=getattr(self, func.__name__)

        # store functions for outputs
        self.out_funcs = out_funcs

        # initialize parameters
        self.diverged = False
        self.savesteps = savesteps
        self.rtol = rtol
        self.period = period*3600  # converting from hours to seconds
        self.nsrot = nsrot
        self.dt = self.period/nsrot  # in seconds
        self.Cmax = Cmax

        # compute the spline for the trajectory
        self.read_trajectory(data_name=data_name)

        # compute the magnitude of the rotation vector
        mag = 2*np.pi/(self.period)

        # compute the magnitude of the current rotation vector
        vecmag = np.sqrt(np.sum(np.square(self.omega.vector()[:3])))

        # assign the rotation vector magnitude
        self.omega.assign(Constant(tuple(self.omega.vector()[:3]*mag/vecmag)))

        # run the simulation, alternating between the rotation cycles and the
        # trajectory steps until the time reaches the end or the simulation
        # diverges
        while (self.t <= self.end_time) and not (self.diverged):
            self.compute_rotation()
            self.compute_trajectory_step()

        # save the functions at the end of the runs
        self.save_funcs(self.u, self.p, self.ftides, self.gravity,
                        self.centrifugal, self.coriolis, self.forcing)

        # write the output data to an array
        outnames = ["Times"]
        for func in out_funcs:
            fn = func()[1]
            if type(fn) is list:
                outnames.extend(fn)
            else:
                outnames.append(fn)

        # create array for output
        out_data = np.insert(np.array(self.outputs), 0,
                             np.array(self.times+self.mint), axis=1)

        # write output data to dataframe
        outFrame = pd.DataFrame(out_data, columns=outnames)

        # write output dataframe to csv
        outFrame.to_csv(
            'logs/Outputs_{}_{}.csv'.format(self.name,
                                            int(np.log10(float(self.mu)))))

        # write to the logfile
        self.logfile.write("{}: --- Finished, Run Time: {:.3e} --- \n"
                           .format(self.convert_time
                                   (time.time()-self.start_time),
                                   (time.time()-self.start_time)))
        # close the log file
        self.logfile.close()

        return(outFrame)

    def reset_model(self, name=None, a=None, b=None, c=None, mu=None,
                    omegavec=None, rho=None, szscale=None, n=0):
        """
        Reset the model, using the __init__() function.

        All parameters have the same function as in __init__().

        Parameters
        ----------
        Same as __init__(), all optional. If not provided, uses current values.

        Returns
        -------
        None.

        """
        if name is None:
            name = self.name
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        if c is None:
            c = self.c
        if mu is None:
            mu = float(self.mu)
        if omegavec is None:
            omegavec = self.omegavec
        if rho is None:
            rho = float(self.rho)
        if szscale is None:
            szscale = self.szscale

        assert(len(omegavec) == 3)
        assert(szscale >= 1)
        assert(n >= 0)

        # set the name
        self.name = name

        # set the rotation axis
        self.omegavec = omegavec

        # set the principal axes
        self.a = a
        self.b = b
        self.c = c

        # convert the axes from meters to cm
        a *= 100
        b *= 100
        c *= 100

        # set the maximum allowed size
        self.sizecut = szscale*np.max([a, b, c])/2

        self.mu = Constant(mu)  # create a Constant to avoid slowdowns

        # initialize the time, and the number of cycles
        self.t = 0
        self.ind = 0
        self.rho = Constant(rho)  # create a Constant to avoid slowdowns

        # read in mesh, with n refinements
        nmesh = Mesh('3ball%s.xml' % (n))

        # rescale the mesh to the input ellipsoids
        nmesh.coordinates()[:, 0] *= a/2
        nmesh.coordinates()[:, 1] *= b/2
        nmesh.coordinates()[:, 2] *= c/2

        bmesh = BoundaryMesh(nmesh, 'exterior')
        ALE.move(self.mesh, bmesh)

        # compute the spline for the trajectories
        self.get_trajectory()

        # set rotation axis
        self.omega = interpolate(Constant(tuple(omegavec)), self.V)

        # set solution functions to 0
        self.up.assign(Constant((0, 0, 0, 0)))
        self.u_p_.assign(Constant((0, 0, 0, 0)))
