 #This file contains the forward Euler solver for the self-regularized rigid body motion and the Crank-Nicolson solver for the energetic self-regularization of rigid body motion
#Author: Michal Pavelka; pavelka@karlin.mff.cuni.cz

from scipy.optimize import fsolve
from math import *
import numpy as np

class RigidBody(object): #Parent Rigid body class
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, T=100, verbose = False):
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz

        self.d2E= d2E

        if Iz > 0 and Iy > 0 and Iz > 0:
            self.Jx = 1/Iz - 1/Iy
            self.Jy = 1/Ix - 1/Iz
            self.Jz = 1/Iy - 1/Ix

        self.mx = mx
        self.my = my
        self.mz = mz

        self.mx0 = mx
        self.my0 = my
        self.mz0 = mz

        self.dt = dt
        self.tau = dt*alpha

        self.hbar = 1.0545718E-34 #reduced Planck constant [SI]
        self.rho = 8.92E+03 #for copper
        self.myhbar = self.hbar * self.rho #due to rescaled mass
        self.kB = 1.38064852E-23 #Boltzmann constant
        self.umean = 4600 #mean sound speed in the low temperature solid (Copper) [SI]
        self.Einconst = pi**2/10 * pow(15/(2* pi**2), 4.0/3) * self.hbar * self.umean * pow(self.kB, -4.0/3) #Internal energy prefactor, Characterisitic volume = 1
        if verbose:
            print("Internal energy prefactor = ", self.Einconst)

        self.sin = self.ST(T) #internal entropy
        if verbose:
            print("Internal entropy set to Sin = ", self.sin, " at T=",T," K")

        self.Ein_init = 1
        self.Ein_init = self.Ein()
        self.sin_init = self.sin
        if verbose:
            print("Initial total energy = ", self.Etot())

        if verbose:
            print("RB set up.")

    def energy_x(self):
        return 0.5*self.mx*self.mx/self.Ix

    def energy_y(self):
        return 0.5*self.my*self.my/self.Iy

    def energy_z(self):
        return 0.5*self.mz*self.mz/self.Iz

    def energy(self):#returns kinetic energy
        return 0.5*(self.mx*self.mx/self.Ix+self.my*self.my/self.Iy+self.mz*self.mz/self.Iz)

    def omega_x(self):
        return self.mx/self.Ix

    def omega_y(self):
        return self.my/self.Iy

    def omega_z(self):
        return self.mz/self.Iz

    def m2(self):#returns m^2
        return self.mx*self.mx+self.my*self.my+self.mz*self.mz

    def mx2(self):#returns mx^2
        return self.mx*self.mx

    def my2(self):#returns my^2
        return self.my*self.my

    def mz2(self):#returns mz^2
        return self.mz*self.mz

    def m_magnitude(self):#returns |m|
        return sqrt(self.m2())

    def Ein(self):#returns normalized internal energy
        #return exp(2*(self.sin-1))/self.Iz
        return self.Einconst*pow(self.sin,4.0/3)/self.Ein_init

    def Ein_s(self): #returns normalized derivative of internal energy with respect to entropy (inverse temperature)
        #return 2*exp(2*(self.sin-1))/self.Iz
        return self.Einconst*4.0/3*pow(self.sin, 1.0/3) / self.Ein_init

    def ST(self, T): #returns entropy of a Copper body with characteristic volume equal to one (Debye), [T] = K
        return 2 * pi**2/15 * self.kB * (self.kB/self.hbar *T/self.umean)**3

    def Etot(self):#returns normalized total energy
        return self.energy() + self.Ein()

    def Sin(self): #returns normalized internal entorpy
        return self.sin/self.sin_init

    def S_x(self):#kinetic entropy for rotation around x, beta = 1/4Iz
        m2 = self.m2()
        return -m2/self.Ix - 0.5*0.25/self.Iz*(m2-self.mx0*self.mx0)**2

    def S_z(self):#kinetic entropy for rotation around z
        m2 = self.m2()
        return -m2/self.Iz - 0.5*0.25/self.Iz*(m2-self.mz0*self.mz0)**2

    def Phi_x(self): #Returns the Phi potential for rotation around the x-axis
        return self.energy() + self.S_x()

    def Phi_z(self):
        return self.energy() + self.S_z()

class RBESeReCN(RigidBody):#E-SeRe with Crank Nicolson
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha):
        super(RBESeReCN, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha)

    def f(self, mNew):#defines the function f zero of which is sought

        mOld = [self.mx, self.my, self.mz]
        dot = np.dot(self.d2E, mOld)
        ham = np.cross(mOld, dot)

        #regularized part t
        dotR = np.dot(self.d2E, ham)
        reg  = np.cross(dotR, mOld)

        #Hamiltionian part t+1
        dotNNew = np.dot(self.d2E, mNew)
        hamNew = np.cross(mNew, dotNNew)

        #regularized part t+1
        dotRNew = np.dot(self.d2E, hamNew)
        regNew  = np.cross(dotRNew, mNew)

        res = mNew - mOld - self.dt/2*(ham + hamNew) + self.dt*self.tau/4*(reg + regNew)

        return (res[0], res[1], res[2])

    def m_new(self, with_entropy = False): #return new m and update RB
        #calculate
        m_new = fsolve(self.f, (self.mx, self.my, self.mz))

        #update
        self.mx = m_new[0]
        self.my = m_new[1]
        self.mz = m_new[2]

        return m_new

class RBESeReFE(RigidBody):#SeRe forward Euler
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha):
        super(RBESeReFE, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha)

    def m_new(self, with_entropy = False):

        #Construct mOld
        mOld = [self.mx, self.my, self.mz]

        #calculate
        dot = np.dot(self.d2E, mOld)
        ham = np.cross(mOld, dot)

        #regularized part t
        dotR = np.dot(self.d2E, ham)
        reg  = np.cross(dotR, mOld)

        m = mOld + self.dt*ham - self.dt*self.tau/2*reg

        #update
        self.mx = m[0]
        self.my = m[1]
        self.mz = m[2]

        if with_entropy: #calculate new entropy using explicit forward Euler
            sin_new = self.sin+ 0.5*(tau-dt)*dt/self.Ein_s() * ((my*mz*Jx)**2/Ix + (mz*mx*Jy)**2/Iy + (mx*my*Jz)**2/Iz)
            self.sin = sin_new

        return m
