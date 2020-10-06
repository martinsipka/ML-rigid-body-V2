import numpy as np
import math
from RigidBody import *
from FileOperations import store_m



from scipy.optimize import curve_fit

class LearnRBEnergy(object):
    def __init__(self, args):
        self.d2E = [[0,0,0],[0,0,0],[0,0,0]]
        self.dtau = 0
        self.dt = 0
        self.d2E_exact = [[0,0,0],[0,0,0],[0,0,0]]
        self.dtau_exact = args.dt*args.alpha
        self.trajectory = []
        self.args = args

    def load_trajectory_from_file(self, file_name, readevery=1, stopat=0):
        #Assumed format: t, mx, my, mz \n
        traj_file = open(file_name,"r")
        lines = traj_file.readlines()
        if stopat == 0:
            stopat = len(lines) #read the whole file

        line = [float(j) for j in lines[0].split(" ")]
        self.trajectory = [line[:4]] #contains entry at t=0
        for i in range(1,min(stopat,len(lines))):
            if (i+1) % readevery == 0:
                line = [float(j) for j in lines[i].split(" ")]
                self.trajectory.append(line[:4])

        if self.args.verbose:
            print("Trajectory shape: ", np.array(self.trajectory).shape)
            print("Number of points: ", len(self.trajectory))

    def load_trajectory(self, trajectory, dt, readevery = 1, stopat = 0):
        #Assumed format: mx, my, mz \n

        if stopat == 0 or stopat > len(trajectory):
            self.trajectory = trajectory[:end:readevery]

        self.trajectory = trajectory[:stopat:readevery]

        if self.args.verbose:
            print("Trajectory shape: ", np.array(self.trajectory).shape)
            print("Number of points: ", len(self.trajectory))

    def print_trajectory(self, stopat = 0):
        if stopat == 0:
            stopat = len(self.trajectory)

        print("Prining the trajectory")
        for i in range(min(len(self.trajectory),stopat)):
            print(self.trajectory[i])

    #Nonlinear fit Forward Euler. Using Energetic Ehrenfest regularization
    def fitFE(self, m, d2E11, d2E12, d2E13, d2E22, d2E23, d2E33, dtau):
        d2E = [[d2E11, d2E12, d2E13],[d2E12, d2E22, d2E23], [d2E13, d2E23, d2E33]]
        m = m[0]
        #mNew = m[1]
        #Hamiltionian part
        dot = np.dot(m, np.transpose(d2E))
        ham = np.cross(m, dot, axisa=1, axisb=1)

        #regularized part
        dotR = np.dot(ham, np.transpose(d2E))
        reg  = np.cross(dotR, m, axisa=1, axisb=1)


        res = self.dt*ham - self.dt*dtau/2*reg
        return res.ravel()


    #Nonlinear fit Crank Nicholson. Using Energetic Ehrenfest regularization
    def fitCN(self, m, d2E11, d2E12, d2E13, d2E22, d2E23, d2E33, dtau):
        d2E = [[d2E11, d2E12, d2E13],[d2E12, d2E22, d2E23], [d2E13, d2E23, d2E33]]
        mOld = m[0]
        mNew = m[1]
        #Hamiltionian part t
        dot = np.dot(mOld, np.transpose(d2E))
        ham = np.cross(mOld, dot, axisa=1, axisb=1)

        #regularized part t
        dotR = np.dot(ham, np.transpose(d2E))
        reg  = np.cross(dotR, mOld, axisa=1, axisb=1)

        #Hamiltionian part t+1
        dotNNew = np.dot(mNew, np.transpose(d2E))
        hamNew = np.cross(mNew, dotNNew, axisa=1, axisb=1)

        #regularized part t+1
        dotRNew = np.dot(hamNew, np.transpose(d2E))
        regNew  = np.cross(dotRNew, mNew, axisa=1, axisb=1)

        res = self.dt/2*(ham + hamNew) - self.dt*dtau/4*(reg + regNew)
        return res.ravel()

    def fit(self, verbose = False):
        xOld = []
        xNew = []
        y = []
        for i in range(len(self.trajectory)-1):
            y.append([self.trajectory[i+1][j]-self.trajectory[i][j] for j in range(1,4)])

            self.dt = self.trajectory[i+1][0] - self.trajectory[i][0]

            xOld.append(self.trajectory[i][1:])
            xNew.append(self.trajectory[i+1][1:])

        x = (xOld, xNew)
        if self.args.scheme == "FE":
            function = self.fitFE
        elif self.args.scheme == "BE":
            raise NotImplementedError("Method not implemented")
        elif self.args.scheme == "CN":
            function = self.fitCN
        else:
            raise NotImplementedError("Method not implemented")


        try:
            d2E, _ = curve_fit(function, x, np.array(y).ravel(), [1,1,1,1,1,1,1])
        except RuntimeError:
            print("Did not converge!")
            d2E = [float("NAN"), float("NAN"), float("NAN"),float("NAN"),
             float("NAN"), float("NAN"), float("NAN")]

        [E11, E12, E13, E22, E23, E33, dtau] = d2E
        if self.args.verbose:
            print("dt: ", self.dt)
            print("learned dtau is: ", dtau)
            print(d2E)
        self.d2E = [[E11, E12, E13], [E12, E22, E23], [E13, E23, E33]]


    def tr(self, verbose = False):
        tr = np.trace(self.d2E)
        if verbose:
            print("tr d2E = ", tr)
        return tr

    def det(self):
        det = np.linalg.det(self.d2E)
        if self.args.verbose:
            print("det d2E = ", det)
        return det


    def print_d2E(self):
        print("d2E = \n", np.array_str(np.array(self.d2E)))


    def spectrum(self, verbose = False):
        spectrum, eigvec = np.linalg.eig(self.d2E)
        if verbose:
            print("Eigenvalues(d2E) = ", spectrum)
        return spectrum

    def spectrum_exact(self, verbose = False):
        spectrum, eigvec = np.linalg.eig(self.d2E_exact)
        if verbose:
            print("Eigenvalues(d2E) = ", spectrum)
        return spectrum

    def predict(self, dt, m1, m2, m3):
        args = self.args
        m = np.array([m1,m2,m3])
        if args.scheme == "FE":
            solver = RBESeReFE(0,0,0, self.d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha)
        elif args.scheme == "BE":
            raise NotImplementedError("Method not implemented")
        elif args.scheme == "CN":
            solver = RBESeReCN(0,0,0, self.d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha)
        else:
            raise NotImplementedError("Method not implemented")

        with open("lm.xyz",'w') as mfile:
            msq = m*m
            store_m(mfile, [0, m[0],m[1],m[2], np.linalg.norm(m), 0.5 * m @ self.d2E @ m])

            store_each = 1

            #calculate evolution
            for i in range(args.steps-1):

                m = solver.m_new()

                if i % store_each == 0:
                    t = dt * i
                    msq = m*m
                    store_m(mfile, [t, m[0],m[1],m[2], np.linalg.norm(m), 0.5 * m @ self.d2E @ m])
                if i % (args.steps/10) == 0 and args.verbose:
                    print(i/args.steps*100, "%")
                    print("|m| = ", np.linalg.norm(m))
                    print("E = ", 0.5 * m @ self.d2E @ m)

    def normalize(self):
        #SO3 dynamics is not canonical and has Casimirs. Therefore, we can only learn energy up to the Casimirs.
        #Casimirs are multiples of |m|^2.
        #This method normalized the learned d2E by adding a the unit matrix multiplied by a constant\
        #so that it matches the exact energy.

        tr = np.trace(self.d2E)
        tr_exact = np.trace(self.d2E_exact)

        self.d2E[0][0] += (tr_exact-tr)/3.0
        self.d2E[1][1] += (tr_exact-tr)/3.0
        self.d2E[2][2] += (tr_exact-tr)/3.0

    def energy_residual(self): #sum of squares of differences between entries of the exact and learned matrices
        result = 0
        for i in range(3):
            for j in range(3):
                result += (self.d2E[i][j]-self.d2E_exact[i][j])**2
        return result

    def energy_score(self): #1-u/v, where u is the energy residual and v is square of the exact matrix
        #Close to 1.0 is good, negative is bad
        if math.isnan(self.energy_residual()):
            return 0

        square_exact = 0
        for i in range(3):
            for j in range(3):
                square_exact += (self.d2E_exact[i][j])**2
        return 1-self.energy_residual()/square_exact

    def dtau_error(self):
        return (self.dtau - self.dtau_exact)**2

    def moments_of_inertia(self):
        spectrum = self.spectrum()
        return [1/spectrum[0], 1/spectrum[1], 1/spectrum[2]]

    def moments_of_inertia_exact(self):
        spectrum = self.spectrum_exact()
        return [1/spectrum[0], 1/spectrum[1], 1/spectrum[2]]

    def update_exact(self, Ix, Iy, Iz):
        self.d2E_exact= [[1/Ix, 0, 0],[0, 1/Iy, 0], [0, 0, 1/Iz]]
