import argparse
import numpy as np
import matplotlib.pyplot as plt

from RigidBody import *
from LearnRB import LearnRBEnergy
from FileOperations import store_m

def learn(args):
    #Only Energetic regularization enabled.
    #create appropriate solver

    #Create d2E matrix
    d2E = np.array([[1/args.Ix,0,0],\
                    [0,1/args.Iy,0],\
                    [0,0, 1/args.Iz]])


    if args.scheme == "FE":
        solver = RBESeReFE(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha)
    elif args.scheme == "BE":
        raise NotImplementedError("Method not implemented")
    elif args.scheme == "CN":
        solver = RBESeReCN(args.Ix,args.Iy,args.Iz, d2E, args.init_mx,args.init_my,args.init_mz,args.dt,args.alpha)
    else:
        raise NotImplementedError("Method not implemented")

    #Create m vector
    ms = []
    m = np.array([args.init_mx,args.init_my,args.init_mz])

    #Timesteps
    dt = args.dt
    dtau = args.alpha * args.dt


    #Preparing files for output
    if args.simulate:

        with open("m.xyz",'w') as mfile:
            msq = m*m
            store_m(mfile, [0, m[0],m[1],m[2], np.linalg.norm(m), 0.5 * m @ d2E @ m])
            ms.append(m)

            store_each = 1


            #calculate evolution
            for i in range(args.steps-1):

                m = solver.m_new()

                if i % store_each == 0:
                    t = dt * i
                    msq = m*m
                    store_m(mfile, [t, m[0],m[1],m[2], np.linalg.norm(m), 0.5 * m @ d2E @ m])
                    ms.append(m)
                if i % (args.steps/10) == 0 and args.verbose:
                    print(i/args.steps*100, "%")
                    print("|m| = ", np.linalg.norm(m))
                    print("E = ", 0.5 * m @ d2E @ m)


    #Learning part. Initialize learner and use it to infer Energy matrix.

    learner = LearnRBEnergy(args)

    learner.update_exact(args.Ix, args.Iy, args.Iz)
    learner.load_trajectory_from_file("m.xyz", readevery = 1, stopat = 3000)
    #learner.load_trajectory(ms, dt)
    if args.verbose:
        learner.print_trajectory(stopat=10)
    learner.fit(verbose=True)

    if args.verbose:
        print("\n----Comparing matrices: ")
        print("Original energy matrix: \n", np.array_str(np.array(learner.d2E_exact)))
        print("tr_exact = ", np.trace(learner.d2E_exact))
        print("Spectrum exact: ", learner.spectrum_exact())
        print("Moments of inertia exact: ", learner.moments_of_inertia_exact())

        print("\n")
        print("Learned energy matrix: \n", learner.print_d2E())
        print("Spectrum: ", learner.spectrum())
        learner.tr(verbose=True)
        print("Moments of inertia: ", learner.moments_of_inertia())

    if args.normalise:
        print("\n")
        learner.normalize()
        print("Normalized energy matrix: \n", learner.print_d2E())
        print("Spectrum: ", learner.spectrum())
        learner.tr(verbose=True)
        print("Moments of inertia: ", learner.moments_of_inertia())
        print("Residual: ", learner.energy_residual())
        print("Energy Score: ", learner.energy_score())

    learner.predict(args.dt, args.init_mx, args.init_my, args.init_mz)
    return learner.energy_residual(), learner.dtau_error()

if __name__ == "__main__":
    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--regularization", default="ESeRe", type=str, help="Type of regularization. \
        ESeRe for energetic Ehrenfest, Sere for standard Ehrenfest and SeSere for entropic")
    parser.add_argument("--scheme", default="FE", type=str, help="Numerical scheme. FE forward euler, BE backward euler, CN Crank-Nicholson")
    parser.add_argument("--steps", default=5000, type=int, help="Number of simulation steps")
    parser.add_argument("--init_mx", default=1.0, type=float, help="Initial momentum, x component")
    parser.add_argument("--init_my", default=0.3, type=float, help="Initial momentum, x component")
    parser.add_argument("--init_mz", default=0.3, type=float, help="Initial momentum, x component")
    parser.add_argument("--Ix", default=10.0, type=float, help="Ix")
    parser.add_argument("--Iy", default=20.0, type=float, help="Iy")
    parser.add_argument("--Iz", default=40.0, type=float, help="Iz")
    parser.add_argument("--dt", default=0.08, type=float, help="Timestep")
    parser.add_argument("--alpha", default=2.0, type=float, help="Coeficient for dtau. dtau = alpha * dt")
    parser.add_argument("--simulate", default=True, type=bool, help="Simulate new trajectory")
    parser.add_argument("--normalise", default=False, type=bool, help="Normalise energy matrix at the end")
    parser.add_argument("--verbose", default=False, type=bool, help="Print a lot of useful output")

    args = parser.parse_args([] if "__file__" not in globals() else None)

    #Either run only one learning step, or run varying alpha
    learn(args)
    #drawAlphas(args)


#Try to see the learning for varying alpha
def drawAlphas(args):
    with open("alphas",'w') as f:
        alphas = []
        dtaus = []
        E_errs = []
        dtau_errs = []
        for alpha in np.arange(0.0,0.1,0.001):
            args.alpha = alpha
            print(alpha)
            energy_err, dtau_error = learn(args)
            alphas.append(alpha)
            dtaus.append(args.dt*alpha)
            E_errs.append(energy_err)
            dtau_errs.append(dtau_error)
            f.write(str(alpha) + '\t' + str(energy_err) + '\t' + str(dtau_error) + '\n')

        #Plot how the errors look like
        ax1 = plt.subplot(211)
        ax1.plot(dtaus, E_errs)
        ax1.set_ylabel(r'$(E_{learned}-E_{exact})^2$')
        ax2 = plt.subplot(212, sharex=ax1)
        ax2.plot(dtaus, dtau_errs)
        ax2.set_xlabel(r'$\tau$')
        ax2.set_ylabel(r'$(\tau_{exact}-\tau_{learned})^2$')

        plt.show()
