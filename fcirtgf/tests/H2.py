import numpy
import scipy.linalg
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import pyscf.ao2mo
import pyscf.lib.logger as logger
import math
from pyscf import fcirtgf
#from compute_gf import *
#from fcigf.rungf import *

def run(r):
    mol = pyscf.gto.Mole()
    mol.verbose = 0
    mol.atom = [
        ['H', ( 0., 0.    ,  0.0   )],
        ['H', ( 0., 0.    ,  r)],]
    mol.basis = {'H': '6-31G'}
    mol.build()
    
    mf = pyscf.scf.RHF(mol)
    mf.conv_threshold = 1e-10
    ehf = mf.scf()
    print "SCF energy:", ehf, "\n"

    mc = pyscf.mcscf.CASCI(mf, 4, 2)
    mc.verbose = 4

    emc = mc.casci()[0]

    print "CASSCF energy:", emc

    # User input for GF-CASSCF
    time_real = True
    time_range = (0., 3000.)
    npoints = 30000
    # Number of iterations for Green's function computation
    maxiter = 1000
    tol_gf = 1e-4

    eta = 0.05
    
    # Run GF-CASSCF code
    fcirtgf.run_gf.run_gf_casscf(mc, mf, mol, emc, time_range, time_real, eta, npoints)
   
start = 0
stop = 1

for i in range(start, stop):
    r = 1.1
    print "R = ", r, " ang"
    run(r)

