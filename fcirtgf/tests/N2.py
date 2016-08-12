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
        ['N', ( 0., 0.    ,  -r/2   )],
        ['N', ( 0., 0.    ,  r/2)],]
    mol.basis = {'N': 'sto-3g'}
    mol.build()
    
    mf = pyscf.scf.RHF(mol)
    mf.conv_threshold = 1e-10
    ehf = mf.scf()
    print "SCF energy:", ehf, "\n"

    mc = pyscf.mcscf.CASCI(mf, 10, 14)
    mc.verbose = 4
#    mc.conv_threshold = 1e-10
#    mc.conv_threshold_grad = 1e-10
#    mc.ah_conv_threshold = 1e-10
#    mc.max_cycle_macro = 100

    emc = mc.casci()[0]

    print "CASSCF energy:", emc

    # User input for GF-CASSCF
    time_real = True
    time_range = (0., 500.)
    npoints = 20000
    # Number of iterations for Green's function computation
    maxiter = 1000
    tol_gf = 1e-4
    
    # Run GF-CASSCF code
    fcirtgf.run_gf.run_gf_casscf(mc, mf, mol, emc, time_range, time_real, npoints, maxiter, tol_gf, dos_only=True)
   
start = 0
stop = 1

for i in range(start, stop):
    r = 1.090105133
    print "R = ", r, " ang"
    run(r)

