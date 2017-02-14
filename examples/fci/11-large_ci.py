#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Largest CI coefficients
'''

from pyscf import gto, scf, mcscf, fci

mol = gto.Mole()
mol.build(
    verbose = 0,
    atom = 'N, 0., 0., 0. ; N,  0., 0., 1.4',
    basis = 'cc-pvdz',
    symmetry = True,
)
m = scf.RHF(mol)
m.kernel()

mc = mcscf.CASSCF(m, 6, 6)
mc.kernel()

print(' string alpha, string beta, CI coefficients')
for c,ia,ib in mc.fcisolver.large_ci(mc.ci, 6, 6, tol=.05):
    print('  %9s    %9s    %.12f' % (ia, ib, c))
