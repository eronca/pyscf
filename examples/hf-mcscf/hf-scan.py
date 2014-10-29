#!/usr/bin/env python
import numpy
from pyscf import scf
from pyscf import gto
from pyscf.future import mcscf
import pyscf.future.mcscf.mc1step
import pyscf.future.mcscf.addons

ehf = []
emc = []

def run(b, caslst):
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_hf-%2.1f' % b
    mol.atom = [
        ["F", (0., 0., 0.)],
        ["H", (0., 0., b)],]

    mol.basis = {'F': 'cc-pvdz',
                 'H': 'cc-pvdz',}
    mol.build()
    m = scf.RHF(mol)
    m.chkfile = 'hf-scan.chk'
    m.init_guess = 'chkfile'
    m.set_mo_occ = scf.addons.dynamic_occ(m, 1e-3)
    ehf.append(m.scf())

    mc = mcscf.CASSCF(mol, m, 6, 6)
    mo = mcscf.addons.sort_mo(mc, m.mo_coeff, caslst, 1)
    e1 = mc.mc1step(mo)[0]
    mcscf.addons.map2hf(mc)
    emc.append(e1+mol.nuclear_repulsion())
    return ehf, emc

for b in numpy.arange(4.0,3.1-.01,-.1):
    caslst = [3,4,5,6,8,9]
    run(b, caslst)

for b in numpy.arange(3.0,1.7-.01,-.1):
    caslst = [3,4,5,6,9,10]
    run(b, caslst)

for b in numpy.arange(1.6,0.7-.01,-.1):
    caslst = [3,4,5,6,8,9]
    run(b, caslst)

import os
os.remove('hf-scan.chk')

for b in numpy.arange(0.7,1.61,.1):
    caslst = [3,4,5,6,8,9]
    run(b, caslst)

for b in numpy.arange(1.7,3.01,.1):
    caslst = [3,4,5,6,9,10]
    run(b, caslst)

for b in numpy.arange(3.1,4.01,.1):
    caslst = [3,4,5,6,8,9]
    run(b, caslst)

x = numpy.arange(0.7, 4.01, .1)
ehf1 = ehf[:len(x)]
ehf2 = ehf[len(x):]
emc1 = emc[:len(x)]
emc2 = emc[len(x):]
ehf1.reverse()
emc1.reverse()
with open('hf-scan.txt', 'w') as fout:
    fout.write('     HF-A          CAS(6,6)-A    HF-B          CAS(6,6)-B\n')
    for i, xi in enumerate(x):
        fout.write('%2.1f  %12.8f  %12.8f  %12.8f  %12.8f\n'
                   % (xi, ehf1[i], emc1[i], ehf2[i], emc2[i]))

import matplotlib.pyplot as plt
plt.plot(x, ehf1, label='HF,4.0->0.7')
plt.plot(x, ehf2, label='HF,0.7->4.0')
plt.plot(x, emc1, label='CAS(6,6),4.0->0.7')
plt.plot(x, emc2, label='CAS(6,6),0.7->4.0')
plt.legend()
plt.show()