#!/usr/bin/env python

import mc1step
import mc1step_symm
import casci
import casci_symm
import addons
import pyscf.fci

def CASSCF(mol, *args, **kwargs):
    if mol.symmetry:
        mc = mc1step_symm.CASSCF(mol, *args, **kwargs)
        mc.fcisolver = pyscf.fci.solver(mol)
    else:
        mc = mc1step.CASSCF(mol, *args, **kwargs)
    return mc

def CASCI(mol, *args, **kwargs):
    if mol.symmetry:
        mc = casci_symm.CASCI(mol, *args, **kwargs)
        mc.fcisolver = pyscf.fci.solver(mol)
    else:
        mc = casci.CASCI(mol, *args, **kwargs)
    return mc