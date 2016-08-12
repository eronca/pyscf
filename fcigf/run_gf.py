import numpy as np
import scipy.linalg
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import pyscf.ao2mo
import pyscf.lib.logger as logger
import math
import sys
from pyscf.fcigf import compute_gf
#from pyscf.fcigf import run_gf
from compute_gf import *

    
def run_gf_casscf(mc, mf, mol, emc, omega, omega_real, delta, npoints, maxiter, tol_gf, dos_only=False):

    print "\nStarting GF code..."
    print "(c) Alexander Sokolov, 2015\n"
    
    np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

    # Collect general information
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    mo = mc.mo_coeff
    nmo = mo.shape[1]
    nvir = nmo - nocc
    nelec = mol.nelectron
    nelecas = mc.nelecas
    ovlp = mf.get_ovlp()
    
    print "Ground state CASSCF energy: ", emc
    print "Number of electrons:        ", nelec
    print "Number of basis functions:  ", nmo
    print "Number of core orbitals:    ", ncore
    print "Number of external orbitals:", nvir
    print "Number of active orbitals:  ", ncas
    print "Number of active electrons: ", nelecas
    
    print "SCF orbital energies:"
    print mf.mo_energy
    
    # Transform integrals
    h1e_ao = mc.get_hcore()
    h1e_mo = reduce(np.dot, (mo.T, h1e_ao, mo))
    
    v2e = pyscf.ao2mo.incore.full(mf._eri, mo) # v2e has 4-fold symmetry now
    v2e = pyscf.ao2mo.restore(1, v2e, nmo) # to remove 4-fold symmetry, turn v2e to n**4 array
    
    # Compute core Hamiltonian with frozen-core two-electron component
    h1eff = h1e_mo + 2 * np.einsum('pqrr->pq', v2e[:,:,:ncore,:ncore]) - np.einsum('prqr->pq', v2e[:,:ncore,:,:ncore])
    
    # Test CASSCF energy
    fcivec = mc.ci
    eri_cas = pyscf.fci.direct_spin1.absorb_h1e(h1eff[ncore:nocc,ncore:nocc], v2e[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc], ncas, nelecas, .5)
    ham_vec = pyscf.fci.direct_spin1.contract_2e(eri_cas, fcivec, ncas, nelecas )
    
    e_cas = np.dot(np.ravel(fcivec), np.ravel(ham_vec))
    e_fc = 2.0 * np.einsum('pp', h1e_mo[:ncore,:ncore])
    print v2e[:ncore,:ncore,:ncore,:ncore].shape
    v_ppqq = np.einsum('prqq->pr', v2e[:ncore,:ncore,:ncore,:ncore])
    e_ppqq = np.einsum('pp', v_ppqq)
    v_pqpq = np.einsum('pqrq->pr', v2e[:ncore,:ncore,:ncore,:ncore])
    e_pqpq = np.einsum('pp', v_pqpq)
    e_fc += 2 * e_ppqq - e_pqpq
#    e_fc += 2 * np.einsum('ppqq', v2e[:ncore,:ncore,:ncore,:ncore]) - np.einsum('pqpq', v2e[:ncore,:ncore,:ncore,:ncore])
    
    print "Energy(FC)                   %15.10f" % e_fc
    print "Energy(CAS)                  %15.10f" % e_cas 
    print "Total Electronic Energy      %15.10f" % (e_cas + e_fc) 
    
    # Compute Green's function matrix elements
    gf_real_removal = np.zeros((ncas,ncas))
    gf_imag_removal = np.zeros((ncas,ncas))
    gf_real_addition = np.zeros((ncas,ncas))
    gf_imag_addition = np.zeros((ncas,ncas))
    
    # Range
    omega_range = omega[1] - omega[0]
    step = omega_range / npoints
    
    print "\nOmega range: ", omega
    print "Delta: ", delta
    print "Number of points: ", npoints
    print "Step size: ", step
    
    omega_array = np.arange(omega[0], omega[1], step)
    gf_real_trace = []
    gf_imag_trace = []
    density_of_states = []
    density_of_states_mf = []
    se_real_trace = []
    se_imag_trace = []
    
    self_energy = []
    
    #test
    gf_mf_real_trace = []
    
    # Compute density of states only
    if(dos_only):
        for omega_val in omega_array:
            if(omega_real):
                romega = omega_val
                iomega = 0.0
            else:
                romega = 0.0
                iomega = omega_val
        
            print "\nOmega value (real, imag): ", romega, iomega
            
            # Compute CAS Green's function
            for p in range(ncas):
                print "\nGF({0:d},{1:d}):".format(p, p)
                gf_real_addition[p,p], gf_imag_addition[p,p] = gf_addition(romega, iomega, delta, e_cas, p, p, mc, h1eff, v2e, tol_gf, maxiter)
                gf_real_removal[p,p], gf_imag_removal[p,p] = gf_removal(romega, iomega, delta, e_cas, p, p, mc, h1eff, v2e, tol_gf, maxiter)
        
            gf_real_trace.append(np.trace(gf_real_removal + gf_real_addition))
            gf_imag_trace.append(np.trace(gf_imag_removal + gf_imag_addition))
            dos = -(1/math.pi) * np.trace(gf_imag_removal + gf_imag_addition)
            density_of_states.append(dos)
            print "Density of states (interacting):             ", dos
            
            # Compute mean-field Green's function in the AO basis
            gf_mf_ao = gf_mean_field(romega, iomega, delta, mf)
            
            # Transform to CASSCF MO basis
            gf_mf = reduce(np.dot, (mo.T, gf_mf_ao, mo))
        
            print "\nSummary for Omega (real, imag): %10.5f %10.5f" % (romega, iomega)
            dos_mf = -(1/math.pi) * np.trace(gf_mf[ncore:nocc,ncore:nocc].imag)
            tr_real_mf = np.trace(gf_mf[ncore:nocc,ncore:nocc].real)
            density_of_states_mf.append(dos_mf)
            gf_mf_real_trace.append(tr_real_mf)
            print "Density of states (mean-field):             ", dos_mf
        
    #Compute all elements of GF and SF
    else:
        for omega_val in omega_array:
            if(omega_real):
                romega = omega_val
                iomega = 0.0
            else:
                romega = 0.0
                iomega = omega_val
        
            print "\nOmega value (real, imag): ", romega, iomega
            
            # Compute CAS Green's function
            for p in range(ncas):
                for q in range(ncas):
                    print "\nGF({0:d},{1:d}):".format(p, q)
                    gf_real_addition[p,q], gf_imag_addition[p,q] = gf_addition(romega, iomega, delta, e_cas, p, q, mc, h1eff, v2e, tol_gf, maxiter)
                    gf_real_removal[p,q], gf_imag_removal[p,q] = gf_removal(romega, iomega, delta, e_cas, q, p, mc, h1eff, v2e, tol_gf, maxiter)
            
            gf_cas = (gf_real_addition + gf_real_removal) + 1j * (gf_imag_addition + gf_imag_removal)
        
            # Compute mean-field Green's function in the AO basis
            gf_mf_ao = gf_mean_field(romega, iomega, delta, mf)
            
            # Transform to CASSCF MO basis
            gf_mf = reduce(np.dot, (mo.T, gf_mf_ao, mo))

            density_of_states_mf.append(-(1/math.pi) * np.trace(gf_mf[ncore:nocc,ncore:nocc].imag))
            gf_mf_real_trace.append(np.trace(gf_mf[ncore:nocc,ncore:nocc].real))

            print "\nSummary for Omega (real, imag): %10.5f %10.5f" % (romega, iomega)
            gf_cas_trace_real = np.trace(gf_cas).real
            gf_cas_trace_imag = np.trace(gf_cas).imag
            print "Trace of real part:            ", gf_cas_trace_real
            print "Trace of imaginary part:       ", gf_cas_trace_imag
            
            gf_real_trace.append(gf_cas_trace_real)
            gf_imag_trace.append(gf_cas_trace_imag)
            density_of_states.append(-(1/math.pi) * gf_cas_trace_imag)
            
        sys.stdout.flush()
        
    import scipy.integrate
    print "\nIntegral of the mean-field Green's function: ", scipy.integrate.simps(np.array(density_of_states_mf), omega_array)
    print "Integral of the interacting Green's function: ", scipy.integrate.simps(np.array(density_of_states), omega_array)
    sys.stdout.flush()
    
    print "\nCASSCF-GF Done!\n"
    
    # Plot the results
    with open('density_of_states.txt', 'w') as fout:
        fout.write('     Omega          A(Omega)\n')
        for i, om_val in enumerate(omega_array):
            fout.write('%6.3f  %8.4f\n' % (om_val, density_of_states[i]))
    
    with open('density_of_states_mf.txt', 'w') as fout:
        fout.write('     Omega          A(Omega)\n')
        for i, om_val in enumerate(omega_array):
            fout.write('%6.3f  %8.4f\n' % (om_val, density_of_states_mf[i]))
    
    with open('real.txt', 'w') as fout:
        fout.write('     Omega          A(Omega)\n')
        for i, om_val in enumerate(omega_array):
            fout.write('%6.3f  %8.4f\n' % (om_val, gf_real_trace[i]))

    import matplotlib.pyplot as plt
    
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'A($\omega$)')
    plt.plot(omega_array, density_of_states, label='A')
    plt.plot(omega_array, density_of_states_mf, label='A_MF')
    plt.legend()
    plt.show()
     
    #test
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'Re(G)')
    plt.plot(omega_array, gf_real_trace, label='Re(G)')
    plt.plot(omega_array, gf_mf_real_trace, label='Re(G)_MF')
    plt.legend()
    plt.show()
    
   
