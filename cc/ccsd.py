#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import time
import tempfile
from functools import reduce
import numpy
import h5py
from pyscf import lib
from pyscf.lib import logger
import pyscf.ao2mo
from pyscf.cc import _ccsd

BLKMIN = 4

# t2 as ijab

# default max_memory = 2000 MB
def kernel(cc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6,
           max_memory=2000, verbose=logger.INFO):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(cc.stdout, verbose)

    if t1 is None and t2 is None:
        t1, t2 = cc.init_amps(eris)[1:]
    elif t1 is None:
        t1 = numpy.zeros((nocc,nvir))
    elif t2 is None:
        t2 = cc.init_amps(eris)[2]

    cput1 = cput0 = (time.clock(), time.time())
    nocc, nvir = t1.shape
    eold = 0
    eccsd = 0
    if cc.diis:
        adiis = lib.diis.DIIS(cc, cc.diis_file)
        adiis.space = cc.diis_space
    else:
        adiis = lambda t1,t2,*args: (t1,t2)

    conv = False
    for istep in range(max_cycle):
        t1new, t2new = cc.update_amps(t1, t2, eris, max_memory)
        normt = numpy.linalg.norm(t1new-t1) + numpy.linalg.norm(t2new-t2)
        t1, t2 = t1new, t2new
        t1new = t2new = None
        if cc.diis:
            t1, t2 = cc.diis(t1, t2, istep, normt, eccsd-eold, adiis)
        eold, eccsd = eccsd, energy(cc, t1, t2, eris)
        log.info('istep = %d  E(CCSD) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                 istep, eccsd, eccsd - eold, normt)
        cput1 = log.timer('CCSD iter', *cput1)
        if abs(eccsd-eold) < tol and normt < tolnormt:
            conv = True
            break
    log.timer('CCSD', *cput0)
    return conv, eccsd, t1, t2


def update_amps(cc, t1, t2, eris, max_memory=2000):
    time0 = time.clock(), time.time()
    log = logger.Logger(cc.stdout, cc.verbose)
    nocc, nvir = t1.shape
    nov = nocc*nvir
    fock = eris.fock
    t1new = numpy.zeros_like(t1)
    t2new = numpy.zeros_like(t2)

#** make_inter_F
    fov = fock[:nocc,nocc:].copy()

    foo = fock[:nocc,:nocc].copy()
    foo[range(nocc),range(nocc)] = 0
    foo += .5 * numpy.einsum('ia,ja->ij', fock[:nocc,nocc:], t1)

    fvv = fock[nocc:,nocc:].copy()
    fvv[range(nvir),range(nvir)] = 0
    fvv -= .5 * numpy.einsum('ia,ib->ab', t1, fock[:nocc,nocc:])

    #: woooo = numpy.einsum('la,ikja->ikjl', t1, eris.ooov)
    eris_ooov = _cp(eris.ooov)
    foo += numpy.einsum('kc,jikc->ij', 2*t1, eris_ooov)
    foo += numpy.einsum('kc,jkic->ij',  -t1, eris_ooov)
    woooo = lib.dot(eris_ooov.reshape(-1,nvir), t1.T).reshape((nocc,)*4)
    woooo = lib.transpose_sum(woooo.reshape(nocc*nocc,-1), inplace=True)
    woooo += _cp(eris.oooo).reshape(nocc**2,-1)
    woooo = _cp(woooo.reshape(nocc,nocc,nocc,nocc).transpose(0,2,1,3))
    eris_ooov = None
    time1 = log.timer_debug1('woooo', *time0)

    unit = _memory_usage_inloop(nocc, nvir)*1e6/8
    max_memory = max_memory - lib.current_memory()[0]
    blksize = max(BLKMIN, int(max_memory*.95e6/8/unit))
    log.debug1('block size = %d, nocc = %d is divided into %d blocks',
               blksize, nocc, int((nocc+blksize-1)//blksize))

    for p0, p1 in prange(0, nocc, blksize):
# ==== read eris.ovvv ====
        eris_ovvv = _cp(eris.ovvv[p0:p1])
        eris_ovvv = _ccsd.unpack_tril(eris_ovvv.reshape((p1-p0)*nvir,-1))
        eris_ovvv = eris_ovvv.reshape(p1-p0,nvir,nvir,nvir)

        fvv += numpy.einsum('kc,kcba->ab', 2*t1[p0:p1], eris_ovvv)
        fvv += numpy.einsum('kc,kbca->ab',  -t1[p0:p1], eris_ovvv)

    #: tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
    #: tmp = numpy.einsum('ijcd,kdcb->ijbk', tau, eris.ovvv)
    #: t2new += numpy.einsum('ka,ijbk->ijba', -t1, tmp)
        #: eris_vvov = eris_ovvv.transpose(1,2,0,3).copy()
        eris_vvov = _cp(eris_ovvv.transpose(2,1,0,3).reshape(nvir*nvir,-1))
        tmp = numpy.empty((nocc,nocc,p1-p0,nvir))
        taubuf = numpy.empty((blksize,nocc,nvir,nvir))
        for j0, j1 in prange(0, nocc, blksize):
            tau = make_tau(t2[j0:j1], t1[j0:j1], t1, 1, out=taubuf[:j1-j0])
            lib.dot(tau.reshape(-1,nvir*nvir), eris_vvov, 1,
                    tmp[j0:j1].reshape((j1-j0)*nocc,-1), 0)
        tmp = _cp(tmp.transpose(0,1,3,2).reshape(-1,p1-p0))
        lib.dot(tmp, t1[p0:p1], -1, t2new.reshape(-1,nvir), 1)
        tau = tmp = eris_vvov = None
        #==== mem usage blksize*(nvir**3*2+nvir*nocc**2*2)

    #: wOVov += numpy.einsum('iabc,jc->ijab', eris.ovvv, t1)
    #: wOVov -= numpy.einsum('jbik,ka->jiba', eris.ovoo, t1)
    #: t2new += woVoV.transpose()
        #: wOVov = -numpy.einsum('jbik,ka->ijba', eris.ovoo[p0:p1], t1)
        tmp = _cp(eris.ovoo[p0:p1].transpose(2,0,1,3))
        wOVov = lib.dot(tmp.reshape(-1,nocc), t1, -1)
        tmp = None
        wOVov = wOVov.reshape(nocc,p1-p0,nvir,nvir)
        #: wOVov += numpy.einsum('iabc,jc->jiab', eris_ovvv, t1)
        lib.dot(t1, eris_ovvv.reshape(-1,nvir).T, 1, wOVov.reshape(nocc,-1), 1)
        t2new[p0:p1] += wOVov.transpose(1,0,2,3)

        eris_ooov = _cp(eris.ooov[p0:p1])
        #: woVoV = numpy.einsum('ka,ijkb->ijba', t1, eris.ooov[p0:p1])
        #: woVoV -= numpy.einsum('jc,icab->ijab', t1, eris_ovvv)
        woVoV = lib.dot(_cp(eris_ooov.transpose(0,1,3,2).reshape(-1,nocc)), t1)
        woVoV = woVoV.reshape(p1-p0,nocc,nvir,nvir)
        for i in range(eris_ovvv.shape[0]):
            lib.dot(t1, eris_ovvv[i].reshape(nvir,-1), -1,
                    woVoV[i].reshape(nocc,-1), 1)

    #: theta = t2.transpose(0,1,3,2) * 2 - t2
    #: t1new += numpy.einsum('ijcb,jcba->ia', theta, eris.ovvv)
        theta = numpy.empty((p1-p0,nocc,nvir,nvir))
        for i in range(p1-p0):
            theta[i] = t2[p0+i].transpose(0,2,1) * 2
            theta[i] -= t2[p0+i]
            lib.dot(_cp(theta[i].transpose(0,2,1).reshape(nocc,-1)),
                    eris_ovvv[i].reshape(-1,nvir), 1, t1new, 1)
        eris_ovvv = None
        time2 = log.timer_debug1('ovvv [%d:%d]'%(p0, p1), *time1)
        #==== mem usage blksize*(nvir**3+nocc*nvir**2*4)

# ==== read eris.ovov ====
        eris_ovov = _cp(eris.ovov[p0:p1])
        #==== mem usage blksize*(nocc*nvir**2*4)

        for i in range(p1-p0):
            t2new[p0+i] += eris_ovov[i].transpose(1,0,2) * .5

        fov[p0:p1] += numpy.einsum('kc,iakc->ia', t1, eris_ovov) * 2
        fov[p0:p1] -= numpy.einsum('kc,icka->ia', t1, eris_ovov)

    #: theta = t2.transpose(1,0,2,3) * 2 - t2
    #: t1new += numpy.einsum('jb,ijba->ia', fov, theta)
    #: t1new -= numpy.einsum('kijb,kjba->ia', eris_ooov, theta)
        t1new += numpy.einsum('jb,jiab->ia', fov[p0:p1], theta)
        #: t1new -= numpy.einsum('kijb,kjab->ia', eris.ooov[p0:p1], theta)
        lib.dot(_cp(eris_ooov.transpose(1,0,2,3).reshape(nocc,-1)),
                theta.reshape(-1,nvir), -1, t1new, 1)
        eris_ooov = None

    #: wOVov += eris.ovov.transpose(0,1,3,2)
    #: theta = t2.transpose(1,0,2,3) * 2 - t2
    #: tau = theta - numpy.einsum('ic,kb->ikcb', t1, t1*2)
    #: wOVov += .5 * numpy.einsum('jakc,ikcb->jiba', eris.ovov, tau)
    #: wOVov -= .5 * numpy.einsum('jcka,ikcb->jiba', eris.ovov, t2)
    #: t2new += numpy.einsum('ikca,kjbc->ijba', theta, wOVov)
        theta = _cp(theta.transpose(0,3,1,2))
        wOVov = _cp(wOVov.transpose(0,3,1,2))
        eris_OVov = lib.transpose(eris_ovov.reshape(-1,nov)).reshape(nocc,nvir,-1,nvir)
        eris_OvoV = _cp(eris_OVov.transpose(0,3,2,1))
        wOVov += eris_OVov
        for j0, j1 in prange(0, nocc, blksize):
            t2iajb = t2[j0:j1].transpose(0,2,1,3).copy()
            #: wOVov[j0:j1] -= .5 * numpy.einsum('iack,jkbc->jbai', eris_ovov, t2)
            lib.dot(t2iajb.reshape(-1,nov), eris_OvoV.reshape(nov,-1),
                    -.5, wOVov[j0:j1].reshape((j1-j0)*nvir,-1), 1)
            tau, t2iajb = t2iajb, None
            for i in range(j1-j0):
                tau[i] *= 2
                tau[i] -= t2[j0+i].transpose(2,0,1)
                tau[i] -= numpy.einsum('a,jb->bja', t1[j0+i]*2, t1)
            #: wOVov[j0:j1] += .5 * numpy.einsum('iakc,jbkc->jbai', eris_ovov, tau)
            lib.dot(tau.reshape(-1,nov), eris_OVov.reshape(nov,-1),
                    .5, wOVov[j0:j1].reshape((j1-j0)*nvir,-1), 1)

            #theta = t2[p0:p1] * 2 - t2[p0:p1].transpose(0,1,3,2)
            #: t2new[j0:j1] += numpy.einsum('iack,jbck->jiba', theta, wOVov[j0:j1])
            tmp, tau = tau, None
            lib.dot(wOVov[j0:j1].reshape((j1-j0)*nvir,-1), theta.reshape(-1,nov),
                    1, tmp.reshape(-1,nov))
            for i in range(j1-j0):
                t2new[j0+i] += tmp[i].transpose(1,0,2)
            tmp = None
            #==== mem usage blksize*(nocc*nvir**2*8)
        theta = wOVov = eris_OvoV = eris_OVov = None
        time2 = log.timer_debug1('wOVov [%d:%d]'%(p0, p1), *time2)
        #==== mem usage blksize*(nocc*nvir**2*2)

    #: fvv -= numpy.einsum('ijca,ibjc->ab', theta, eris.ovov)
    #: foo += numpy.einsum('iakb,jkba->ij', eris.ovov, theta)
        for i in range(p1-p0):
            tau = numpy.einsum('a,jb->jab', t1[p0+i]*.5, t1)
            tau += t2[p0+i]
            theta = tau.transpose(0,2,1) * 2
            theta -= tau
            lib.dot(_cp(eris_ovov[i].transpose(1,2,0)).reshape(nocc,-1),
                    theta.reshape(nocc,-1).T, 1, foo, 1)
            lib.dot(theta.reshape(-1,nvir).T,
                    eris_ovov[i].reshape(nvir,-1).T, -1, fvv, 1)
        tau = theta = None

# ==== read eris.oovv ====
        eris_oovv = _cp(eris.oovv[p0:p1])
        #==== mem usage blksize*(nocc*nvir**2*3)

        #:tmp = numpy.einsum('ic,jkbc->jibk', t1, eris_oovv)
        #:t2new[p0:p1] += numpy.einsum('ka,jibk->jiab', -t1, tmp)
        #:tmp = numpy.einsum('ic,jbkc->jibk', t1, eris_ovov)
        #:t2new[p0:p1] += numpy.einsum('ka,jibk->jiba', -t1, tmp)
        for j in range(p1-p0):
            tmp = lib.dot(t1, eris_oovv[j].reshape(-1,nvir).T)
            tmp = _cp(tmp.reshape(nocc,nocc,nvir).transpose(0,2,1))
            t2new[p0+j] += lib.dot(tmp.reshape(-1,nocc), t1,
                                   -1).reshape(nocc,nvir,nvir).transpose(0,2,1)
            lib.dot(t1, eris_ovov[j].reshape(-1,nvir).T, 1, tmp.reshape(nocc,-1))
            lib.dot(tmp.reshape(-1,nocc), t1, -1, t2new[p0+j].reshape(-1,nvir), 1)
        tmp = None

    #: g2 = 2 * eris.oOVv - eris.oovv
    #: t1new += numpy.einsum('jb,ijba->ia', t1, g2)
        t1new[p0:p1] += numpy.einsum('jb,iajb->ia', 2*t1, eris_ovov)
        t1new[p0:p1] += numpy.einsum('jb,ijba->ia',  -t1, eris_oovv)

    #: tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
    #: woooo += numpy.einsum('ijba,klab->ijkl', eris.oOVv, tau)
    #: woVoV -= eris.oovv
    #: tau = .5*t2 + numpy.einsum('ia,jb->ijab', t1, t1)
    #: woVoV += numpy.einsum('ka,ijkb->ijab', t1, eris.ooov)
    #: woVoV += numpy.einsum('jkca,ikbc->ijab', tau, eris.oOVv)
        woVoV -= eris_oovv
        woVoV = woVoV.transpose(1,3,0,2).copy()
        eris_oVOv = _cp(eris_ovov.transpose(0,3,2,1))
        eris_oOvV = _cp(eris_ovov.transpose(0,2,1,3))
        #==== mem usage blksize*(nocc*nvir**2*4)

        taubuf = numpy.empty((blksize,nocc,nvir,nvir))
        for j0, j1 in prange(0, nocc, blksize):
            tau = make_tau(t2[j0:j1], t1[j0:j1], t1, 1, out=taubuf[:j1-j0])
            #: woooo[p0:p1,:,j0:j1] += numpy.einsum('ijab,klab->ijkl', eris_oOvV, tau)
            lib.numpy_helper._dgemm('N', 'T', (p1-p0)*nocc, (j1-j0)*nocc, nvir*nvir,
                                    eris_oOvV.reshape(-1,nvir*nvir),
                                    tau.reshape(-1,nvir*nvir),
                                    woooo[p0:p1].reshape(-1,nocc*nocc), 1, 1,
                                    0, 0, j0*nocc)
            for i in range(j1-j0):
                tau[i] -= t2[j0+i] * .5
            #: woVoV[j0:j1] += numpy.einsum('jkca,ickb->jiab', tau, eris_ovov)
            lib.dot(_cp(tau.transpose(0,3,1,2).reshape(-1,nov)),
                    eris_oVOv.reshape(-1,nov).T,
                    1, woVoV[j0:j1].reshape((j1-j0)*nvir,-1), 1)
            #==== mem usage blksize*(nocc*nvir**2*6)
        time2 = log.timer_debug1('woVoV [%d:%d]'%(p0, p1), *time2)

        tau = make_tau(t2[p0:p1], t1[p0:p1], t1, 1, out=taubuf[:p1-p0])
        #: t2new += .5 * numpy.einsum('klij,klab->ijab', woooo[p0:p1], tau)
        lib.dot(woooo[p0:p1].reshape(-1,nocc*nocc).T, tau.reshape(-1,nvir*nvir),
                .5, t2new.reshape(nocc*nocc,-1), 1)
        eris_oovv = eris_ovov = eris_oVOv = eris_oOvV = taubuf = tau = None
        #==== mem usage blksize*(nocc*nvir**2*1)

        t2iajb = _cp(t2[p0:p1].transpose(0,2,1,3))
        t2ibja = _cp(t2[p0:p1].transpose(0,3,1,2))
        tmp = numpy.empty((blksize,nvir,nocc,nvir))
        for j0, j1 in prange(0, nocc, blksize):
            #: t2new[j0:j1] += numpy.einsum('ibkc,kcja->ijab', woVoV[j0:j1], t2ibja)
            lib.dot(woVoV[j0:j1].reshape((j1-j0)*nvir,-1),
                    t2ibja.reshape(-1,nov), 1, tmp[:j1-j0].reshape(-1,nov))
            for i in range(j1-j0):
                t2new[j0+i] += tmp[i].transpose(1,2,0)

            #: t2new[j0:j1] += numpy.einsum('iakc,kcjb->ijab', woVoV[j0:j1], t2iajb)
            lib.dot(woVoV[j0:j1].reshape((j1-j0)*nvir,-1),
                    t2iajb.reshape(-1,nov), 1, tmp[:j1-j0].reshape(-1,nov))
            for i in range(j1-j0):
                t2new[j0+i] += tmp[i].transpose(1,0,2)
        t2ibja = t2iajb = woVoV = tmp = None
        #==== mem usage blksize*(nocc*nvir**2*3)
        time1 = log.timer_debug1('contract occ [%d:%d]'%(p0, p1), *time1)
# ==================
    time1 = log.timer_debug1('contract loop', *time0)

    woooo = None
    ft_ij = foo + numpy.einsum('ja,ia->ij', .5*t1, fov)
    ft_ab = fvv - numpy.einsum('ia,ib->ab', .5*t1, fov)
    #: t2new += numpy.einsum('ijac,bc->ijab', t2, ft_ab)
    #: t2new -= numpy.einsum('ki,kjab->ijab', ft_ij, t2)
    lib.dot(t2.reshape(-1,nvir), ft_ab.T, 1, t2new.reshape(-1,nvir), 1)
    lib.dot(ft_ij.T, t2.reshape(nocc,-1),-1, t2new.reshape(nocc,-1), 1)

    #: t2new = t2new + t2new.transpose(1,0,3,2)
    t2new_tril = numpy.empty((nocc*(nocc+1)//2,nvir,nvir))
    ij = 0
    for i in range(nocc):
        for j in range(i+1):
            t2new_tril[ij]  = t2new[i,j]
            t2new_tril[ij] += t2new[j,i].T
            ij += 1
    t2new = None
    time1 = log.timer_debug1('t2 tril', *time1)
    cc.add_wvvVV_(t1, t2, eris, t2new_tril, max_memory)
    time1 = log.timer_debug1('vvvv', *time1)

    mo_e = fock.diagonal()
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    p0 = 0
    for i in range(nocc):
        t2new_tril[p0:p0+i+1] /= lib.direct_sum('a,jb->jab', eia[i], eia[:i+1])
        p0 += i+1
    time1 = log.timer_debug1('g2/dijab', *time1)

    t2new = numpy.empty((nocc,nocc,nvir,nvir))
    ij = 0
    for i in range(nocc):
        for j in range(i):
            t2new[i,j] = t2new_tril[ij]
            t2new[j,i] = t2new_tril[ij].T
            ij += 1
        t2new[i,i] = t2new_tril[ij]
        ij += 1
    t2new_tril = None

#** update_amp_t1
    t1new += fock[:nocc,nocc:] \
           + numpy.einsum('ib,ab->ia', t1, fvv) \
           - numpy.einsum('ja,ji->ia', t1, foo)

    mo_e = fock.diagonal()
    eia = mo_e[:nocc,None] - mo_e[None,nocc:]
    t1new /= eia
#** end update_amp_t1
    time0 = log.timer_debug1('update t1 t2', *time0)

    return t1new, t2new

def energy(cc, t1, t2, eris):
    nocc, nvir = t1.shape
    fock = eris.fock
    e = numpy.einsum('ia,ia', fock[:nocc,nocc:], t1) * 2
    tau = numpy.empty((1,nocc,nvir,nvir))
    for p0 in range(nocc):
        p1 = p0 + 1
        make_tau(t2[p0:p1], t1[p0:p1], t1, 1, out=tau)
        theta = tau*2 - tau.transpose(0,1,3,2)
        e += numpy.einsum('ijab,ijab', theta,
                          eris.ovov[p0:p1].transpose(0,2,1,3))
    return e


class CCSD(lib.StreamObject):
    '''CCSD

    Args

    Returns
        t1[i,a]
        t2[i,j,a,b]
    '''
    def __init__(self, mf, frozen=[], mo_energy=None, mo_coeff=None, mo_occ=None):
        from pyscf import gto
        if isinstance(mf, gto.Mole):
            raise RuntimeError('''
You see this error message because of the API updates in pyscf v0.10.
In the new API, the first argument of CC class is HF objects.  Please see
http://sunqm.net/pyscf/code-rule.html#api-rules for the details of API conventions''')
        if mo_energy is None: mo_energy = mf.mo_energy
        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ

        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.max_cycle = 50
        self.conv_tol = 1e-7
        self.conv_tol_normt = 1e-5
        self.diis_space = 6
        self.diis_file = None
        self.diis_start_cycle = 0
# FIXME: Should we avoid DIIS starting early?
        self.diis_start_energy_diff = 1e9

        self.frozen = frozen

##################################################
# don't modify the following attributes, they are not input options
        self.mo_energy = mo_energy
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._conv = False
        self.emp2 = None
        self.ecc = None
        self.e_corr = None
        self.t1 = None
        self.t2 = None
        self.l1 = None
        self.l2 = None

        self._keys = set(self.__dict__.keys())

    def nocc(self):
        if isinstance(self.frozen, (int, numpy.integer)):
            self._nocc = int(self.mo_occ.sum()) // 2 - self.frozen
        else:
            mo_occ = self.mo_occ.copy()
            if len(self.frozen) > 0:
                mo_occ[numpy.asarray(self.frozen)] = 0
            self._nocc = int(mo_occ.sum()) // 2
        return self._nocc

    def nmo(self):
        if isinstance(self.frozen, (int, numpy.integer)):
            self._nmo = len(self.mo_energy) - self.frozen
        else:
            self._nmo = len(self.mo_energy) - len(self.frozen)
        return self._nmo

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        log.info('')
        log.info('******** %s flags ********', self.__class__)
        nocc = self.nocc()
        nvir = self.nmo() - nocc
        log.info('CAS nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen:
            log.info('frozen orbitals %s', str(self.frozen))
        log.info('max_cycle = %d', self.max_cycle)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_normt = %s', self.conv_tol_normt)
        log.info('diis_space = %d', self.diis_space)
        #log.info('diis_file = %s', self.diis_file)
        log.info('diis_start_cycle = %d', self.diis_start_cycle)
        log.info('diis_start_energy_diff = %g', self.diis_start_energy_diff)
        if self.mo_coeff is None:
            log.warn('mo_coeff, mo_energy are not given.\n'
                     'You may need mf.kernel() to generate them.')

    def init_amps(self, eris):
        time0 = time.clock(), time.time()
        mo_e = eris.fock.diagonal()
        nocc = self.nocc()
        nvir = mo_e.size - nocc
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        t1 = eris.fock[:nocc,nocc:] / eia
        t2 = numpy.empty((nocc,nocc,nvir,nvir))
        self.emp2 = 0
        for i in range(nocc):
            gi = eris.ovov[i].transpose(1,0,2)
            t2i = t2[i] = gi/lib.direct_sum('jb,a->jba', eia, eia[i])
            self.emp2 += 4 * numpy.einsum('jab,jab', t2i[:i], gi[:i])
            self.emp2 += 2 * numpy.einsum('ab,ab'  , t2i[i] , gi[i] )
            self.emp2 -= 2 * numpy.einsum('jab,jba', t2i[:i], gi[:i])
            self.emp2 -=     numpy.einsum('ab,ba'  , t2i[i] , gi[i] )

        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        logger.timer(self, 'init mp2', *time0)
        return self.emp2, t1, t2


    def kernel(self, t1=None, t2=None, mo_coeff=None, eris=None):
        return self.ccsd(t1, t2, mo_coeff, eris)
    def ccsd(self, t1=None, t2=None, mo_coeff=None, eris=None):
        if eris is None: eris = self.ao2mo(mo_coeff)
        self._conv, self.ecc, self.t1, self.t2 = \
                kernel(self, eris, t1, t2, max_cycle=self.max_cycle,
                       tol=self.conv_tol,
                       tolnormt=self.conv_tol_normt,
                       max_memory=self.max_memory, verbose=self.verbose)
        self.e_corr = self.ecc
        if self._conv:
            logger.info(self, 'CCSD converged')
        else:
            logger.info(self, 'CCSD not converge')
        if self._scf.e_tot == 0:
            logger.info(self, 'E_corr = %.16g', self.ecc)
        else:
            logger.info(self, 'E(CCSD) = %.16g  E_corr = %.16g',
                        self.ecc+self._scf.e_tot, self.ecc)
        return self.ecc, self.t1, self.t2

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, mo_coeff=None,
                     eris=None):
        from pyscf.cc import ccsd_lambda
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(mo_coeff)
        conv, self.l1, self.l2 = \
                ccsd_lambda.kernel(self, eris, t1, t2, l1, l2,
                                   max_cycle=self.max_cycle,
                                   tol=self.conv_tol_normt,
                                   max_memory=self.max_memory,
                                   verbose=self.verbose)
        return conv, self.l1, self.l2

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None):
        '''1-particle density matrix in MO space'''
        from pyscf.cc import ccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)[1:]
        return ccsd_rdm.make_rdm1(self, t1, t2, l1, l2)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None):
        '''2-particle density matrix in MO space'''
        from pyscf.cc import ccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2
        if l1 is None: l1, l2 = self.solve_lambda(t1, t2)[1:]
        return ccsd_rdm.make_rdm2(self, t1, t2, l1, l2)

    def ao2mo(self, mo_coeff=None):
        #nocc = self.nocc()
        #nmo = self.nmo()
        #nvir = nmo - nocc
        #eri1 = pyscf.ao2mo.incore.full(self._scf._eri, mo_coeff)
        #eri1 = pyscf.ao2mo.restore(1, eri1, nmo)
        #eris = lambda:None
        #eris.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
        #eris.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
        #eris.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
        #eris.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
        #eris.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
        #ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
        #eris.ovvv = numpy.empty((nocc,nvir,nvir*(nvir+1)//2))
        #for i in range(nocc):
        #    for j in range(nvir):
        #        eris.ovvv[i,j] = lib.pack_tril(ovvv[i,j])
        #eris.vvvv = pyscf.ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:].copy(), nvir)
        #eris.fock = numpy.diag(self.mo_energy)
        #return eris
        return _ERIS(self, mo_coeff)

    def add_wvvVV_(self, t1, t2, eris, t2new_tril, max_memory=2000):
        time0 = time.clock(), time.time()
        nocc, nvir = t1.shape
        #: tau = t2 + numpy.einsum('ia,jb->ijab', t1, t1)
        #: t2new += numpy.einsum('ijcd,acdb->ijab', tau, vvvv)
        tau = numpy.empty((nocc*(nocc+1)//2,nvir,nvir))
        p0 = 0
        for i in range(nocc):
            tau[p0:p0+i+1] = numpy.einsum('a,jb->jab', t1[i], t1[:i+1])
            tau[p0:p0+i+1] += t2[i,:i+1]
            p0 += i + 1
        time0 = logger.timer_debug1(self, 'vvvv-tau', *time0)

        p0 = 0
        outbuf = numpy.empty((nvir,nvir,nvir))
        for a in range(nvir):
            buf = _ccsd.unpack_tril(eris.vvvv[p0:p0+a+1], out=outbuf[:a+1])
            #: t2new_tril[i,:i+1, a] += numpy.einsum('xcd,cdb->xb', tau[:,:a+1], buf)
            lib.numpy_helper._dgemm('N', 'N', nocc*(nocc+1)//2, nvir, (a+1)*nvir,
                                    tau.reshape(-1,nvir*nvir), buf.reshape(-1,nvir),
                                    t2new_tril.reshape(-1,nvir*nvir), 1, 1,
                                    0, 0, a*nvir)

            #: t2new_tril[i,:i+1,:a] += numpy.einsum('xd,abd->xab', tau[:,a], buf[:a])
            if a > 0:
                lib.numpy_helper._dgemm('N', 'T', nocc*(nocc+1)//2, a*nvir, nvir,
                                        tau.reshape(-1,nvir*nvir), buf.reshape(-1,nvir),
                                        t2new_tril.reshape(-1,nvir*nvir), 1, 1,
                                        a*nvir, 0, 0)
            p0 += a+1
            time0 = logger.timer_debug1(self, 'vvvv %d'%a, *time0)
        return t2new_tril
    def add_wvvVV(self, t1, t2, eris, max_memory=2000):
        nocc, nvir = t1.shape
        t2new_tril = numpy.zeros((nocc*(nocc+1)//2,nvir,nvir))
        return self.add_wvvVV_(t1, t2, eris, t2new_tril, max_memory)

    def update_amps(self, t1, t2, eris, max_memory=2000):
        return update_amps(self, t1, t2, eris, max_memory)

    def diis(self, t1, t2, istep, normt, de, adiis):
        return self.diis_(t1, t2, istep, normt, de, adiis)
    def diis_(self, t1, t2, istep, normt, de, adiis):
        if (istep > self.diis_start_cycle and
            abs(de) < self.diis_start_energy_diff):
            t1t2 = numpy.hstack((t1.ravel(),t2.ravel()))
#NOTE: here overwriting .data to reduce memory usage, the contents of
# t1, t2 are CHANGED!  If the pass-in t1/t2 are used elsewhere, be very
# careful to call this function
            t1.data = t1t2.data
            t2.data = t1t2.data
            t1t2 = adiis.update(t1t2)
            t1 = t1t2[:t1.size].reshape(t1.shape)
            t2 = t1t2[t1.size:].reshape(t2.shape)
            logger.debug(self, 'DIIS for step %d', istep)
        return t1, t2

CC = CCSD

class _ERIS:
    def __init__(self, cc, mo_coeff=None, method='incore'):
        cput0 = (time.clock(), time.time())
        moidx = numpy.ones(cc.mo_energy.size, dtype=numpy.bool)
        if isinstance(cc.frozen, (int, numpy.integer)):
            moidx[:cc.frozen] = False
        elif len(cc.frozen) > 0:
            moidx[numpy.asarray(cc.frozen)] = False
        if mo_coeff is None:
            self.mo_coeff = mo_coeff = cc.mo_coeff[:,moidx]
            self.fock = numpy.diag(cc.mo_energy[moidx])
        else:  # If mo_coeff is not canonical orbital
            self.mo_coeff = mo_coeff = mo_coeff[:,moidx]
            dm = cc._scf.make_rdm1(cc.mo_coeff, cc.mo_occ)
            fockao = cc._scf.get_hcore() + cc._scf.get_veff(cc.mol, dm)
            self.fock = reduce(numpy.dot, (mo_coeff.T, fockao, mo_coeff))

        nocc = cc.nocc()
        nmo = cc.nmo()
        nvir = nmo - nocc
        mem_incore, mem_outcore, mem_basic = _mem_usage(nocc, nvir)
        mem_now = pyscf.lib.current_memory()[0]

        log = logger.Logger(cc.stdout, cc.verbose)
        if (method == 'incore' and cc._scf._eri is not None and
            (mem_incore+mem_now < cc.max_memory) or cc.mol.incore_anyway):
            eri1 = pyscf.ao2mo.incore.full(cc._scf._eri, mo_coeff)
            #:eri1 = pyscf.ao2mo.restore(1, eri1, nmo)
            #:self.oooo = eri1[:nocc,:nocc,:nocc,:nocc].copy()
            #:self.ooov = eri1[:nocc,:nocc,:nocc,nocc:].copy()
            #:self.ovoo = eri1[:nocc,nocc:,:nocc,:nocc].copy()
            #:self.oovv = eri1[:nocc,:nocc,nocc:,nocc:].copy()
            #:self.ovov = eri1[:nocc,nocc:,:nocc,nocc:].copy()
            #:ovvv = eri1[:nocc,nocc:,nocc:,nocc:].copy()
            #:self.ovvv = numpy.empty((nocc,nvir,nvir*(nvir+1)//2))
            #:for i in range(nocc):
            #:    for j in range(nvir):
            #:        self.ovvv[i,j] = lib.pack_tril(ovvv[i,j])
            #:self.vvvv = pyscf.ao2mo.restore(4, eri1[nocc:,nocc:,nocc:,nocc:], nvir)
            nvir_pair = nvir * (nvir+1) // 2
            self.oooo = numpy.empty((nocc,nocc,nocc,nocc))
            self.ooov = numpy.empty((nocc,nocc,nocc,nvir))
            self.ovoo = numpy.empty((nocc,nvir,nocc,nocc))
            self.oovv = numpy.empty((nocc,nocc,nvir,nvir))
            self.ovov = numpy.empty((nocc,nvir,nocc,nvir))
            self.ovvv = numpy.empty((nocc,nvir,nvir_pair))
            self.vvvv = numpy.empty((nvir_pair,nvir_pair))
            ij = 0
            outbuf = numpy.empty((nmo,nmo,nmo))
            for i in range(nocc):
                buf = _ccsd.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
                for j in range(i+1):
                    self.oooo[i,j] = self.oooo[j,i] = buf[j,:nocc,:nocc]
                    self.ooov[i,j] = self.ooov[j,i] = buf[j,:nocc,nocc:]
                    self.oovv[i,j] = self.oovv[j,i] = buf[j,nocc:,nocc:]
                ij += i + 1
            ij1 = 0
            for i in range(nocc,nmo):
                buf = _ccsd.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
                self.ovoo[:,i-nocc] = buf[:nocc,:nocc,:nocc]
                self.ovov[:,i-nocc] = buf[:nocc,:nocc,nocc:]
                for j in range(nocc):
                    self.ovvv[j,i-nocc] = lib.pack_tril(_cp(buf[j,nocc:,nocc:]))
                for j in range(nocc, i+1):
                    self.vvvv[ij1] = lib.pack_tril(_cp(buf[j,nocc:,nocc:]))
                    ij1 += 1
                ij += i + 1
        else:
            cput1 = time.clock(), time.time()
            _tmpfile1 = tempfile.NamedTemporaryFile()
            _tmpfile2 = tempfile.NamedTemporaryFile()
            self.feri1 = h5py.File(_tmpfile1.name)
            orbo = mo_coeff[:,:nocc]
            orbv = mo_coeff[:,nocc:]
            nvpair = nvir * (nvir+1) // 2
            self.oooo = self.feri1.create_dataset('oooo', (nocc,nocc,nocc,nocc), 'f8')
            self.ooov = self.feri1.create_dataset('ooov', (nocc,nocc,nocc,nvir), 'f8')
            self.ovoo = self.feri1.create_dataset('ovoo', (nocc,nvir,nocc,nocc), 'f8')
            self.oovv = self.feri1.create_dataset('oovv', (nocc,nocc,nvir,nvir), 'f8')
            self.ovov = self.feri1.create_dataset('ovov', (nocc,nvir,nocc,nvir), 'f8')
            self.ovvv = self.feri1.create_dataset('ovvv', (nocc,nvir,nvpair), 'f8')

            max_memory = max(2000,cc.max_memory-pyscf.lib.current_memory()[0])
            self.feri2 = h5py.File(_tmpfile2.name, 'w')
            pyscf.ao2mo.full(cc.mol, orbv, self.feri2, max_memory=max_memory, verbose=log)
            self.vvvv = self.feri2['eri_mo']
            cput1 = log.timer_debug1('transforming vvvv', *cput1)

            tmpfile3 = tempfile.NamedTemporaryFile()
            with h5py.File(tmpfile3.name, 'w') as feri:
                max_memory = max(2000, cc.max_memory-pyscf.lib.current_memory()[0])
                pyscf.ao2mo.general(cc.mol, (orbo,mo_coeff,mo_coeff,mo_coeff),
                                    feri, max_memory=max_memory, verbose=log)
                cput1 = log.timer_debug1('transforming oppp', *cput1)
                eri1 = feri['eri_mo']
                outbuf = numpy.empty((nmo,nmo,nmo))
                for i in range(nocc):
                    buf = _ccsd.unpack_tril(_cp(eri1[i*nmo:(i+1)*nmo]), out=outbuf)
                    self.oooo[i] = buf[:nocc,:nocc,:nocc]
                    self.ooov[i] = buf[:nocc,:nocc,nocc:]
                    self.ovoo[i] = buf[nocc:,:nocc,:nocc]
                    self.oovv[i] = buf[:nocc,nocc:,nocc:]
                    self.ovov[i] = buf[nocc:,:nocc,nocc:]
                    self.ovvv[i] = _ccsd.pack_tril(_cp(buf[nocc:,nocc:,nocc:]))
                    cput1 = log.timer_debug1('sorting %d'%i, *cput1)
                for key in feri.keys():
                    del(feri[key])
        log.timer('CCSD integral transformation', *cput0)

    def __del__(self):
        if hasattr(self, 'feri1'):
            for key in self.feri1.keys(): del(self.feri1[key])
            for key in self.feri2.keys(): del(self.feri2[key])
            self.feri1.close()
            self.feri2.close()


# assume nvir > nocc, minimal requirements on memory in loop of update_amps
def _memory_usage_inloop(nocc, nvir):
    v = max(nvir**3*2+nvir*nocc**2*2,
            nvir**3+nocc*nvir**2*5+nvir*nocc**2*2,
            nocc*nvir**2*9)
    return v*8/1e6
# assume nvir > nocc, minimal requirements on memory
def _mem_usage(nocc, nvir):
    basic = _memory_usage_inloop(nocc, nvir)*1e6/8 + nocc**4
    basic = max(basic, nocc*(nocc+1)//2*nvir**2) + (nocc*nvir)**2*2
    basic = basic * 8/1e6
    nmo = nocc + nvir
    incore = (max((nmo*(nmo+1)//2)**2*2*8/1e6, basic) +
              (nocc*nvir**3/2 + nvir**4/4 + nocc**2*nvir**2*2 +
               nocc**3*nvir*2)*8/1e6)
    outcore = basic
    return incore, outcore, basic

def residual_as_diis_errvec(mycc):
    def fupdate(t1, t2, istep, normt, de, adiis):
        nocc, nvir = t1.shape
        nov = nocc*nvir
        moidx = numpy.ones(mycc.mo_energy.size, dtype=numpy.bool)
        if isinstance(mycc.frozen, (int, numpy.integer)):
            moidx[:mycc.frozen] = False
        else:
            moidx[mycc.frozen] = False
        mo_e = mycc.mo_energy[moidx]
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        if (istep > mycc.diis_start_cycle and
            abs(de) < mycc.diis_start_energy_diff):
            if mycc.t1 is None:
                mycc.t1 = t1
                mycc.t2 = t2
            else:
                tbuf = numpy.empty(nov*(nov+1))
                tbuf[:nov] = ((t1-mycc.t1)*eia).ravel()
                pbuf = tbuf[nov:].reshape(nocc,nocc,nvir,nvir)
                for i in range(nocc):
                    pbuf[i] = (t2[i]-mycc.t2[i]) * lib.direct_sum('jb,a->jba', eia, eia[i])
                adiis.push_err_vec(tbuf)
                tbuf = numpy.empty(nov*(nov+1))
                tbuf[:nov] = t1.ravel()
                tbuf[nov:] = t2.ravel()
                t1.data = tbuf.data # release memory
                t2.data = tbuf.data

                tbuf = adiis.update(tbuf)
                mycc.t1 = t1 = tbuf[:nov].reshape(nocc,nvir)
                mycc.t2 = t2 = tbuf[nov:].reshape(nocc,nocc,nvir,nvir)
            logger.debug(mycc, 'DIIS for step %d', istep)
        return t1, t2
    return fupdate


def _fp(nocc, nvir):
    '''Total float points'''
    return (nocc**3*nvir**2*2 + nocc**2*nvir**3*2 +     # Ftilde
            nocc**4*nvir*2 * 2 + nocc**4*nvir**2*2 +    # Wijkl
            nocc*nvir**4*2 * 2 +                        # Wabcd
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +
            nocc**3*nvir**3*2 + nocc**3*nvir**3*2 +
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +
            nocc**3*nvir**3*2 +                         # Wiabj
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +     # t1
            nocc**3*nvir**2*2 * 2 + nocc**4*nvir**2*2 +
            nocc*(nocc+1)/2*nvir**4*2 +                 # vvvv
            nocc**2*nvir**3*2 * 2 + nocc**3*nvir**2*2 * 2 +     # t2
            nocc**3*nvir**3*2 * 3 + nocc**3*nvir**2*2 * 4)      # Wiabj


# t2 + numpy.einsum('ia,jb->ijab', t1a, t1b)
def make_tau(t2, t1a, t1b, fac=1, out=None):
    return _ccsd.make_tau(t2, t1a, t1b, fac, out)

# t2.transpose(0,1,3,2)*2 - t2
def make_theta(t2, out=None):
    return _ccsd.make_0132(t2, t2, -1, 2, out)

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

def _cp(a):
    return numpy.array(a, copy=False, order='C')


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_h2o'
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()
    rhf = scf.RHF(mol)
    rhf.scf() # -76.0267656731

    mcc = CCSD(rhf)
    eris = mcc.ao2mo()
    emp2, t1, t2 = mcc.init_amps(eris)
    print(abs(t2).sum() - 4.9556571218177)
    print(emp2 - -0.2040199672883385)
    t1, t2 = update_amps(mcc, t1, t2, eris)
    print(abs(t1).sum()-0.0475038989126)
    print(abs(t2).sum()-5.401823846018721)
    print(energy(mcc, t1, t2, eris) - -0.208967840546667)
    t1, t2 = update_amps(mcc, t1, t2, eris)
    print(energy(mcc, t1, t2, eris) - -0.212173678670510)
    print(abs(t1).sum() - 0.05470123093500083)
    print(abs(t2).sum() - 5.5605208391876539)

    mcc.ccsd()
    print(mcc.ecc - -0.213343234198275)
    print(abs(mcc.t2).sum() - 5.63970304662375)

    mcc.max_memory = 1
    mcc.ccsd()
    print(mcc.ecc - -0.213343234198275)
    print(abs(mcc.t2).sum() - 5.63970304662375)

    mcc.diis = residual_as_diis_errvec(mcc)
    mcc.ccsd()
    print(mcc.ecc - -0.213343234198275)
    print(abs(mcc.t2).sum() - 5.63970304662375)

