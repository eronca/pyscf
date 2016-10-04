#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''Density expansion on plane waves'''

import sys
import copy
import numpy
import h5py
from pyscf import lib
from pyscf import gto
from pyscf import dft
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.gto import pseudo
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import numint
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df import fft_jk
from pyscf.pbc.df import fft_ao2mo


def get_nuc(mydf, kpts=None):
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    gs = mydf.gs
    charge = -cell.atom_charges()
    Gv = cell.get_Gv(gs)
    SI = cell.get_SI(Gv)
    rhoG = numpy.dot(charge, SI)

    coulG = tools.get_coulG(cell, gs=gs, Gv=Gv)
    vneG = rhoG * coulG
    vneR = tools.ifft(vneG, mydf.gs).real

    vne = [lib.dot(aoR.T.conj()*vneR, aoR)
           for k, aoR in mydf.aoR_loop(cell, gs, kpts_lst)]

    if kpts is None or numpy.shape(kpts) == (3,):
        vne = vne[0]
    return vne

def get_pp(mydf, kpts=None):
    '''Get the periodic pseudotential nuc-el AO matrix, with G=0 removed.
    '''
    cell = mydf.cell
    if kpts is None:
        kpts_lst = numpy.zeros((1,3))
    else:
        kpts_lst = numpy.reshape(kpts, (-1,3))

    gs = mydf.gs
    SI = cell.get_SI()
    Gv = cell.get_Gv(gs)
    vpplocG = pseudo.get_vlocG(cell, Gv)
    vpplocG = -numpy.einsum('ij,ij->j', SI, vpplocG)
    vpplocG[0] = numpy.sum(pseudo.get_alphas(cell)) # from get_jvloc_G0 function
    ngs = len(vpplocG)
    nao = cell.nao_nr()

    # vpploc evaluated in real-space
    vpplocR = tools.ifft(vpplocG, cell.gs).real
    vpp = [lib.dot(aoR.T.conj()*vpplocR, aoR)
           for k, aoR in mydf.aoR_loop(cell, gs, kpts_lst)]

    # vppnonloc evaluated in reciprocal space
    fakemol = gto.Mole()
    fakemol._atm = numpy.zeros((1,gto.ATM_SLOTS), dtype=numpy.int32)
    fakemol._bas = numpy.zeros((1,gto.BAS_SLOTS), dtype=numpy.int32)
    ptr = gto.PTR_ENV_START
    fakemol._env = numpy.zeros(ptr+10)
    fakemol._bas[0,gto.NPRIM_OF ] = 1
    fakemol._bas[0,gto.NCTR_OF  ] = 1
    fakemol._bas[0,gto.PTR_EXP  ] = ptr+3
    fakemol._bas[0,gto.PTR_COEFF] = ptr+4

    def vppnl_by_k(kpt):
        Gk = Gv + kpt
        G_rad = lib.norm(Gk, axis=1)
        aokG = ft_ao.ft_ao(cell, Gv, kpt=kpt) * (ngs/cell.vol)
        vppnl = 0
        for ia in range(cell.natm):
            symb = cell.atom_symbol(ia)
            if symb not in cell._pseudo:
                continue
            pp = cell._pseudo[symb]
            for l, proj in enumerate(pp[5:]):
                rl, nl, hl = proj
                if nl > 0:
                    hl = numpy.asarray(hl)
                    fakemol._bas[0,gto.ANG_OF] = l
                    fakemol._env[ptr+3] = .5*rl**2
                    fakemol._env[ptr+4] = rl**(l+1.5)*numpy.pi**1.25
                    pYlm_part = dft.numint.eval_ao(fakemol, Gk, deriv=0)

                    pYlm = numpy.empty((nl,l*2+1,ngs))
                    for k in range(nl):
                        qkl = pseudo.pp._qli(G_rad*rl, l, k)
                        pYlm[k] = pYlm_part.T * qkl
                    # pYlm is real
                    SPG_lmi = numpy.einsum('g,nmg->nmg', SI[ia].conj(), pYlm)
                    SPG_lm_aoG = numpy.einsum('nmg,gp->nmp', SPG_lmi, aokG)
                    tmp = numpy.einsum('ij,jmp->imp', hl, SPG_lm_aoG)
                    vppnl += numpy.einsum('imp,imq->pq', SPG_lm_aoG.conj(), tmp)
        return vppnl * (1./ngs**2)

    for k, kpt in enumerate(kpts_lst):
        vppnl = vppnl_by_k(kpt)
        if abs(kpt).sum() < 1e-9:  # gamma_point
            vpp[k] = vpp[k].real + vppnl.real
        else:
            vpp[k] += vppnl

    if kpts is None or numpy.shape(kpts) == (3,):
        vpp = vpp[0]
    return vpp


class DF(lib.StreamObject):
    '''Density expansion on plane waves
    '''
    def __init__(self, cell, kpts=numpy.zeros((1,3))):
        self.cell = cell
        self.stdout = cell.stdout
        self.verbose = cell.verbose
        self.max_memory = cell.max_memory

        self.kpts = kpts
        self.gs = cell.gs

# Not input options
        self.exxdiv = None  # to mimic KRHF/KUHF object in function get_coulG
        self._numint = numint._KNumInt()
        self._keys = set(self.__dict__.keys())

    def dump_flags(self):
        log = logger.Logger(self.stdout, self.verbose)
        logger.info(self, '\n')
        logger.info(self, '******** %s flags ********', self.__class__)
        logger.info(self, 'gs = %s', self.gs)
        logger.info(self, 'len(kpts) = %d', len(self.kpts))
        logger.debug1(self, '    kpts = %s', self.kpts)

    def aoR_loop(self, cell, gs=None, kpts=None, kpt_band=None):
        if kpts is None: kpts = self.kpts
        kpts = numpy.asarray(kpts)

        if gs is None:
            gs = self.gs
        else:
            self.gs = gs
        ngrids = numpy.prod(numpy.asarray(gs)*2+1)

        if (self._numint.cell is None or id(cell) != id(self._numint.cell) or
            self._numint._deriv != 0 or
            self._numint._kpts.shape != kpts.shape or
            abs(self._numint._kpts - kpts).sum() > 1e-9 or
            self._numint._coords.shape[0] != ngrids or
            (gs is not None and numpy.any(gs != self.gs))):

            nkpts = len(kpts)
            coords = gen_grid.gen_uniform_grids(cell, gs)
            nao = cell.nao_nr()
            blksize = int(self.max_memory*1e6/(nkpts*nao*16*numint.BLKSIZE))*numint.BLKSIZE
            blksize = min(max(blksize, numint.BLKSIZE), ngrids)
            try:
                self._numint.cache_ao(cell, kpts, 0, coords, nao, blksize)
            except IOError as e:
                sys.stderr.write('HDF5 file cannot be opened twice in different mode.\n')
                raise e

        with h5py.File(self._numint._ao.name, 'r') as f:
            if kpt_band is None:
                for k in range(len(kpts)):
                    aoR = f['ao/%d'%k].value
                    yield k, aoR
            else:
                kpt_band = numpy.reshape(kpt_band, 3)
                where = numpy.argmin(pyscf.lib.norm(kpts-kpt_band,axis=1))
                if abs(kpts[where]-kpt_band).sum() > 1e-9:
                    where = None
                    coords = gen_grid.gen_uniform_grids(cell, gs)
                    yield 0, numint.eval_ao(cell, coords, kpt_band, deriv=deriv)
                else:
                    yield where, f['ao/%d'%where].value

    get_pp = get_pp
    get_nuc = get_nuc

    def get_jk(self, dm, hermi=1, kpts=None, kpt_band=None,
               with_j=True, with_k=True, exxdiv='ewald'):
        if kpts is None:
            if numpy.all(self.kpts == 0):
                # Gamma-point calculation by default
                kpts = numpy.zeros(3)
            else:
                kpts = self.kpts
        else:
            kpts = numpy.asarray(kpts)

        vj = vk = None
        if kpts.shape == (3,):
            if with_k:
                vk = fft_jk.get_k(self, dm, hermi, kpts, kpt_band, exxdiv)
            if with_j:
                vj = fft_jk.get_j(self, dm, hermi, kpts, kpt_band)
        else:
            if with_k:
                vk = fft_jk.get_k_kpts(self, dm, hermi, kpts, kpt_band, exxdiv)
            if with_j:
                vj = fft_jk.get_j_kpts(self, dm, hermi, kpts, kpt_band)
        return vj, vk

    get_eri = get_ao_eri = fft_ao2mo.get_eri
    ao2mo = get_mo_eri = fft_ao2mo.general
    get_ao_pairs_G = get_ao_pairs = fft_ao2mo.get_ao_pairs_G
    get_mo_pairs_G = get_mo_pairs = fft_ao2mo.get_mo_pairs_G

    def update_mf(self, mf):
        mf = copy.copy(mf)
        mf.with_df = self
        return mf


if __name__ == '__main__':
    from pyscf.pbc import gto as pbcgto
    import pyscf.pbc.scf.hf as phf
    cell = pbcgto.Cell()
    cell.verbose = 0
    cell.atom = 'C 0 0 0; C 1 1 1; C 0 2 2; C 2 0 2'
    cell.h = numpy.diag([4, 4, 4])
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.gs = [10, 10, 10]
    cell.build()
    k = numpy.ones(3)*.25
    df = PWDF(cell)
    v1 = get_pp(df, k)

