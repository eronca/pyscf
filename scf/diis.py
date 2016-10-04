#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

"""
DIIS
"""

from functools import reduce
import numpy
import scipy.linalg
import pyscf.lib
import pyscf.lib.diis
from pyscf.lib import logger


# J. Mol. Struct. 114, 31-34
# PCCP, 4, 11
# GEDIIS, JCTC, 2, 835
# C2DIIS, IJQC, 45, 31
# SCF-EDIIS, JCP 116, 8255
# error vector = SDF-FDS
# error vector = F_ai ~ (S-SDS)*S^{-1}FDS = FDS - SDFDS ~ FDS-SDF in converge
class DIIS(pyscf.lib.diis.DIIS):
    def __init__(self, mf, filename):
        pyscf.lib.diis.DIIS.__init__(self, mf, filename)
        self.rollback = False
    def update(self, s, d, f):
        if isinstance(f, numpy.ndarray) and f.ndim == 2:
            sdf = reduce(numpy.dot, (s,d,f))
            errvec = sdf.T.conj() - sdf

        elif isinstance(f, numpy.ndarray) and f.ndim == 3 and s.ndim == 3:
            errvec = []
            for i in range(f.shape[0]):
                sdf = reduce(numpy.dot, (s[i], d[i], f[i]))
                errvec.append((sdf.T.conj() - sdf))
            errvec = numpy.vstack(errvec)

        elif f.ndim == s.ndim+1 and f.shape[0] == 2:  # for UHF
            nao = s.shape[-1]
            s = pyscf.lib.asarray((s,s)).reshape(-1,nao,nao)
            fnew = self.update(s, d.reshape(s.shape), f.reshape(s.shape))
            return fnew.reshape(f.shape)
        else:
            raise RuntimeError('Unknown SCF DIIS type')
        logger.debug1(self, 'diis-norm(errvec)=%g', numpy.linalg.norm(errvec))
        xnew = pyscf.lib.diis.DIIS.update(self, f, xerr=errvec)
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
        return xnew

    def get_num_vec(self):
        if self.rollback:
            return self._head
        else:
            return len(self._bookkeep)

SCFDIIS = SCF_DIIS = DIIS
