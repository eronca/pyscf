import unittest
import numpy
from pyscf import lib
import pyscf.pbc
from pyscf import ao2mo
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.df import mdf
#from mpi4pyscf.pbc.df import mdf
pyscf.pbc.DEBUG = False

L = 5.
n = 5
cell = pgto.Cell()
cell.a = numpy.diag([L,L,L])
cell.gs = numpy.array([n,n,n])

cell.atom = '''C    3.    2.       3.
               C    1.    1.       1.'''
cell.basis = 'ccpvdz'
cell.verbose = 0
cell.max_memory = 1000
cell.nimgs = [2,2,2]
cell.build(0,0)

mf0 = pscf.RHF(cell)
mf0.exxdiv = 'vcut_sph'


numpy.random.seed(1)
kpts = numpy.random.random((5,3))
kpts[0] = 0
kpts[3] = kpts[0]-kpts[1]+kpts[2]
kpts[4] *= 1e-5

kmdf = mdf.MDF(cell)
kmdf.auxbasis = 'weigend'
kmdf.kpts = kpts
kmdf.gs = (5,)*3


def finger(a):
    w = numpy.cos(numpy.arange(a.size))
    return numpy.dot(a.ravel(), w)

class KnowValues(unittest.TestCase):
    def test_vbar(self):
        auxcell = mdf.make_modrho_basis(cell, 'ccpvdz', 1.)
        vbar = mdf.MDF(cell).auxbar(auxcell)
        self.assertAlmostEqual(finger(vbar), -0.00438699039629, 9)

    def test_get_eri_gamma(self):
        odf = mdf.MDF(cell)
        odf.auxbasis = 'weigend'
        odf.gs = (5,)*3
        eri0000 = odf.get_eri()
        self.assertTrue(eri0000.dtype == numpy.double)
        self.assertAlmostEqual(eri0000.real.sum(), 140.52297323243539, 9)
        self.assertAlmostEqual(finger(eri0000), -1.2233877452643904, 9)

        eri1111 = kmdf.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        self.assertTrue(eri1111.dtype == numpy.double)
        self.assertAlmostEqual(eri1111.real.sum(), 140.52297323243539, 9)
        self.assertAlmostEqual(eri1111.imag.sum(), 0, 9)
        self.assertAlmostEqual(finger(eri1111), -1.2233877452643904, 9)
        self.assertTrue(numpy.allclose(eri1111, eri0000))

        eri4444 = kmdf.get_eri((kpts[4],kpts[4],kpts[4],kpts[4]))
        self.assertTrue(eri4444.dtype == numpy.complex128)
        self.assertAlmostEqual(eri4444.real.sum(), 259.45730831500038, 9)
        self.assertAlmostEqual(abs(eri4444.imag).sum(), 0.00044186906554179585, 9)
        self.assertAlmostEqual(finger(eri4444), 1.9676993803147795-3.6138386194340396e-07j, 8)
        eri0000 = ao2mo.restore(1, eri0000, cell.nao_nr()).reshape(eri4444.shape)
        self.assertTrue(numpy.allclose(eri0000, eri4444, atol=1e-7))

    def test_get_eri_1111(self):
        eri1111 = kmdf.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        self.assertTrue(eri1111.dtype == numpy.complex128)
        self.assertAlmostEqual(eri1111.real.sum(), 258.81064470037302, 9)
        self.assertAlmostEqual(abs(eri1111.imag).sum(), 16.275794801827171, 9)
        self.assertAlmostEqual(finger(eri1111), 2.2311170496926014+0.10954499150150997j, 9)
        check2 = kmdf.get_eri((kpts[1]+5e-9,kpts[1]+5e-9,kpts[1],kpts[1]))
        self.assertTrue(numpy.allclose(eri1111, check2, atol=1e-7))

    def test_get_eri_0011(self):
        eri0011 = kmdf.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        self.assertTrue(eri0011.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0011.real.sum(), 259.12265200691797, 9)
        self.assertAlmostEqual(abs(eri0011.imag).sum(), 8.4042050419197558, 9)
        self.assertAlmostEqual(finger(eri0011), 2.1346873932253221+0.12350214171925518j, 9)

    def test_get_eri_0110(self):
        eri0110 = kmdf.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        self.assertTrue(eri0110.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0110.real.sum(), 411.85221934191958, 9)
        self.assertAlmostEqual(abs(eri0110.imag).sum(), 136.58689851732211, 9)
        self.assertAlmostEqual(finger(eri0110), 1.3738510915132758+0.12346499206902237j, 9)
        check2 = kmdf.get_eri((kpts[0]+5e-9,kpts[1]+5e-9,kpts[1],kpts[0]))
        self.assertTrue(numpy.allclose(eri0110, check2, atol=1e-7))

    def test_get_eri_0123(self):
        eri0123 = kmdf.get_eri(kpts[:4])
        self.assertTrue(eri0123.dtype == numpy.complex128)
        self.assertAlmostEqual(eri0123.real.sum(), 410.39194901276232, 9)
        self.assertAlmostEqual(abs(eri0123.imag.sum()), 0.18510581788339481, 9)
        self.assertAlmostEqual(finger(eri0123), 1.7611714642035703+0.30805724628410736j, 9)



if __name__ == '__main__':
    print("Full Tests for mdf")
    unittest.main()
