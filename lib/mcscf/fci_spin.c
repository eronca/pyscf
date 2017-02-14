/*
 * 2-particle spin density matrix
 * Gamma(ia,jb,kb,la) or Gamma(ib,ja,ka,lb)
 */

#include <stdlib.h>
#include <string.h>
#include "config.h"
#include "vhf/fblas.h"
#define CSUMTHR         1e-28

#define EXTRACT_CRE(tab, i)  (tab[i*4+0])
#define EXTRACT_DES(tab, i)  (tab[i*4+1])
#define EXTRACT_ADDR(tab, i) (tab[i*4+2])
#define EXTRACT_SIGN(tab, i) (tab[i*4+3])

/*
 * the intermediate determinants ~ (norb,neleca+1;norb,nelecb-1)
 * Annihilating one alpha electron and creating one beta electron lead
 * to the input ground state CI |0>
 * stra_id is the ID of the intermediate determinants.  t1 is a buffer
 * of size [nstrb_or_fillcnt,norb*norb].  fillcnt is the dim of beta
 * strings for intermediate determinants
 */
static double ades_bcre_t1(double *ci0, double *t1, int fillcnt, int stra_id,
                           int norb, int nstrb, int neleca, int nelecb,
                           int *ades_index, int *bcre_index)
{
        const int nnorb = norb * norb;
        const int inelec = neleca + 1;
        const int invir  = norb - nelecb + 1;
        int ic, id, i, j, k, str1, sign, signa;
        const int *tab;
        double *pt1, *pci;
        double csum = 0;

        ades_index = ades_index + stra_id * inelec * 4;
        for (id = 0; id < inelec; id++) {
                j     = EXTRACT_DES (ades_index, id);
                str1  = EXTRACT_ADDR(ades_index, id);
                signa = EXTRACT_SIGN(ades_index, id);
                pci = ci0 + str1 * (size_t)nstrb;
                pt1 = t1 + j*norb;
                for (k = 0; k < fillcnt; k++) {
                        tab = bcre_index + k * invir * 4;
                        for (ic = 0; ic < invir; ic++) {
                                i    = EXTRACT_CRE (tab, ic);
                                str1 = EXTRACT_ADDR(tab, ic);
                                sign = EXTRACT_SIGN(tab, ic) * signa;
                                pt1[i] += pci[str1] * sign;
                                csum += pci[str1] * pci[str1];
                        }
                        pt1 += nnorb;
                }
        }
        return csum;
}

/*
 * the intermediate determinants ~ (norb,neleca-1;norb,nelecb+1)
 * Annihilating one beta electron and creating one alpha electron lead
 * to the input ground state CI |0>
 * stra_id is the ID of the intermediate determinants.  t1 is a buffer
 * of size [nstrb_or_fillcnt,norb*norb].  fillcnt is the dim of beta
 * strings for intermediate determinants
 */
static double acre_bdes_t1(double *ci0, double *t1, int fillcnt, int stra_id,
                           int norb, int nstrb, int neleca, int nelecb,
                           int *acre_index, int *bdes_index)
{
        const int nnorb = norb * norb;
        const int inelec = nelecb + 1;
        const int invir  = norb - neleca + 1;
        int ic, id, i, j, str0, str1, sign, signa;
        const int *tab;
        double *pci, *pt1;
        double csum = 0;

        acre_index = acre_index + stra_id * invir * 4;
        for (ic = 0; ic < invir; ic++) {
                i     = EXTRACT_CRE (acre_index, ic);
                str1  = EXTRACT_ADDR(acre_index, ic);
                signa = EXTRACT_SIGN(acre_index, ic);
                pci = ci0 + str1 * (size_t)nstrb;
                pt1 = t1 + i;
                tab = bdes_index;
                for (str0 = 0; str0 < fillcnt; str0++) {
                        for (id = 0; id < inelec; id++) {
                                j    = EXTRACT_DES (tab, id);
                                str1 = EXTRACT_ADDR(tab, id);
                                sign = EXTRACT_SIGN(tab, id) * signa;
                                pt1[j*norb] += sign * pci[str1];
                                csum += pci[str1] * pci[str1];
                        }
                        tab += inelec * 4;
                        pt1 += nnorb;
                }
        }
        return csum;
}


static void _transpose_jikl(double *dm2, int norb)
{
        int nnorb = norb * norb;
        int i, j;
        double *p0, *p1;
        double *tmp = malloc(sizeof(double)*nnorb*nnorb);
        memcpy(tmp, dm2, sizeof(double)*nnorb*nnorb);
        for (i = 0; i < norb; i++) {
                for (j = 0; j < norb; j++) {
                        p0 = tmp + (j*norb+i) * nnorb;
                        p1 = dm2 + (i*norb+j) * nnorb;
                        memcpy(p1, p0, sizeof(double)*nnorb);
                }
        }
        free(tmp);
}

/*
 * If symm != 0, symmetrize rdm1 and rdm2
 * For spin density matrix, return rdm2 e.g.
 *      [beta alpha beta^+ alpha]
 * transpose(1,0,2,3) to get the right order [alpha^+ beta beta^+ alpha]
 * na, nb, nlinka, nlinkb label the intermediate determinants
 * see ades_bcre_t1 and acre_bdes_t1 of fci_spin.c
 *
 * Note: na counts the alpha strings of intermediate determinants
 * but nb counts the beta strings of ket
 */
void FCIspindm12_drv(void (*dm12kernel)(),
                     double *rdm1, double *rdm2, double *bra, double *ket,
                     int norb, int na, int nb, int neleca, int nelecb,
                     int *link_indexa, int *link_indexb, int symm)
{
        const int nnorb = norb * norb;
        int strk, i, j;
        memset(rdm1, 0, sizeof(double) * nnorb);
        memset(rdm2, 0, sizeof(double) * nnorb*nnorb);

#pragma omp parallel default(none) \
        shared(dm12kernel, bra, ket, norb, na, nb, neleca, nelecb, \
               link_indexa, link_indexb, rdm1, rdm2), \
        private(strk, i)
{
        double *pdm1 = calloc(nnorb, sizeof(double));
        double *pdm2 = calloc(nnorb*nnorb, sizeof(double));
#pragma omp for schedule(static, 40)
        for (strk = 0; strk < na; strk++) {
                (*dm12kernel)(pdm1, pdm2, bra, ket,
                              strk, norb, na, nb, neleca, nelecb,
                              link_indexa, link_indexb);
        }
#pragma omp critical
{
        for (i = 0; i < nnorb; i++) {
                rdm1[i] += pdm1[i];
        }
        for (i = 0; i < nnorb*nnorb; i++) {
                rdm2[i] += pdm2[i];
        }
}
        free(pdm1);
        free(pdm2);
}
        if (symm) {
                for (i = 0; i < norb; i++) {
                        for (j = 0; j < i; j++) {
                                rdm1[j*norb+i] = rdm1[i*norb+j];
                        }
                }
                for (i = 0; i < nnorb; i++) {
                        for (j = 0; j < i; j++) {
                                rdm2[j*nnorb+i] = rdm2[i*nnorb+j];
                        }
                }
        }
        _transpose_jikl(rdm2, norb);
}


/*
 * dm(pq,rs) * [p(alpha)^+ q(beta) r(beta)^+ s(alpha)]
 */
void FCIdm2_abba_kern(double *rdm1, double *rdm2, double *bra, double *ket,
                      int stra_id, int norb, int na, int nb,
                      int neleca, int nelecb,
                      int *acre_index, int *bdes_index)
{
        const char UP = 'U';
        const char TRANS_N = 'N';
        const double D1 = 1;
        const int nnorb = norb * norb;
        const int instrb = nb * (norb-nelecb) / (nelecb+1);
        double csum;
        double *buf = calloc(nnorb * instrb, sizeof(double));

        csum = acre_bdes_t1(ket, buf, instrb, stra_id, norb, nb,
                            neleca, nelecb, acre_index, bdes_index);
        if (csum > CSUMTHR) {
                dsyrk_(&UP, &TRANS_N, &nnorb, &instrb,
                       &D1, buf, &nnorb, &D1, rdm2, &nnorb);
        }
        free(buf);
}
/*
 * dm(pq,rs) * [p(beta)^+ q(alpha) r(alpha)^+ s(beta)]
 */
void FCIdm2_baab_kern(double *rdm1, double *rdm2, double *bra, double *ket,
                      int stra_id, int norb, int na, int nb,
                      int neleca, int nelecb,
                      int *ades_index, int *bcre_index)
{
        const char UP = 'U';
        const char TRANS_N = 'N';
        const double D1 = 1;
        const int nnorb = norb * norb;
        const int instrb = nb * nelecb / (norb-nelecb+1);
        double csum;
        double *buf = calloc(nnorb * instrb, sizeof(double));

        csum = ades_bcre_t1(ket, buf, instrb, stra_id, norb, nb,
                            neleca, nelecb, ades_index, bcre_index);
        if (csum > CSUMTHR) {
                dsyrk_(&UP, &TRANS_N, &nnorb, &instrb,
                       &D1, buf, &nnorb, &D1, rdm2, &nnorb);
        }
        free(buf);
}

