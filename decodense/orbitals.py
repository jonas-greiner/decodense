#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
orbitals module
"""

__author__ = "Janus Juul Eriksen, Technical University of Denmark, DK"
__maintainer__ = "Janus Juul Eriksen"
__email__ = "janus@kemi.dtu.dk"
__status__ = "Development"

import numpy as np
from pyscf import gto, scf, dft, lo
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from typing import List, Union, Tuple

from .tools import dim, contract

AD = False
adnp = np
adiao = lo.iao
adorth = lo.orth


def set_adnp(ad, adnp_in, adiao_in, adorth_in):
    """
    this function sets the AD numpy package
    """
    global AD, adnp, adiao, adorth
    AD = ad
    adnp = adnp_in
    adiao = adiao_in
    adorth = adorth_in


def assign_rdm1s(
    mol: Union[gto.Mole, pbc_gto.Cell],
    mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT, pbc_scf.hf.RHF, pbc_dft.rks.RKS],
    mo_coeff: Tuple[np.ndarray, np.ndarray],
    mo_occ: Tuple[np.ndarray, np.ndarray],
    minao: str,
    pop_method: str,
    ndo: bool,
    verbose: int,
) -> List[np.ndarray]:
    """
    this function returns a list of population weights of each spin-orbital on the
    individual atoms
    """
    # dft logical
    dft_calc = isinstance(mf, dft.rks.KohnShamDFT)

    # rhf reference
    if mo_occ[0].size == mo_occ[1].size:
        rhf = adnp.allclose(mo_coeff[0], mo_coeff[1]) and adnp.allclose(
            mo_occ[0], mo_occ[1]
        )
    else:
        rhf = False

    if isinstance(mol, pbc_gto.Cell):
        s = mol.pbc_intor("int1e_ovlp_sph")
    else:
        s = mol.intor_symmetric("int1e_ovlp")

    # molecular dimensions
    alpha, beta = dim(mo_occ)

    # mol object projected into minao basis
    if pop_method == "iao":
        # ndo assertion
        if ndo:
            raise NotImplementedError(
                "IAO-based populations for NDOs is not implemented"
            )
        pmol = lo.iao.reference_mol(mol, minao=minao)
    else:
        pmol = mol

    # number of atoms
    natm = pmol.natm

    # AO slices per atom
    ao_slices = pmol.aoslice_by_atom()[:, 2:]

    # overlap matrix
    if pop_method == "mulliken":
        ovlp = s
    else:
        ovlp = np.eye(pmol.nao_nr())

    def get_weights(mo: np.ndarray, mocc: np.ndarray) -> np.ndarray:
        """
        this function computes the full set of population weights
        """
        if pop_method == "becke":
            # population weights of orb
            return _population_becke(charge_matrix, mo)
        else:
            # orbital-specific rdm1s
            rdm1_orb = contract("p,ip,jp->ijp", mocc, mo, mo, ad=AD)
            # population weights of rdm1_orb
            return _population_mul(natm, ao_slices, ovlp, rdm1_orb)

    # init population weights array
    weights = []

    # loop over spin
    for i, spin_mo in enumerate((alpha, beta)):

        # get mo coefficients and occupation
        if pop_method == "mulliken":
            mo = mo_coeff[i][:, spin_mo]
        elif pop_method == "lowdin":
            mo = contract(
                "ki,kl,lj->ij",
                lo.orth.orth_ao(pmol, method="lowdin", s=s),
                s,
                mo_coeff[i][:, spin_mo],
                ad=AD,
            )
        elif pop_method == "meta_lowdin":
            mo = contract(
                "ki,kl,lj->ij",
                lo.orth.orth_ao(pmol, method="meta_lowdin", s=s),
                s,
                mo_coeff[i][:, spin_mo],
                ad=AD,
            )
        elif pop_method == "iao":
            iao = adiao.iao(mol, mo_coeff[i][:, spin_mo], minao=minao)
            iao = adorth.vec_lowdin(iao, s)
            mo = contract("ki,kl,lj->ij", iao, s, mo_coeff[i][:, spin_mo], ad=AD)
        elif pop_method == "becke":
            if getattr(pmol, "pbc_intor", None):
                raise NotImplementedError("PM becke scheme for PBC systems")
            if dft_calc:
                grid_coords, grid_weights = mf.grids.get_partition(mol, concat=False)
                ni = mf._numint
            else:
                mf_becke = mol.RKS()
                grid_coords, grid_weights = mf_becke.grids.get_partition(
                    mol, concat=False
                )
                ni = mf_becke._numint
            charge_matrix = np.zeros(
                [natm, pmol.nao_nr(), pmol.nao_nr()], dtype=np.float64
            )
            for j in range(natm):
                ao = ni.eval_ao(mol, grid_coords[j], deriv=0)
                aow = np.einsum("pi,p->pi", ao, grid_weights[j])
                charge_matrix[j] = contract("ki,kj->ij", aow, ao)
            mo = mo_coeff[i][:, spin_mo]
        mocc = mo_occ[i][spin_mo]

        # get weights
        weights.append(get_weights(mo, mocc))

        # closed-shell reference
        if rhf:
            weights.append(weights[0])
            break

    # verbose print
    if 0 < verbose:
        symbols = tuple(pmol.atom_pure_symbol(i) for i in range(pmol.natm))
        print("\n *** partial population weights: ***")
        print(
            " spin  " + "MO       " + "      ".join(["{:}".format(i) for i in symbols])
        )
        for i, spin_mo in enumerate((alpha, beta)):
            for j in spin_mo:
                with np.printoptions(
                    suppress=True, linewidth=200, formatter={"float": "{:6.3f}".format}
                ):
                    print(
                        "  {:s}    {:>2d}   {:}".format(
                            "a" if i == 0 else "b", j, weights[i][j]
                        )
                    )
        with np.printoptions(
            suppress=True, linewidth=200, formatter={"float": "{:6.3f}".format}
        ):
            print(
                "   total    {:}".format(
                    np.sum(weights[0], axis=0) + np.sum(weights[1], axis=0)
                )
            )

    return weights


def _population_mul(
    natm: int, ao_slices: np.ndarray, ovlp: np.ndarray, rdm1: np.ndarray
) -> np.ndarray:
    """
    this function returns the mulliken populations on the individual atoms
    """

    # mulliken population array
    pop = contract("ijp,ji->ip", rdm1, ovlp)

    if not AD:
        # init populations
        populations = np.empty((rdm1.shape[2], natm), dtype=np.float64)

        # loop over atoms
        for iatm in range(natm):
            populations[:, iatm] = np.sum(
                pop[ao_slices[iatm, 0] : ao_slices[iatm, 1]], axis=0
            )
    else:
        from jax import vmap

        # define vectorization function
        def fn(ao_slice: np.ndarray, pop: np.ndarray, idx: np.ndarray) -> np.ndarray:
            # directly slicing does not work because dynamic shapes are not allowed
            # within transforms, instead we fill the values we do not want with zeros
            # and sum the entire array
            mask = (idx >= ao_slice[0]) & (idx < ao_slice[1])
            return adnp.where(mask, pop, 0.0).sum(axis=0)

        # vectorize over atoms
        idx = np.arange(pop.shape[0])
        populations = vmap(fn, in_axes=(0, None, None), out_axes=1)(
            ao_slices, pop, idx[:, None]
        )

    return populations


def _population_becke(charge_matrix: np.ndarray, orbs: np.ndarray) -> np.ndarray:
    """
    this function returns the becke populations on the individual atoms
    """
    # calculate populations
    populations = contract("mi,amn,ni->ia", orbs, charge_matrix, orbs)

    return populations
