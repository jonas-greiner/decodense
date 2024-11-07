#!/usr/bin/env python
# -*- coding: utf-8 -*

import unittest
import numpy as np
import jax
from jax import numpy as adnp
from pyscfad import gto, scf as scf_ad, dft as dft_ad, config

import decodense

jax.config.update("jax_enable_x64", True)
config.update("pyscfad_scf_implicit_diff", True)


# finite difference shift
FIN_DIFF = 5e-3

# settings
SPIN = (0, 2)
# meta-GGAs currently do not work in pyscfad
MF_METHOD = ("hf", "pbe", "b3lyp", "wb97x-v")
SPIN_SYMMETRY = ("restricted", "unrestricted")
PART = ("orbitals", "eda", "atoms")
POP_METHOD = ("mulliken", "lowdin", "meta_lowdin", "becke", "iao")
GAUGE_ORIGIN = (
    np.zeros(3, dtype=np.float64),
    np.array([1.0, 2.0, 3.0], dtype=np.float64),
)

# external electric field
e_field = np.zeros(3, dtype=np.float64)


def energy(e_field, mol, mf_method, spin_symmetry, decomp, gauge_origin):
    """
    full atomic energy calculation
    """
    # mf calc
    if mf_method == "hf":
        if mol.spin == 0:
            mf = scf_ad.RHF(mol)
        elif spin_symmetry == "restricted":
            mf = scf_ad.ROHF(mol)
        else:
            mf = scf_ad.UHF(mol)
    else:
        mf = dft_ad.RKS(mol)
        mf.xc = mf_method

    # electronic contribution due to electric field
    with mol.with_common_origin(gauge_origin):
        ao_dip = mol.intor_symmetric("int1e_r", comp=3)
    ext_el = adnp.einsum("x,xij->ij", e_field, ao_dip)

    # modified one-electron hamiltonian
    h1 = mf.get_hcore()
    mf.get_hcore = lambda *args, **kwargs: h1 + ext_el
    mf.conv_tol = 1e-10
    mf.kernel()

    # nuclear contribution due to electric field
    coords = mol.atom_coords()
    form_charges = mol.atom_charges()
    ext_nuc = -adnp.einsum("i,ix,x->i", form_charges, coords - gauge_origin, e_field)

    if mf.mo_occ.ndim == 1:
        # occupied orbitals
        alpha = np.where(mf.mo_occ > 0.0)[0]
        beta = np.where(mf.mo_occ > 1.0)[0]

        # mo coefficients
        mo_coeff = (mf.mo_coeff[:, alpha], mf.mo_coeff[:, beta])
    else:
        # occupied orbitals
        alpha = np.where(mf.mo_occ[0] > 0.0)[0]
        beta = np.where(mf.mo_occ[1] > 0.0)[0]

        # mo coefficients
        mo_coeff = (mf.mo_coeff[0][:, alpha], mf.mo_coeff[1][:, beta])

    # decomposition
    res = decodense.main(
        mol, decomp, mf, mo_coeff, ad=True, ext_el=ext_el, ext_nuc=ext_nuc
    )

    # total atomic energy
    tot_energy = res.el
    if decomp.part in ["atoms", "eda"]:
        tot_energy += res.struct

    return tot_energy


class KnownValues(unittest.TestCase):
    def test(self):
        for spin in SPIN:
            # init molecule
            mol = gto.Mole(
                verbose=0,
                output=None,
                basis="6-31g",
                symmetry=True,
                atom="geom/h2o.xyz",
                spin=spin,
            )
            mol.build(trace_coords=False, trace_exp=False, trace_ctr_coeff=False)
            mol_pyscf = mol.to_pyscf()

            # get atom coordinates and charges
            coords = mol.atom_coords()
            form_charges = mol.atom_charges()

            # open-shell calculations are only enabled for hf in pyscfad
            if spin == 0:
                mf_methods = MF_METHOD
            else:
                mf_methods = ("hf",)
            for mf_method in mf_methods:
                if spin == 0:
                    spin_symmetries = ("restricted",)
                else:
                    spin_symmetries = SPIN_SYMMETRY

                # set tolerance for total dipole moment
                if mf_method == "wb97x-v":
                    rtol_tot = 1e-2
                else:
                    rtol_tot = 1e-4

                # set tolerance for comparison of atomic dipole moment to finite differences
                if spin > 0:
                    rtol_fin_diff = 1e-1
                else:
                    rtol_fin_diff = 1e-2
                for spin_symmetry in spin_symmetries:
                    # mf calc
                    if mf_method == "hf":
                        if spin == 0:
                            mf = mol_pyscf.RHF()
                        elif spin_symmetry == "restricted":
                            mf = mol_pyscf.ROHF()
                        else:
                            mf = mol_pyscf.UHF()
                    else:
                        mf = mol_pyscf.KS()
                        mf.xc = mf_method
                    mf.conv_tol = 1e-10
                    mf.kernel()
                    for gauge_origin in GAUGE_ORIGIN:
                        nuc_dip = np.einsum(
                            "i,ix->x", form_charges, coords - gauge_origin
                        )
                        # total dipole moment
                        tot_dip = mf.dip_moment(
                            unit="au", verbose=0, origin=gauge_origin
                        )

                        for part in PART:
                            if part in ["atoms", "eda"]:
                                for pop_method in POP_METHOD:
                                    with self.subTest(
                                        spin=spin,
                                        mf_method=mf_method,
                                        spin_symmetry=spin_symmetry,
                                        gauge_origin=gauge_origin,
                                        part=part,
                                        pop_method=pop_method,
                                    ):
                                        # AD dipole moment
                                        decomp = decodense.DecompCls(
                                            pop_method=pop_method,
                                            part=part,
                                            prop="energy",
                                        )
                                        ad_dip = -jax.jacrev(energy)(
                                            e_field,
                                            mol,
                                            mf_method,
                                            spin_symmetry,
                                            decomp,
                                            gauge_origin,
                                        )

                                        # finite differences dipole moment
                                        fin_diff_dip = np.empty(
                                            (mol.natm, 3), dtype=np.float64
                                        )
                                        for i in range(3):
                                            shift = np.zeros(3, dtype=np.float64)
                                            shift[i] = FIN_DIFF
                                            e_p = energy(
                                                e_field + shift,
                                                mol,
                                                mf_method,
                                                spin_symmetry,
                                                decomp,
                                                gauge_origin,
                                            )
                                            e_m = energy(
                                                e_field - shift,
                                                mol,
                                                mf_method,
                                                spin_symmetry,
                                                decomp,
                                                gauge_origin,
                                            )
                                            fin_diff_dip[:, i] = -(e_p - e_m) / (
                                                2 * FIN_DIFF
                                            )

                                        np.testing.assert_allclose(
                                            ad_dip.sum(axis=0),
                                            tot_dip,
                                            rtol=rtol_tot,
                                            atol=1e-12,
                                        )
                                        np.testing.assert_allclose(
                                            ad_dip,
                                            fin_diff_dip,
                                            rtol=rtol_fin_diff,
                                            atol=1e-9,
                                        )
                            else:
                                with self.subTest(
                                    spin=spin,
                                    mf_method=mf_method,
                                    spin_symmetry=spin_symmetry,
                                    gauge_origin=gauge_origin,
                                    part=part,
                                ):
                                    # AD dipole moment
                                    decomp = decodense.DecompCls(
                                        part=part, prop="energy"
                                    )
                                    ad_dip = [
                                        -dip
                                        for dip in jax.jacrev(energy)(
                                            e_field,
                                            mol,
                                            mf_method,
                                            spin_symmetry,
                                            decomp,
                                            gauge_origin,
                                        )
                                    ]

                                    # finite differences dipole moment
                                    if mf.mo_occ.ndim == 1:
                                        n_alpha = np.count_nonzero(mf.mo_occ > 0.0)
                                        n_beta = np.count_nonzero(mf.mo_occ > 1.0)
                                    else:
                                        n_alpha = np.count_nonzero(mf.mo_occ[0] > 0.0)
                                        n_beta = np.count_nonzero(mf.mo_occ[1] > 0.0)
                                    fin_diff_dip = [
                                        np.empty((n_alpha, 3), dtype=np.float64),
                                        np.empty((n_beta, 3), dtype=np.float64),
                                    ]
                                    for i in range(3):
                                        shift = np.zeros(3, dtype=np.float64)
                                        shift[i] = FIN_DIFF
                                        e_p = energy(
                                            e_field + shift,
                                            mol,
                                            mf_method,
                                            spin_symmetry,
                                            decomp,
                                            gauge_origin,
                                        )
                                        e_m = energy(
                                            e_field - shift,
                                            mol,
                                            mf_method,
                                            spin_symmetry,
                                            decomp,
                                            gauge_origin,
                                        )
                                        for spin_idx in range(2):
                                            fin_diff_dip[spin_idx][:, i] = -(
                                                e_p[spin_idx] - e_m[spin_idx]
                                            ) / (2 * FIN_DIFF)

                                    np.testing.assert_allclose(
                                        nuc_dip
                                        + sum(
                                            orb_dip.sum(axis=0) for orb_dip in ad_dip
                                        ),
                                        tot_dip,
                                        rtol=rtol_tot,
                                        atol=1e-12,
                                    )
                                    for ad_dip_mo, fin_diff_dip_mo in zip(
                                        ad_dip, fin_diff_dip
                                    ):
                                        np.testing.assert_allclose(
                                            ad_dip_mo,
                                            fin_diff_dip_mo,
                                            rtol=rtol_fin_diff,
                                            atol=1e-9,
                                        )


if __name__ == "__main__":
    print("test: h2o_ad_dipmom_gs")
    unittest.main()
