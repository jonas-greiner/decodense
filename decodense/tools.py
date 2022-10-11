#!/usr/bin/env python
# -*- coding: utf-8 -*

"""
tools module
"""

__author__ = 'Dr. Janus Juul Eriksen, University of Bristol, UK'
__maintainer__ = 'Dr. Janus Juul Eriksen'
__email__ = 'janus.eriksen@bristol.ac.uk'
__status__ = 'Development'

import sys
import os
import copy
import numpy as np
try:
    import opt_einsum as oe
    OE_AVAILABLE = True
except ImportError:
    OE_AVAILABLE = False
from subprocess import Popen, PIPE
from pyscf import gto, scf, dft, symm, lib
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc.tools import k2gamma
from pyscf import tools as pyscf_tools
from typing import Tuple, List, Dict, Union

MAX_CYCLE = 100
NATORB_THRES = 1.e-12

class Logger(object):
        """
        this class pipes all write statements to both stdout and output_file
        """
        def __init__(self, output_file, both=True) -> None:
            """
            init Logger
            """
            self.terminal = sys.stdout
            self.log = open(output_file, 'a')
            self.both = both

        def write(self, message) -> None:
            """
            define write
            """
            self.log.write(message)
            if self.both:
                self.terminal.write(message)

        def flush(self) -> None:
            """
            define flush
            """
            pass


def git_version() -> str:
        """
        this function returns the git revision as a string
        """
        def _minimal_ext_cmd(cmd):
            env = {}
            for k in ['SYSTEMROOT', 'PATH', 'HOME']:
                v = os.environ.get(k)
                if v is not None:
                    env[k] = v
            # LANGUAGE is used on win32
            env['LANGUAGE'] = 'C'
            env['LANG'] = 'C'
            env['LC_ALL'] = 'C'
            out = Popen(cmd, stdout=PIPE, env=env, \
                        cwd=os.path.dirname(__file__)).communicate()[0]
            return out

        try:
            out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
            GIT_REVISION = out.strip().decode('ascii')
        except OSError:
            GIT_REVISION = "Unknown"

        return GIT_REVISION


def dim(mo_occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        determine molecular dimensions
        """
        return np.where(np.abs(mo_occ[0]) > 0.)[0], np.where(np.abs(mo_occ[1]) > 0.)[0]


def mf_info(mf: Union[scf.hf.SCF, dft.rks.KohnShamDFT]) -> Tuple[Tuple[np.ndarray, np.ndarray], \
                                                                 Tuple[np.ndarray, np.ndarray]]:
        """
        retrieve mf information (mo coefficients & occupations)
        """
        # mo occupations
        if np.asarray(mf.mo_occ).ndim == 1:
            mo_occ = (np.ones(np.count_nonzero(0. < mf.mo_occ)), np.ones(np.count_nonzero(1. < mf.mo_occ)))
        else:
            mo_occ = (mf.mo_occ[0][np.nonzero(mf.mo_occ[0])], mf.mo_occ[1][np.nonzero(mf.mo_occ[1])])
        # dimensions
        alpha, beta = dim(mo_occ)
        # mo coefficients
        if np.asarray(mf.mo_coeff).ndim == 2:
            mo_coeff = (mf.mo_coeff[:, alpha].real, mf.mo_coeff[:, beta].real)
        else:
            mo_coeff = (mf.mo_coeff[0][:, alpha].real, mf.mo_coeff[1][:, beta].real)

        return mo_coeff, mo_occ


def orbsym(mol, mo_coeff):
        """
        this functions returns orbital symmetries
        """
        if isinstance(mo_coeff, np.ndarray):
            if mo_coeff.ndim == 2:
                try:
                    orbsymm = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff)
                except:
                    orbsymm = np.array(['A'] * mo_coeff.shape[1])
            else:
                try:
                    orbsymm = np.array([symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, c) for c in mo_coeff])
                except:
                    orbsymm = np.array([['A'] * c.shape[1] for c in mo_coeff])
        else:
            try:
                orbsymm = np.array([symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, c) for c in mo_coeff])
            except:
                orbsymm = np.array([['A'] * c.shape[1] for c in mo_coeff])

        return orbsymm


def make_rdm1(mo: np.ndarray, occup: np.ndarray) -> np.ndarray:
        """
        this function returns an 1-RDM (in ao basis) corresponding to given mo(s)
        """
        return contract('ip,jp->ij', occup * mo, mo)


def make_natorb(mol: Union[gto.Mole, pbc_gto.Cell], mo_coeff: np.ndarray, \
                rdm1: np.ndarray, thres: float = NATORB_THRES) -> Tuple[Tuple[np.ndarray, np.ndarray], \
                                                                        Tuple[np.ndarray, np.ndarray]]:
        """
        this function returns no coefficients and occupations corresponding
        to given mo coefficients and rdm1
        """
        # reshape mo_coeff and rdm1
        if mo_coeff.ndim == 2:
            c = np.asarray((mo_coeff,) * 2)
        else:
            c = mo_coeff
        if rdm1.ndim == 2:
            d = np.array([rdm1, rdm1]) * .5
        else:
            d = rdm1
        # overlap matrix
        if isinstance(mol, pbc_gto.Cell):
            s = mol.pbc_intor('int1e_ovlp_sph')
        else:
            s = mol.intor_symmetric('int1e_ovlp')
        # ao to mo transformation of dm
        rdm1_mo = contract('xpi,pq,xqr,rs,xsj->xij', c, s, d, s, c)
        # diagonalize rdm1_mo
        occ_no, u = np.linalg.eigh(rdm1_mo)
        # transform to no basis
        mo_no = contract('xip,xpj->xij', c, u)
        # retain only significant nos
        return (mo_no[0][:, np.where(np.abs(occ_no[0]) >= thres)[0]], mo_no[1][:, np.where(np.abs(occ_no[1]) >= thres)[0]]), \
               (occ_no[0][np.where(np.abs(occ_no[0]) >= thres)], occ_no[1][np.where(np.abs(occ_no[1]) >= thres)])


def write_rdm1(mol: gto.Mole, part: str, \
               mo_coeff: np.ndarray, mo_occ: np.ndarray, fmt: str, \
               weights: List[np.ndarray], \
               suffix: str = '') -> None:
        """
        this function writes a 1-RDM as a numpy or cube (default) file
        """
        # assertion
        assert part == 'atoms', '`write_rdm1` function only implemented for `atoms` partitioning'
        assert fmt in ['cube', 'numpy'], 'fmt arg to `write_rdm1` must be `cube` or `numpy`'
        # molecular dimensions
        alpha, beta = dim(mo_occ)
        # compute total 1-RDM (AO basis)
        rdm1_tot = np.array([make_rdm1(mo_coeff[0], mo_occ[0]), make_rdm1(mo_coeff[1], mo_occ[1])])
        # loop over atoms
        for a in range(mol.natm):
            # atom-specific rdm1
            rdm1_atom = np.zeros_like(rdm1_tot)
            # loop over spins
            for i, spin_mo in enumerate((alpha, beta)):
                # loop over spin-orbitals
                for m, j in enumerate(spin_mo):
                    # get orbital(s)
                    orb = mo_coeff[i][:, j].reshape(mo_coeff[i].shape[0], -1)
                    # orbital-specific rdm1
                    rdm1_orb = make_rdm1(orb, mo_occ[i][j])
                    # weighted contribution to rdm1_atom
                    rdm1_atom[i] += rdm1_orb * weights[i][m][a]
            if fmt == 'cube':
                # write rdm1_atom as cube file
                pyscf_tools.cubegen.density(mol, f'atom_{mol.atom_symbol(a).upper():s}{a:d}_rdm1{suffix:}.cube', \
                                            np.sum(rdm1_atom, axis=0))
            else:
                # write rdm1_atom as numpy file
                np.save(f'atom_{mol.atom_symbol(a).upper():s}{a:d}_rdm1{suffix:}.npy', np.sum(rdm1_atom, axis=0))


def res_add(res_a, res_b):
        """
        this function adds two result dictionaries
        """
        return {key: res_a[key] + res_b[key] for key in res_a.keys()}


def res_sub(res_a, res_b):
        """
        this function subtracts two result dictionaries
        """
        return {key: res_a[key] - res_b[key] for key in res_a.keys()}


def contract(eqn, *tensors):
        """
        interface to optimized einsum operation
        """
        if OE_AVAILABLE:
            return oe.contract(eqn, *tensors)
        else:
            return np.einsum(eqn, *tensors, optimize=True)


