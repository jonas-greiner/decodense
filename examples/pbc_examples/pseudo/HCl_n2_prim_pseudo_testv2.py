#!/usr/bin/env python

import numpy as np
from pyscf.pbc import df
from pyscf.pbc import gto, scf
from pyscf.pbc.tools.pbc import super_cell
from pyscf import gto as mgto
from pyscf import scf as mscf
import decodense
import pbctools
from typing import List, Tuple, Dict, Union, Any


# decodense variables
PARAMS = {
    'prop': 'energy',
#    'basis': 'ccpvdz',
#    'xc': 'pbe0',
#    'loc': 'ibo-2',
    'loc': '',
#    'pop': 'iao',
    'part': 'atoms'
}

def check_decomp(cell, mf):
    ''' test which params work for cell '''

    ehf = mf.energy_tot()
    nat = cell.natm
    res_all = []
    for i in ['', 'fb', 'pm', 'ibo-2', 'ibo-4']:
        for j in ['mulliken', 'iao']:
            decomp = decodense.DecompCls(prop='energy', part='atoms', loc=i, pop=j)
            res = decodense.main(cell, decomp, mf)
            print('Decodense res for cell, loc: {}, pop: {}'.format(i,j))
            for k, v in res.items():
                print(k, v)
            print()
            print('E_hf_pyscf - E_hf_dec = ', ehf - (np.sum(res['kin']) + np.sum(res['coul']) + np.sum(res['exch']) + np.sum(res['nuc_att_glob']) + np.sum(res['nuc_att_loc']) + np.sum(res['struct'])) )
            print('---------------------------------')
            print()
            #res_all.append(res)
    return print('Done!')
    

def _h_core(mol: Union[gto.Cell, mgto.Mole], mf=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        this function returns the kin and nuc attraction ints
        """
        # kin ints
        if (isinstance(mol, gto.Cell) and isinstance(mf, scf.hf.RHF)):
            kin = mol.pbc_intor('int1e_kin')
            # individual atomic potentials
            mydf = mf.with_df
            #mydf = df.FFTDF(mol)
            sub_nuc = pbctools.get_nuc_atomic(mydf, kpts=np.zeros(3)) 
        elif isinstance(mol, mgto.Mole): 
            kin = mol.intor_symmetric('int1e_kin')
            # coordinates and charges of nuclei
            coords = mol.atom_coords()
            charges = mol.atom_charges()
            # individual atomic potentials
            sub_nuc = np.zeros([mol.natm, mol.nao_nr(), mol.nao_nr()], dtype=np.float64)
            for k in range(mol.natm):
                with mol.with_rinv_origin(coords[k]):
                    sub_nuc[k] = -1. * mol.intor('int1e_rinv') * charges[k]
        else:
            print('Wrong object passed to _h_core pbc')
        # total nuclear potential
        nuc = np.sum(sub_nuc, axis=0)
        return kin, nuc, sub_nuc 


##########################################
######### CELL OBJECT FOR TESTING ########
##########################################
#
# cell
cell = gto.Cell()
cell.atom = '''
 H   0.686524  1.000000  0.686524
 Cl  0.981476  1.000000  0.981476
'''
#cell.basis = 'sto3g'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = np.eye(3) * 4.18274
cell.a[1, 0] = -0.23273
cell.a[1, 1] = 4.17626
cell.a[2, 0] = -1.92719
cell.a[2, 1] = -2.13335
cell.a[2, 2] = 3.03810
cell.build()

supcell = super_cell(cell, [2, 2, 2])

mf = scf.RHF(supcell).density_fit()
ehf = mf.kernel()
dm = mf.make_rdm1()
print("HF energy (per supcell) = %.17g" % ehf)


## init decomp object for cell
#decomp = decodense.DecompCls(**PARAMS)
##print('decomp', dir(decomp))
######
#res = decodense.main(cell, decomp, mf)

# J, K int
J_int, K_int = mf.get_jk()
J_int *= .5
K_int *= -0.25
e_j = np.einsum('ij,ij', J_int, dm)
e_k = np.einsum('ij,ij', K_int, dm)

# kin, nuc atrraction 
# in decodense: glob>trace(sub_nuc_i, rdm1_tot), loc>trace(nuc,rdm1_atom_i)
kinetic, nuc, sub_nuc = _h_core(supcell, mf)
print('nuc dim', np.shape(nuc))
print('subnuc dim', np.shape(sub_nuc))
print('kin dim', np.shape(kinetic))
e_kin = np.einsum('ij,ij', kinetic, dm)
#
nuc_att_glob = np.einsum('ij,ij', nuc, dm)
nuc_att_glob *= .5
nuc_att_loc = np.einsum('xij,ij->x', sub_nuc, dm)
nuc_att_loc *= .5

##########################################
##########################################
## printing, debugging, etc.
nuc_att_ints = mf.get_nuc_att()
#nuc_att_ints, nuc_att_ints_atomic = mf.get_nuc_att()
cell_nuc_att = np.einsum('ij,ji', nuc_att_ints, dm)
#cell_nuc_att_atomic = np.einsum('zij,ji->z', nuc_att_ints_atomic, dm)
#print('cell_nuc_att_atomic ints ', np.shape(nuc_att_ints_atomic) )
print('CELL_NUC_ATT ', cell_nuc_att)
#print('CELL_NUC_ATT_ATOMIC ', cell_nuc_att_atomic)
#print('Their difference ', cell_nuc_att - np.einsum('z->', cell_nuc_att_atomic) )
###
###
#######print results
#####print(dir(res))
#print()
#print('Decodense res for cell')
#for k, v in res.items():
#    print(k, v)
print()
print('supcell')
print('energy_tot', mf.energy_tot())
print('energy_elec', mf.energy_elec())
print()



#
# the kinetic energy term for cell
print('CELL')
#print('e_nuc from decodense', np.sum(res['struct']) )
e_struct = pbctools.ewald_e_nuc(supcell)
print('e_kin as trace of T and D matrices (supcell): ', e_kin) 
#
# other terms
print('e_coul as trace of J and D matrices (supcell): ', e_j)
print('e_exch as trace of K and D matrices (supcell): ', e_k)
#
#print('nuc_att_glob as trace of (what would correspond to) sub_nuc and D: ', cell_nuc_att_atomic, np.einsum('z->', cell_nuc_att_atomic) )
#print('nuc_att_loc as trace of nuc and d's: ', 2*np.sum(nuc_att_loc) )
print('nuc_att as trace of nuc from pyscf and D: ', cell_nuc_att )
#print('local for cell computed here:')
#print(nuc_att_loc)
#
#E_total_cell = e_kin + e_j + e_k + 2.*nuc_att_glob + np.sum(res['struct'])
#print('e_kin + e_nuc + e_jk + e_nuc_att_glob ', E_total_cell)
#E_total_cell = e_kin + e_j + e_k + cell_nuc_att + np.sum(res['struct'])
E_total_cell = e_kin + e_j + e_k + cell_nuc_att + np.sum(e_struct)
print('e_kin + e_nuc + e_jk + e_nuc_att_glob + e_struct', E_total_cell)
print('PBC E_tot (here) - E_tot (pyscf) = ', E_total_cell - mf.energy_tot() )
#
print('TEST')
print('from hcore', np.einsum('ij,ij', mf.get_hcore(), dm))
print('my kin+nuc_att ', e_kin + cell_nuc_att )
print('difference hcore: ', np.einsum('ij,ij', mf.get_hcore(), dm) - (e_kin + cell_nuc_att) )
print('e_struct ', e_struct)
print('e_nuc ', supcell.energy_nuc() )
print('e_nuc - e_struct ',  supcell.energy_nuc() - np.sum(e_struct) )
#print(dir(mf))
#
print('e_nuc_att term test')
#vpp, vpp_atomic = mf.get_nuc_att()
vpp = mf.get_nuc_att()
mydf = mf.with_df
vpp_atomic, vloc1, vloc2, vnl = pbctools.get_pp_atomic(mydf)
print('vpp shape', np.shape(vpp) )
print('vpp_atomic shape', np.shape(vpp_atomic) )
e_nuc_att_pp = np.einsum('ij,ij', vpp, dm)
e_nuc_att_pp_atomic = np.einsum('zij,ij->z', vpp_atomic, dm)
print('e_nuc_att_pp', e_nuc_att_pp )
print('e_nuc_att_pp_atomic', np.sum(e_nuc_att_pp_atomic), e_nuc_att_pp_atomic )
print('e_nuc_att_pp - e_nuc_att_pp_atomic', e_nuc_att_pp - np.einsum('z->', e_nuc_att_pp_atomic) )

print('')
print('')
print('Testing v2 of e_nuc_pp atomic code')
vpp_atomic_v2, vloc1_v2_at, vloc2_v2_at, vnl_v2_at = pbctools.get_pp_atomic_v2(mydf)
vloc1_v2 = np.einsum('zij->ij', vloc1_v2_at)
vloc2_v2 = np.einsum('zij->ij', vloc2_v2_at)
vnl_v2 = np.einsum('zij->ij', vnl_v2_at)
print('vloc1, vloc1_v2, vloc1_v2_at shapes', np.shape(vloc1), np.shape(vloc1_v2), np.shape(vloc1_v2_at) )
print('all close vloc1?', np.allclose(np.einsum('zab->ab', vloc1), vloc1_v2, atol=1e-08) )
print('all close vloc1_at?', np.allclose(vloc1, vloc1_v2_at, atol=1e-08) )
print('')
#print('all close vloc2?', np.allclose(np.einsum('zab->ab', vloc1), vloc1_v2, atol=1e-58) )
#print('vloc2', np.shape(vloc2), np.shape(vloc2_v2) )
print('vloc2, vloc2_v2, vloc2_v2_at shapes', np.shape(vloc2), np.shape(vloc2_v2), np.shape(vloc2_v2_at) )
print('all close vloc2?', np.allclose(np.einsum('zab->ab', vloc2), vloc2_v2, atol=1e-14) )
print('all close vloc2_at?', np.allclose(vloc2, vloc2_v2_at, atol=1e-14) )
print('')
print('vpp_atomic, vpp_atomic_v2 shapes', np.shape(vpp_atomic), np.shape(vpp_atomic_v2) )
print('all close vpp_atomic?', np.allclose(vpp_atomic, vpp_atomic_v2, atol=1e-08) )
print('sum, all close vpp_atomic?', np.allclose(vpp_atomic, vloc1_v2_at+vloc2_v2_at+vnl_v2_at, atol=1e-08) )
check_decomp(supcell, mf)

