import os
from symmer.symplectic.base import PauliwordOp
import json
import numpy as np
from ncon import ncon
from functools import reduce
from operator import iconcat
from quimb.tensor.tensor_1d import MatrixProductOperator
from quimb.tensor.tensor_dmrg import DMRG2

def expand_local_basis(H):
    i, _ = H.shape
    i = int(np.log2(i))

    H_ = H.reshape([2]*(i*2))
    return H_


def group_opposite_legs(H_exp):
    l = len(H_exp.shape)
    offset = int(l / 2)
    tlist = [[i, i+offset] for i in range(offset)]
    tlist = reduce(iconcat, tlist, [])
    return H_exp.transpose(tlist)


def mpo_decomp_step(H_curr, Dmax=8):
    l = len(H_curr.shape)
    a, i, j = H_curr.shape[:3]

    M = H_curr.reshape(a*i*j, -1)

    U, S, V = truncated_SVD(M, Dmax=Dmax)

    D = len(S)

    A = U.reshape(a, i, j, D)
    M = np.diag(S) @ V

    M = M.reshape(D, *[2]*(l-4), 1)
    return A, M

def truncated_SVD(M, Dmax=None):
    U, S, V = np.linalg.svd(M, full_matrices=False)

    if Dmax is not None and len(S) > Dmax:
        S = S[:Dmax]
        U = U[:, :Dmax]
        V = V[:Dmax, :]

    return U, S, V

def construct_MPO(H_local, Dmax=8, debug=False):
    l = int(len(H_local.shape) / 2) - 1
    As = []
    M = H_local
    for i in range(l-1):
        A, M = mpo_decomp_step(M, Dmax)
        if debug:
            print(i)
            print('  A shape: {}'.format(A.shape))
            print('  M shape: {}'.format(M.shape))
        As.append(A)
    if debug:
        print(i+1)
        print('  A shape: {}'.format(M.shape))
    As.append(M)
    return As

def WordOpToMPO(wordop, max_bond_dim=16):
    H_sparse = H_op.to_sparse_matrix
    H = H_sparse.toarray()
    H_exp = expand_local_basis(H)
    H_grouped = group_opposite_legs(H_exp)
    H_grouped = np.expand_dims(H_grouped, axis=(0, -1))
    As = construct_MPO(H_grouped, Dmax=max_bond_dim, debug=False)
    As = [np.squeeze(A) for A in As]
    return MatrixProductOperator(As, 'ldur')

def load_hamiltonian_data(filename, ham_data_dir=None):
    if ham_data_dir is None:
        test_dir = os.path.join(os.path.dirname(os.getcwd()), 'tests')
        ham_data_dir = os.path.join(test_dir, 'hamiltonian_data')
    with open(os.path.join(ham_data_dir, filename), 'r') as infile:
        data_dict = json.load(infile)
    return data_dict

def prepare_filtered_list():
    test_dir = os.path.join(os.path.dirname(os.getcwd()), 'tests')
    ham_data_dir = os.path.join(test_dir, 'hamiltonian_data')

    files = [f for f in os.listdir(ham_data_dir) if os.path.isfile(os.path.join(ham_data_dir, f))]

    data_dicts = [load_hamiltonian_data(f) for f in files]

    qubit_list = []
    filtered_files = []
    for f, d in zip(files, data_dicts):
        n = d['data']['n_qubits']
        qubit_list.append(d['data']['n_qubits'])
        if n <= 10:
            print(f)
            filtered_files.append(f)


    print(np.mean(qubit_list))
    print(np.std(qubit_list))
    print(np.sum([n <= 10 for n in qubit_list]))

    print('Writing output')
    with open('filtered_hamiltonians_10qb.txt', 'w') as f:
        for filt in filtered_files:
            f.write(f'{filt}\n')

if __name__=="__main__":
    from symmer.symplectic import QuantumState
    from symmer.utils import exact_gs_energy
    import csv
    from datetime import datetime
    from tqdm import tqdm

    now = datetime.now().strftime("%d%m%y%H%M%S")


    output_fname = f"{now}_tn_gs_approx.csv"

    with open(output_fname, 'w') as f:
        f.write("File, GS Overlap, GS Energy, HF Energy, DMRG2 Energy\n")
    files = []
    with open('filtered_zero_overlap.txt', 'r') as f:
        for curr in f:
            files.append(curr.rstrip('\n'))

    data_dicts = [load_hamiltonian_data(f) for f in files]

    for fl, dct in zip(tqdm(files), data_dicts):
        tqdm.write(f'Calculating properties for {fl}...')
        H_op = PauliwordOp.from_dictionary(dct['hamiltonian'])
        MPO = WordOpToMPO(H_op, max_bond_dim=64)

        dmrg = DMRG2(MPO, bond_dims=[10, 20, 100, 100, 200], cutoffs=1e-10)
        dmrg.solve(verbosity=0, tol=1e-6)

        dmrg_state = dmrg.state.to_dense()
        dmrg_state = QuantumState.from_array(dmrg_state).cleanup(zero_threshold=1e-5)

        gs_energy, gs_vec = exact_gs_energy(H_op.to_sparse_matrix)
        gs_state = QuantumState.from_array(gs_vec).cleanup(zero_threshold=1e-5)

        gs_overlap = np.linalg.norm(gs_state.dagger * dmrg_state)

        gs_energy = -1*np.linalg.norm(gs_state.dagger * H_op * gs_state)
        dmrg_energy = -1*np.linalg.norm(dmrg_state.dagger * H_op * dmrg_state)
        hf_energy = dct['data']['calculated_properties']['HF']['energy']

        tqdm.write(f"GS Overlap: {gs_overlap}")
        tqdm.write(f"GS Energy: {gs_energy}")
        tqdm.write(f"HF Energy: {hf_energy}")
        tqdm.write(f"DMRG Energy: {dmrg_energy}")
        tqdm.write("")

        output_line = f"{fl}, {gs_overlap}, {gs_energy}, {hf_energy}, {dmrg_energy}\n"

        with open(output_fname, 'a') as f:
            f.write(output_line)
















