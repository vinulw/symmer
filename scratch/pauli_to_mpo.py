import numpy as np
from copy import copy
from ncon import ncon

Paulis = {
        'I': np.eye(2, dtype=np.complex64),
        'X': np.array([[0, 1],
                       [1, 0]], dtype=np.complex64),
        'Y': np.array([[0, -1j],
                       [1j, 0]], dtype=np.complex64),
        'Z': np.array([[1, 0],
                       [0, -1]], dtype=np.complex64),
        }

def random_pauli_string(n):
    '''
    Preapre a random pauli string of length l
    '''
    alphabet = ['I', 'X', 'Y', 'Z']

    return np.random.choice(alphabet, size=n)

def pstrings_to_mpo(pstrings, coeffs=None, Dmax=None, debug=False):
    ''' Convert a list of Pauli Strings into an MPO. If coeff list is given,
    rescale each Pauli string by the corresponding element of the coeff list.
    Bond dim specifies the maximum bond dimension, if None, no maximum bond
    dimension.

    TODO: implement truncation for bond dim to work correctly
    '''
    if coeffs is None:
        coeffs = np.ones(len(pstrings))

    if Dmax is None:
        Dmax = np.inf

    mpo = pstring_to_mpo(pstrings[0], coeffs[0])

    i = 0
    centre = int(len(mpo) / 2)

    for pstr, coeff in zip(pstrings[1:], coeffs[1:]):
        _mpo = pstring_to_mpo(pstr, coeff)

        mpo = sum_mpo(mpo, _mpo)

        if debug:
            print("Summed mpo centre shape: {}".format(mpo[centre].shape))
        mpo = truncate_MPO(mpo, Dmax)
        if debug:
            print("Truncated centre mpo shape: {}".format(mpo[centre].shape))
            print("")


    return mpo


def truncated_SVD(M, Dmax=None):
    U, S, V = np.linalg.svd(M, full_matrices=False)

    if Dmax is not None and len(S) > Dmax:
        S = S[:Dmax]
        U = U[:, :Dmax]
        V = V[:Dmax, :]

    return U, S, V


def truncate_MPO(mpo, Dmax, debug=False):
    if debug:
        print(f"Curr Dmax: {Dmax}")
    As = []
    for n in range(len(mpo) - 1):  # Don't need to run on the last term
        A = mpo[n]
        σ, l, i, j = A.shape
        A = A.reshape(σ * l * i, j)
        U, S, V = truncated_SVD(A, Dmax)
        D = len(S)

        # Update current term
        _A = U.reshape(σ, l, i, D)
        As.append(_A)

        # Update the next term
        M = np.diag(S) @ V
        if debug:
            print(f"n: {n}")
            print(f"A: {mpo[n].shape}")
            print(f"M: {M.shape}")
            print(f"B: {mpo[n+1].shape}")
            print(f"A': {_A.shape}")
        _A1 = ncon([M, mpo[n+1]], ((-3, 1), (-1, -2, 1, -4)))
        mpo[n+1] = _A1

    As.append(mpo[-1])

    return As



def pstring_to_mpo(pstring, scaling=None, debug=False):

    As = []
    for p in pstring:
        pauli = Paulis[p]
        pauli_tensor = np.expand_dims(pauli, axis=(2, 3))
        if debug:
            print(p)
            print(pauli_tensor.shape)
        As.append(pauli_tensor)

    if scaling is not None:
        As = rescale_mpo(As, scaling)
    return As

def pstring_to_matrix(pstring):
    '''
    Construct a matrix out of the pauli string
    '''
    pmatrix = copy(Paulis[pstring[-1]])

    for p in reversed(pstring[:-1]):
        pmatrix = np.kron(Paulis[p], pmatrix)

    return pmatrix

def contract_mpo(mpo, debug=False):
    '''
    Contract mpo tensors along the virtual indices with shape (σ, l, i, j)
    where σ is the physical leg, l is the output leg and i and j are the
    remaining legs.
    '''
    contr = mpo[0]
    if debug:
        print(contr.shape)
    for tensor in mpo[1:]:
        σ1, l1, i1, j1 = contr.shape
        σ2, l2, i2, j2 = tensor.shape
        contr = ncon([contr, tensor], ((-1, -3, -5, 1), (-2, -4, 1, -6)))
        contr = np.reshape(contr, (σ1 * σ2, l1 * l2, i1, j2))
        if debug:
            print(contr.shape)

    contr = np.squeeze(contr)
    if debug:
        print('Squeezed...')
        print(contr.shape)
    return contr

def test_pauli_to_mpo(debug=False, n=5):
    pstring = random_pauli_string(n)

    pmatrix = pstring_to_matrix(pstring)

    pmpo = pstring_to_mpo(pstring, debug=debug)

    pmpo_matrix = contract_mpo(pmpo)

    if debug:
        print(pstring)
        print(np.allclose(pmatrix, pmpo_matrix))
    assert np.allclose(pmatrix, pmpo_matrix)

def sum_mpo(mpo1, mpo2, debug=False):
    summed = [None] * len(mpo1)
    σ10, l10, i10, j10 = mpo1[0].shape
    σ20, l20, i20, j20 = mpo2[0].shape
    t10 = copy(mpo1[0])
    t20 = copy(mpo2[0])
    first_sum = np.zeros((σ10, l10, i10, j10+j20), dtype=complex)
    first_sum[:, :, :, :j10] = t10
    first_sum[:, :, :, j10:] = t20
    summed[0] = first_sum

    σ1l, l1l, i1l, j1l = mpo1[-1].shape
    σ2l, l2l, i2l, j2l = mpo2[-1].shape
    t1l = copy(mpo1[-1])
    t2l = copy(mpo2[-1])
    first_sum = np.zeros((σ1l, l1l, i1l + i2l, j1l), dtype=complex)
    first_sum[:, :, :i1l, :] = t1l
    first_sum[:, :, i1l:, :] = t2l
    summed[-1] = first_sum
    for i in range(1, len(mpo1) - 1):
        σ1, l1, i1, j1 = mpo1[i].shape
        σ2, l2, i2, j2 = mpo2[i].shape
        t1 = copy(mpo1[i])
        t2 = copy(mpo2[i])


        new_shape = (σ1, l1, i1+i2, j1+j2)

        new_tensor = np.zeros(new_shape, dtype=complex)

        new_tensor[:, :, :i1, :j1] = t1
        new_tensor[:, :, i1:, j1:] = t2
        if debug:
            print(f"MPO1 Shape: {mpo1[i].shape}")
            print(f"MPO2 Shape: {mpo2[i].shape}")
            print(f"New Shape: {new_tensor.shape}")
            print("")
        summed[i] = new_tensor

    return summed

def rescale_mpo(mpo, amp):
    mpo[0] = mpo[0] * amp
    return mpo

def test_adding_two_mpos():
    import sys
    np.set_printoptions(threshold=sys.maxsize)

    # Testing addition of pauli strings
    n = 5
    amp1, amp2 = np.random.rand(2)
    print("Amplitudes")
    print(f"A1: {amp1}")
    print(f"A2: {amp2}")
    pstrings = [random_pauli_string(n) for _ in range(2)]

    mpo1, mpo2 = [pstring_to_mpo(pstring) for pstring in pstrings]

    # To add the amplitude just rescale the first element of the MPO by the
    # correct ammount. Equivalent to contracting an arbitrary scalar.
#    for i in range(1):
#        mpo1[i] = mpo1[i] * amp1
#        mpo2[i] = mpo2[i] * amp2
    mpo1 = rescale_mpo(mpo1, amp1)
    mpo2 = rescale_mpo(mpo2, amp2)

    summed = sum_mpo(mpo1, mpo2)

    print("Summed shapes:")
    for s in summed:
        print(s.shape)

    print("")


    matrix1, matrix2 = [pstring_to_matrix(pstring) for pstring in pstrings]

    assert np.allclose(amp1*matrix1, contract_mpo(mpo1))
    assert np.allclose(amp2*matrix2, contract_mpo(mpo2))

    summed_matrix = amp1*matrix1 + amp2*matrix2

    print("Summed close: {}".format(np.allclose(summed_matrix, contract_mpo(summed))))

    # Checking adding a third
    print("Adding another pauli string....")
    pstring3 = random_pauli_string(n)
    mpo3 = pstring_to_mpo(pstring3)
    amp3 = np.random.rand(1)
    mpo3 = rescale_mpo(mpo3, amp3)
    matrix3 = pstring_to_matrix(pstring3)

    assert np.allclose(amp3*matrix3, contract_mpo(mpo3))

    summed = sum_mpo(summed, mpo3)
    summed_matrix = summed_matrix + amp3*matrix3

    print("Summed shapes:")
    for s in summed:
        print(s.shape)

    print("")

    print("Summed allclose third: {}".format(np.allclose(summed_matrix, contract_mpo(summed))))

def PauliWordOp_to_paulis(WordOp):
    pstrings = []
    coeffs = []
    for pauli_vec, coeff in zip(WordOp.symp_matrix, WordOp.coeff_vec):
        pstrings.append(symplectic_to_string(pauli_vec))
        coeffs.append(coeff)

    return pstrings, coeffs

def symplectic_to_string(symp_vec) -> str:
    """
    Returns string form of symplectic vector defined as (X | Z)

    Args:
        symp_vec (array): symplectic Pauliword array

    Returns:
        Pword_string (str): String version of symplectic array

    """
    n_qubits = len(symp_vec) // 2

    X_block = symp_vec[:n_qubits]
    Z_block = symp_vec[n_qubits:]

    Y_loc = np.bitwise_and(X_block, Z_block).astype(bool)
    X_loc = np.bitwise_xor(Y_loc, X_block).astype(bool)
    Z_loc = np.bitwise_xor(Y_loc, Z_block).astype(bool)

    char_aray = np.array(list('I' * n_qubits), dtype=str)

    char_aray[Y_loc] = 'Y'
    char_aray[X_loc] = 'X'
    char_aray[Z_loc] = 'Z'

    Pword_string = ''.join(char_aray)

    return Pword_string

def coefflist_to_complex(coefflist):
    '''
    Convert a list of real + imaginary components into a complex vector
    '''
    arr = np.array(coefflist, dtype=complex)

    return arr[:, 0] + 1j*arr[:, 1]




def test_pauliwordop_to_mpo():
    import os
    from symmer.symplectic.base import PauliwordOp
    import json
    import matplotlib.pyplot as plt
    test_dir = os.path.join(os.path.dirname(os.getcwd()), 'tests')
    ham_data_dir = os.path.join(test_dir, 'hamiltonian_data')

    print(test_dir)
    print(ham_data_dir)

    filename = 'H4_STO-3G_SINGLET_JW.json'

    if filename not in os.listdir(ham_data_dir):
        raise ValueError('unknown file')

    with open(os.path.join(ham_data_dir, filename), 'r') as infile:
        data_dict = json.load(infile)


#    print(data_dict['hamiltonian'])
    pstrings, coefflist = zip(*data_dict['hamiltonian'].items())

    coeffs = coefflist_to_complex(coefflist)

    print("Number of terms: {}".format(len(pstrings)))

    print('Converting data dict')
    for pstr, coeff in zip(pstrings[:5], coeffs[:5]):
        print(f"{pstr}: {coeff}")

#    # Code to do the same conversion with Pauli WordOp
    wordop = PauliwordOp.from_dictionary(data_dict['hamiltonian'])
#    pstrings, coeffs = PauliWordOp_to_paulis(wordop)
#
#    print("From Pauli Wordop")
#    for pstr, coeff in zip(pstrings[:5], coeffs[:5]):
#        print(f"{pstr}: {coeff}")

    print("Preparing mpo...")
    mpo = pstrings_to_mpo(pstrings, coeffs, Dmax=None)

    print("Contracting mpo...")
    mpo_matrix = contract_mpo(mpo)

    print("Converting wordop to matrix...")
    wordop_matrix = wordop.to_sparse_matrix.toarray()

    print(np.allclose(mpo_matrix, wordop_matrix))

    def trace_distance(m1, m2):
        return np.trace((m1 - m2) * np.conj((m1 - m2).T))

    Dmaxs = [16, 32, 64, 128]

    tdists = []
    for D in Dmaxs:
        mpo = pstrings_to_mpo(pstrings, coeffs, Dmax=D)
        mpo_matrix = contract_mpo(mpo)
        tdists.append(trace_distance(wordop_matrix, mpo_matrix))
        print(f"{D}: {tdists[-1]}")

    plt.plot(Dmaxs, tdists, 'x--')
    plt.show()



if __name__=="__main__":
#    test_adding_two_mpos()
    test_pauliwordop_to_mpo()
