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

def pstring_to_mpo(pstring, debug=False):

    As = []
    for p in pstring:
        pauli = Paulis[p]
        pauli_tensor = np.expand_dims(pauli, axis=(2, 3))
        if debug:
            print(p)
            print(pauli_tensor.shape)
        As.append(pauli_tensor)
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




if __name__=="__main__":
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
