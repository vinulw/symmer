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

def pstring_to_mpo(pstring):

    As = []
    for p in pstring:
        print(p)
        pauli = Paulis[p]
        pauli_tensor = np.expand_dims(pauli, axis=(2, 3))
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

def contract_mpo(mpo):
    '''
    Contract mpo tensors along the virtual indices with shape (σ, l, i, j)
    where σ is the physical leg, l is the output leg and i and j are the
    remaining legs.
    '''
    contr = mpo[0]
    print(contr.shape)
    for tensor in mpo[1:]:
        σ1, l1, i1, j1 = contr.shape
        σ2, l2, i2, j2 = tensor.shape
        contr = ncon([contr, tensor], ((-1, -3, -5, 1), (-2, -4, 1, -6)))
        contr = np.reshape(contr, (σ1 * σ2, l1 * l2, i1, j2))
        print(contr.shape)

    contr = np.squeeze(contr)
    print('Squeezed...')
    print(contr.shape)
    return contr


if __name__=="__main__":

    for _ in range(10):
        pstring = random_pauli_string(5)
        print(''.join(pstring))

        pmatrix = pstring_to_matrix(pstring)

        pmpo = pstring_to_mpo(pstring)

        pmpo_matrix = contract_mpo(pmpo)

        print("Checking if allclose...")
        print(np.allclose(pmatrix, pmpo_matrix))
        print("")


