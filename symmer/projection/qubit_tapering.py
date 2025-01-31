import numpy as np
from typing import List, Union
from cached_property import cached_property
from symmer.projection import S3_projection
from symmer.symplectic import PauliwordOp, StabilizerOp

class QubitTapering(S3_projection):
    """ Class for performing qubit tapering as per https://arxiv.org/abs/1701.08213.
    Reduces the number of qubits in the problem whilst preserving its energy spectrum by:

    1. identifying a symmetry of the Hamiltonian,
    2. finding an independent basis therein,
    3. rotating each basis operator onto a single Pauli X, 
    4. dropping the corresponding qubits from the Hamiltonian whilst
    5. fixing the +/-1 eigenvalues

    Steps 1-2 are handled in this class whereas we defer to the parent S3_projection for 3-5.

    """
    def __init__(self,
            operator: PauliwordOp, 
            target_sqp: str = 'X'
        ) -> None:
        """ Input the PauliwordOp we wish to taper.
        There is freedom over the choice of single-qubit Pauli operator we wish to rotate onto, 
        however this is set to X by default (in line with the original tapering paper).
        """
        self.operator = operator
        self.target_sqp = target_sqp
        self.n_taper = self.symmetry_generators.n_terms
        super().__init__(self.symmetry_generators)
        
    @cached_property
    def symmetry_generators(self) -> StabilizerOp:
        """ Find an independent basis for the input operator symmetry
        """
        stabilizers = StabilizerOp.symmetry_basis(self.operator)
        stabilizers.target_sqp = self.target_sqp
        return stabilizers

    def taper_it(self,
            ref_state: Union[List[int], np.array]=None,
            sector: Union[List[int], np.array]=None,
            aux_operator: PauliwordOp = None
        ) -> PauliwordOp:
        """ Finally, once the symmetry generators and sector have been identified, 
        we may perform a projection onto the corresponding stabilizer subspace via 
        the parent S3_projection class.

        This method allows one to input an auxiliary operator other than the internal
        operator itself to be tapered consistently with the identified symmetry. This is 
        especially useful when considering an Ansatz defined over the full system that 
        one wishes to restrict to the same stabilizer subspace as the Hamiltonian for 
        use in VQE, for example.
        """
        # allow an auxiliary operator (e.g. an Ansatz) to be tapered
        if aux_operator is not None:
            operator_to_taper = aux_operator.copy()
        else:
            operator_to_taper = self.operator.copy()

        # taper the operator via S3_projection.perform_projection
        tapered_operator = self.perform_projection(
            operator=operator_to_taper,
            ref_state=ref_state,
            sector=sector
        )

        # if a reference state was supplied, taper it by dropping any
        # qubit positions fixed during the perform_projection method
        if ref_state is not None:
            ref_state = np.array(ref_state)
            self.tapered_ref_state = ref_state[self.free_qubit_indices]

        return tapered_operator