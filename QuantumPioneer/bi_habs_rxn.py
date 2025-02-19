import itertools as it

import numpy as np
from rdkit.Chem import AllChem
from rdmc import RDKitMol
from rdmc import ts as rdmc_ts

from QuantumPioneer import utils as qp_utils

FF = AllChem.ETKDGv3()
# this make sure we get different embedding each time
FF.randomSeed = np.random.randint(1, 10000000)


class BimolecularHydrogenAbstractionReaction:
    """
    A bi-molecular hydrogen abstraction reaction.

    Parameters
    ----------
    rxn_smi : str
        Atom-mapped reaction smiles.

    Attributes
    ----------
    r_complex : RDKitMol
        RDKitMol for reactant complex.
    p_complex : RDKitMol
        RDKitMol for product complex.
    ts_complex : RDKitMol
        RDKitMol for transition state complex.
    pivot_atoms : list
        Pivot atoms indices for the transition state.
    r_fragment_indices : tuple
        Atom indices for reactant fragments.
    p_fragment_indices : tuple
        Atom indices for product fragments.
    r1_neighbour_indices : list
        Atom indices for reactant fragment 1 neighbors.
    r2_neighbour_indices : list
        Atom indices for reactant fragment 2 neighbors.
    p1_neighbour_indices : list
        Atom indices for product fragment 1 neighbors.
    p2_neighbour_indices : list
        Atom indices for product fragment 2 neighbors.
    """

    def __init__(self, rxn_smi: str):
        # generate reactant and product complex RDkitMOl from smiles, atoms are always
        # zero-indexed, use mol.GetAtomMapNumbers() to get atom map specified in the
        # smiles
        r_complex, p_complex = [RDKitMol.FromSmiles(smi) for smi in rxn_smi.split(">>")]

        # perceive reaction center
        # formed, broken bonds indices e.g., fbond = [(1, 3)] means a bond forms between
        fbond, bbond = rdmc_ts.get_formed_and_broken_bonds(r_complex, p_complex)
        # atom with index 1 and 3, notice that atoms are zero-indexed and reaction is
        # analyzed in the forward direction
        # the H atom index in the TS
        the_h_atom = list(set(it.chain(*fbond)).intersection(it.chain(*bbond)))

        # TS pivot atom indices, not yet sorted by given reactants order
        _pivot_atoms = list(set(it.chain(*(fbond + bbond))).difference(the_h_atom))
        pivot_atoms = [None] * len(_pivot_atoms)

        # get atom indices in each molecule fragments, re-arranged to match the
        # reactants and products order specified in the reaction smile
        _frags_r = r_complex.GetMolFrags(asMols=False)  # reactants complex
        _frags_p = p_complex.GetMolFrags(asMols=False)  # products complex

        frags_r = [None] * len(_frags_r)
        frags_p = [None] * len(_frags_p)

        _reactants, _products = qp_utils.split_rxn_smi(rxn_smi=rxn_smi)
        _reactants_atom_map = [
            qp_utils.get_ordered_integers(x, sorted=True) for x in _reactants
        ]
        _products_atom_map = [
            qp_utils.get_ordered_integers(x, sorted=True) for x in _products
        ]

        r_complex_atom_map = r_complex.GetAtomMapNumbers()
        _frags_r_atom_map = [[r_complex_atom_map[i] for i in x] for x in _frags_r]

        for i, x in enumerate(_frags_r_atom_map):
            idx = _reactants_atom_map.index(x)
            frags_r[idx] = _frags_r[i]
        frags_r = tuple(frags_r)

        p_complex_atom_map = p_complex.GetAtomMapNumbers()
        _frags_p_atom_map = [[p_complex_atom_map[i] for i in x] for x in _frags_p]

        for i, x in enumerate(_frags_p_atom_map):
            idx = _products_atom_map.index(x)
            frags_p[idx] = _frags_p[i]
        frags_p = tuple(frags_p)

        # re-arrange pivot_atoms to match the reactants ordered in the smi;
        # pivot = R1 -- H(TS) -- R2; e.g., pivot = [15, 7] means atom with index 15 in
        # R1 and index 7 in R2 are atoms in the reaction coordinate
        for x in _pivot_atoms:
            idx = [x in f for f in frags_r].index(True)
            pivot_atoms[idx] = x

        # embed 3D geometry for reactants and products
        # here we use ETKDGv3() defined on top of the notebook, but can be changed to
        r_complex.EmbedConformer(FF)
        # others if needed
        p_complex.EmbedConformer(FF)
        # we need to add redundant bond to the reactant complex graph to represent the
        # TS
        ts_complex = r_complex.AddRedundantBonds(fbond)
        # geometry ts_complex.GetConformer() # embed TS conformer

        # get indices for neighbouring atoms of pivot atoms
        # ordered by how "bulky" the molecular fragment that the atom is connected to
        _r1_neighbour_indices = list(
            qp_utils.get_neighbour_atom(
                r_complex,
                center_atom_idx=pivot_atoms[0],
                exlude_atom_idx_list=the_h_atom,
            ).keys()
        )
        _r1_neighbour_indices_by_size = [
            (
                x,
                len(
                    qp_utils.find_fragment(
                        r_complex,
                        center_atom_idx=x,
                        exlude_atom_idx_list=[pivot_atoms[0]],
                    )
                ),
            )
            for x in _r1_neighbour_indices
        ]
        _r1_neighbour_indices_by_size.sort(key=lambda x: x[1], reverse=True)
        r1_neighbour_indices = [x[0] for x in _r1_neighbour_indices_by_size]

        _r2_neighbour_indices = list(
            qp_utils.get_neighbour_atom(
                r_complex,
                center_atom_idx=pivot_atoms[1],
                exlude_atom_idx_list=the_h_atom,
            ).keys()
        )
        _r2_neighbour_indices_by_size = [
            (
                x,
                len(
                    qp_utils.find_fragment(
                        r_complex,
                        center_atom_idx=x,
                        exlude_atom_idx_list=[pivot_atoms[1]],
                    )
                ),
            )
            for x in _r2_neighbour_indices
        ]
        _r2_neighbour_indices_by_size.sort(key=lambda x: x[1], reverse=True)
        r2_neighbour_indices = [x[0] for x in _r2_neighbour_indices_by_size]

        self.rxn_smi = rxn_smi
        self.r_complex = r_complex
        self.p_complex = p_complex
        self.ts_complex = ts_complex
        self.formed_bond = fbond  # R2OO
        self.broken_bond = bbond  # R1H
        self.ts_h_index = the_h_atom
        self.ts_pivot_indices = pivot_atoms
        self.r_fragment_indices = frags_r
        self.p_fragment_indices = frags_p
        self.r1_neighbour_indices = r1_neighbour_indices
        self.r2_neighbour_indices = r2_neighbour_indices
