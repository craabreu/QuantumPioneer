"""
Unit and regression test for the BimolecularHydrogenAbstractionReaction class.
"""

# Import package, test suite, and other packages as needed

import re

import pytest

import QuantumPioneer as qp


@pytest.fixture
def hydrogen_abstraction_reaction():
    """
    Fixture that returns a BimolecularHydrogenAbstractionReaction instance
    """
    rnx_smi = (
        "[H:1][O:3][O:2].[H:4][N:8]([H:5])[C:6]#[N:7]"
        ">>"
        "[H:5][N:8][C:6]#[N:7].[H:1][O:3][O:2][H:4]"
    )
    return qp.BimolecularHydrogenAbstractionReaction(rnx_smi)


def test_instantiation(hydrogen_abstraction_reaction):
    """
    Test instantiation of BimolecularHydrogenAbstractionReaction class
    """
    rxn = hydrogen_abstraction_reaction

    assert rxn.formed_bond == [(1, 3)]
    assert rxn.broken_bond == [(3, 7)]
    assert rxn.ts_h_index == 3
    assert rxn.ts_pivot_indices == [7, 1]
    assert rxn.r_fragment_indices == ((3, 4, 5, 6, 7), (0, 1, 2))
    assert rxn.p_fragment_indices == ((4, 5, 6, 7), (0, 1, 2, 3))
    assert rxn.r1_neighbour_indices == [5, 4]
    assert rxn.r2_neighbour_indices == [2]


def test_get_conformer_xyz(hydrogen_abstraction_reaction):
    rxn = hydrogen_abstraction_reaction
    xyz = rxn._get_conformer_xyz(0)
    atoms = [re.split(r"\s+", line) for line in xyz.split("\n") if line]
    assert "".join(atom[0] for atom in atoms) == "HOOHHCNN"
    assert all(len(atom) == 4 for atom in atoms)
    try:
        [float(coord) for atom in atoms for coord in atom[1:4]]
    except ValueError as e:
        raise ValueError("Failed to convert coordinates to floats") from e


if __name__ == "__main__":
    test_instantiation()
