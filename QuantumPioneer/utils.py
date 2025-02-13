import itertools as it
import re
from typing import List, Optional, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
from rdmc import RDKitMol
from rdmc.ts import get_formed_and_broken_bonds

FF = AllChem.ETKDGv3()
FF.randomSeed = np.random.randint(
    1, 10000000
)  # this make sure we get different embedding each time


# SMARTS patterns for substructure match
ROO_GROUP = "[H,C,N,O]-[O;X2]-[O;X1+0]"  # match ROO radical, with R = H, C, O, N only
ROOH_GROUP = "[*]-[O;X2]-[OH]"  # match any ROOH group
NONE_GROUP = "[Xe]"  # using Xe as a None group that will match nothing; use when you do
# not care what the species is
RADICAL_GROUP = "[CX3+0,NX2+0,OX1+0]"  # match radical, with R = C, O, N only


def adjust_atom_map_smi_indexing(
    rxn_smi: str,
    mode: str,
) -> str:
    """
    Helper function to adjust the indexing of atom mapped smiles. Useful for converting
    the smi to zero- or one- indexing.

    Args:
        rxn_smi (str): input atom-mapped reaction smiles
        mode (str): plus_one: increase all indexing by 1
                    minus_one: reduce all indexing by 1

    Returns:
        str: re-numbered smi string
    """

    def decrement(m):
        return str(int(m.group().rstrip("]")) - 1) + "]"

    def increment(m):
        return str(int(m.group().rstrip("]")) + 1) + "]"

    # Use a regular expression to find all the integers and decrement or increment each one
    # Note we search for "{number}]" to ensure that the number corresponds to an atom index.
    # Sometimes, SMILES strings include numbers to indicate connectivity i.e. in rings.
    if mode == "plus_one":
        new_smi = re.sub(r"\d+]", lambda m: increment(m), rxn_smi)
    elif mode == "minus_one":
        new_smi = re.sub(r"\d+]", lambda m: decrement(m), rxn_smi)
    else:
        raise ValueError(
            f"Specificed mode {mode} not recognized. Must be either plus_one or minus_one"
        )

    return new_smi


def test_adjust_atom_map_smi_indexing():
    zero_idx_smi = (
        "[H:10][C:15]([H:11])([H:12])[C:16]([H:13])([H:14])[F:17]"
        "."
        "[H:0][C:5]([H:1])([H:2])[S:9][C:6]([H:3])([H:4])[O:8][O:7]"
        ">>"
        "[H:11][C:15]([H:12])[C:16]([H:13])([H:14])[F:17]"
        "."
        "[H:0][C:5]([H:1])([H:2])[S:9][C:6]([H:3])([H:4])[O:8][O:7][H:10]"
    )
    one_idx_smi = (
        "[H:11][C:16]([H:12])([H:13])[C:17]([H:14])([H:15])[F:18]"
        "."
        "[H:1][C:6]([H:2])([H:3])[S:10][C:7]([H:4])([H:5])[O:9][O:8]"
        ">>"
        "[H:12][C:16]([H:13])[C:17]([H:14])([H:15])[F:18]"
        "."
        "[H:1][C:6]([H:2])([H:3])[S:10][C:7]([H:4])([H:5])[O:9][O:8][H:11]"
    )
    assert (
        adjust_atom_map_smi_indexing(rxn_smi=zero_idx_smi, mode="plus_one")
        == one_idx_smi
    )
    assert (
        adjust_atom_map_smi_indexing(rxn_smi=one_idx_smi, mode="minus_one")
        == zero_idx_smi
    )

    zero_idx_smi = (
        "[C:0]([C:1]([C:2]([C:3]([Br:4])([H:22])[H:23])([H:20])[H:21])([H:18])[H:19])([H:15])([H:16])[H:17]"
        "."
        "[c:5]1([H:24])[c:6]([H:25])[c:7]([S-:8])[c:9]([H:26])[c:10]([H:27])[c:11]1[N+:12](=[O:13])[O-:14]"
    )
    one_idx_smi = (
        "[C:1]([C:2]([C:3]([C:4]([Br:5])([H:23])[H:24])([H:21])[H:22])([H:19])[H:20])([H:16])([H:17])[H:18]"
        "."
        "[c:6]1([H:25])[c:7]([H:26])[c:8]([S-:9])[c:10]([H:27])[c:11]([H:28])[c:12]1[N+:13](=[O:14])[O-:15]"
    )
    assert (
        adjust_atom_map_smi_indexing(rxn_smi=zero_idx_smi, mode="plus_one")
        == one_idx_smi
    )
    assert (
        adjust_atom_map_smi_indexing(rxn_smi=one_idx_smi, mode="minus_one")
        == zero_idx_smi
    )


test_adjust_atom_map_smi_indexing()


def determine_atom_map_smi_indexing(
    rxn_smi: str,
) -> int:
    """
    Helper function to determine the starting index of an atom mapped reaction smile.
    Expect either zero- or one- indexing.

    Args:
        rxn_smi (str): input atom-mapped reaction smiles

    Returns:
        int: starting index
    """

    # Extract all integers from the string
    integers = [int(match.rstrip("]")) for match in re.findall(r"\d+]", rxn_smi)]

    # Return the smallest integer if there's any integer in the string
    return min(integers) if integers else None


def test_determine_atom_map_smi_indexing():
    zero_idx_smi = (
        "[H:10][C:15]([H:11])([H:12])[C:16]([H:13])([H:14])[F:17]"
        "."
        "[H:0][C:5]([H:1])([H:2])[S:9][C:6]([H:3])([H:4])[O:8][O:7]"
        ">>"
        "[H:11][C:15]([H:12])[C:16]([H:13])([H:14])[F:17]"
        "."
        "[H:0][C:5]([H:1])([H:2])[S:9][C:6]([H:3])([H:4])[O:8][O:7][H:10]"
    )
    one_idx_smi = (
        "[H:11][C:16]([H:12])([H:13])[C:17]([H:14])([H:15])[F:18]"
        "."
        "[H:1][C:6]([H:2])([H:3])[S:10][C:7]([H:4])([H:5])[O:9][O:8]"
        ">>"
        "[H:12][C:16]([H:13])[C:17]([H:14])([H:15])[F:18]"
        "."
        "[H:1][C:6]([H:2])([H:3])[S:10][C:7]([H:4])([H:5])[O:9][O:8][H:11]"
    )

    assert determine_atom_map_smi_indexing(rxn_smi=zero_idx_smi) == 0
    assert determine_atom_map_smi_indexing(rxn_smi=one_idx_smi) == 1

    two_idx_smi = (
        "[C:2]([C:3]([C:4]([C:5]([Br:6])([H:24])[H:25])([H:22])[H:23])([H:20])[H:21])([H:17])([H:18])[H:19]"
        "."
        "[c:7]1([H:26])[c:8]([H:27])[c:9]([S-:10])[c:11]([H:28])[c:12]([H:29])[c:13]1[N+:14](=[O:15])[O-:16]"
    )
    assert determine_atom_map_smi_indexing(rxn_smi=two_idx_smi) == 2


test_determine_atom_map_smi_indexing()


def split_rxn_smi(
    rxn_smi: str,
) -> Union[List[str], List[str]]:
    """
    Split a given reaction smile into reactant and product smiles in lists.

    Args:
        rxn_smi (str): input reaction smiles

    Returns:
        Union[List[str], List[str]]: smiles of individual reactants and products, in sepearte lists
    """

    reactants = rxn_smi.split(">>")[0].split(".")
    products = rxn_smi.split(">>")[1].split(".")

    return reactants, products


def reorder_reaction_smile(
    rxn_smi: str,
    r_pattern: Optional[List[str]] = [],
    p_pattern: Optional[List[str]] = [],
) -> str:
    """
    Preprocess a reaction smile to a specified order by user.

    This function will seperate a reaction smile into speices, then reorder them into
    reactants and products based on SMARTS patterns specified by user, and finally
    return the ordered new reaction smile.

    Note: use NONE_GROUP if you do not care about which species to be matched in a
    particular position. Do not use a wild card [*] for this purpose, for it will match
    any species and the logic in this function does not work well with it.
          Make sure to test this function. It is common to make mistakes in SMARTS.

    Input:
        rxn_smi: atom-mapped reaction smiles
        r_pattern (optional): SMARTS pattern for matching reactants, length must match
        number of reactants in rxn_smi
        p_pattern (optional): SMARTS pattern for matching products, length must match
        number of products in rxn_smi

    Output:
        New reaction smiles with species match the order of the pattern specified.
    """

    # return original smile if no pattern to match
    if not any([len(r_pattern), len(p_pattern)]):
        return rxn_smi

    # get reactants and products from provided smi
    _reactants, _products = split_rxn_smi(rxn_smi=rxn_smi)

    # raise if provided number of patterns mismatch given smi
    if not all([len(_reactants) == len(r_pattern), len(_products) == len(p_pattern)]):
        raise ValueError(
            "Provided number of patterns does not match number of species "
            "in the given reaction."
        )

    # turn smi into rdkit molecule for matching
    _r_mols = [MolFromSmiles(smi) for smi in _reactants]
    _p_mols = [MolFromSmiles(smi) for smi in _products]

    reactants = []
    products = []

    # match reactant pattern
    for pattern in r_pattern:
        patt = Chem.MolFromSmarts(
            pattern
        )  # turn pattern into rdkit molecule for substructure matching
        try:
            matched_idx = [
                bool(x) for x in [mol.GetSubstructMatch(patt) for mol in _r_mols]
            ].index(
                True
            )  # return species index of the first match
        except ValueError:
            if (
                NONE_GROUP in r_pattern
            ):  # NONE_group means the user does not care about which species ge
                # matched in the current index
                reactants.append(
                    None
                )  # place holder for index/order keeping, will be replaced by leftover
                # species later
                continue
            else:
                raise ValueError(
                    f"Pattern {pattern} not found in provided reaction species."
                )

        reactants.append(
            _reactants[matched_idx]
        )  # add matched species to new reactants list
        _r_mols.pop(matched_idx)
        _reactants.pop(matched_idx)
    else:
        if reactants == [
            None,
            None,
        ]:  # this means the user does not care about species order
            reactants = _reactants  # so we leave order unchanged
        elif None in reactants:  # one of the species can be any left over species
            reactants[reactants.index(None)] = _reactants[
                0
            ]  # replace the place holder with left over species

    # match product pattern, same logic as reactant
    for pattern in p_pattern:
        patt = Chem.MolFromSmarts(pattern)
        try:
            matched_idx = [
                bool(x) for x in [mol.GetSubstructMatch(patt) for mol in _p_mols]
            ].index(True)
        except ValueError:
            if NONE_GROUP in p_pattern:
                products.append(None)
                continue
            else:
                raise ValueError(
                    f"Pattern {pattern} not found in provided reaction species."
                )
        products.append(_products[matched_idx])
        _p_mols.pop(matched_idx)
        _products.pop(matched_idx)
    else:
        if products == [None, None]:
            products = _products
        elif None in products:
            products[products.index(None)] = _products[0]

    ordered_rxn_smi = ".".join(reactants) + ">>" + ".".join(products)
    return ordered_rxn_smi


def test_reorder_reaction_smile():
    input_rxn_smi = (
        "[H:1][C:6]([H:2])([H:3])[S:10][C:7]([H:4])([H:5])[O:9][O:8]"
        "."
        "[H:11][C:16]([H:12])([H:13])[C:17]([H:14])([H:15])[F:18]"
        ">>"
        "[H:1][C:6]([H:2])([H:3])[S:10][C:7]([H:4])([H:5])[O:9][O:8][H:11]"
        "."
        "[H:12][C:16]([H:13])[C:17]([H:14])([H:15])[F:18]"
    )
    expected_rxn_smi = (
        "[H:11][C:16]([H:12])([H:13])[C:17]([H:14])([H:15])[F:18]"
        "."
        "[H:1][C:6]([H:2])([H:3])[S:10][C:7]([H:4])([H:5])[O:9][O:8]"
        ">>"
        "[H:12][C:16]([H:13])[C:17]([H:14])([H:15])[F:18]"
        "."
        "[H:1][C:6]([H:2])([H:3])[S:10][C:7]([H:4])([H:5])[O:9][O:8][H:11]"
    )

    r_pattern = [NONE_GROUP, ROO_GROUP]
    p_pattern = [RADICAL_GROUP, ROOH_GROUP]

    assert (
        reorder_reaction_smile(
            rxn_smi=input_rxn_smi, r_pattern=r_pattern, p_pattern=p_pattern
        )
        == expected_rxn_smi
    )


test_reorder_reaction_smile()


def isomorphic_check(
    mol1: RDKitMol,
    mol2: RDKitMol,
) -> bool:
    """
    Compare if two rdkit molecules are the same.

    Args:
        mol1 (RDKitMol)
        mol2 (RDKitMol)

    Returns:
        bool: True if the same.
    """

    return mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1)


def test_isomorphic_check():
    smi1 = "c1ccc2c(c1)c3ccccc3[nH]2"
    smi2 = "C1=CC=C2C(=C1)C3=CC=CC=C3N2"
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)

    assert isomorphic_check(mol1, mol2)

    smi1 = "CCC"
    smi2 = "C1=CC=C2C(=C1)C3=CC=CC=C3N2"
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)

    assert not isomorphic_check(mol1, mol2)

    smi1 = "[H:10][C:15]([H:11])([H:12])[C:16]([H:13])([H:14])[F:17]"
    smi2 = "[H:20][C:45]([H:11])([H:12])[C:16]([H:13])([H:14])[F:17]"
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)

    assert isomorphic_check(mol1, mol2)


test_isomorphic_check()


def get_ordered_integers(
    rxn_smi: str,
    sorted: bool = False,
) -> List[int]:
    """
    Get integers ordered by occurence from a string. Useful to extract atom mapping from smiles.

    Args:
        rxn_smi (str): atom-mapped reaction smiles

    Returns:
        List[int]: extracted integer list
    """

    # Extract all integers from the string
    int_list = [int(match.rstrip("]")) for match in re.findall(r"\d+\]", rxn_smi)]
    if sorted:
        int_list.sort()

    return int_list


def test_get_ordered_integers():
    smi1 = "[H:10][C:15]([H:11])([H:12])[C:16]([H:13])([H:14])[F:17]"
    assert get_ordered_integers(smi1) == [10, 15, 11, 12, 16, 13, 14, 17]

    smi2 = "[c:5]1([H:24])[c:6]([H:25])[c:7]([S-:8])[c:9]([H:26])[c:10]([H:27])[c:11]1[N+:12](=[O:13])[O-:14]"
    assert get_ordered_integers(smi2) == [
        5,
        24,
        6,
        25,
        7,
        8,
        9,
        26,
        10,
        27,
        11,
        12,
        13,
        14,
    ]


test_get_ordered_integers()


def get_neighbour_atom(mol, center_atom_idx, exlude_atom_idx_list=None):
    neighbour = {
        nb.GetIdx(): mol.GetAtomWithIdx(nb.GetIdx()).GetSymbol()
        for nb in mol.GetAtomWithIdx(center_atom_idx).GetNeighbors()
    }

    if exlude_atom_idx_list:
        for x in exlude_atom_idx_list:
            if x in neighbour:
                del neighbour[x]

    return neighbour


def find_fragment(mol, center_atom_idx, exlude_atom_idx_list):
    frag = dict()
    max_frag_size = len(mol.GetAtomicNumbers())
    center_atom = {center_atom_idx: mol.GetAtomWithIdx(center_atom_idx).GetSymbol()}

    frag = get_neighbour_atom(mol, center_atom_idx, exlude_atom_idx_list)

    search_list = list(frag.keys())
    to_exlude = list()
    to_exlude.append(center_atom_idx)

    while search_list:
        for k in search_list:
            if mol.GetAtomWithIdx(k).GetSymbol() == "H":
                to_exlude.append(k)
                search_list = list(set([x for x in search_list if x not in to_exlude]))
            neighbour = get_neighbour_atom(mol, k, to_exlude)
            search_list.extend(list(neighbour.keys()))
            to_exlude.append(k)
            search_list = list(set([x for x in search_list if x not in to_exlude]))
            frag.update(neighbour)

            if len(frag.keys()) >= max_frag_size:
                frag.update(center_atom)
                return frag
    else:
        frag.update(center_atom)
        return frag


def perceive_rxn_generate_complex(
    rxn_smi: str,
) -> dict:
    """
    perceive_rxn_generate_complex _summary_

    _extended_summary_

    Args:
        rxn_smi (str): _description_

    Returns:
        dict: _description_
    """

    # generate reactant and product complex RDkitMOl from smiles, atoms are always
    # zero-indexed, use mol.GetAtomMapNumbers() to get atom map specified in the smiles
    r_complex, p_complex = [RDKitMol.FromSmiles(smi) for smi in rxn_smi.split(">>")]

    # perceive reaction center
    fbond, bbond = get_formed_and_broken_bonds(
        r_complex, p_complex
    )  # formed, broken bonds indices e.g., fbond = [(1, 3)] means a bond forms between
    # atom with index 1 and 3, notice that atoms are zero-indexed and reaction is
    # analyzed in the forward direction
    the_h_atom = list(
        set(it.chain(*fbond)).intersection(it.chain(*bbond))
    )  # the H atom index in the TS

    _pivot_atoms = list(
        set(it.chain(*(fbond + bbond))).difference(the_h_atom)
    )  # TS pivot atom indices, not yet sorted by given reactants order
    pivot_atoms = [None] * len(_pivot_atoms)

    # get atom indices in each molecule fragments, re-arranged to match the reactants
    # and products order specified in the reaction smile
    _frags_r = r_complex.GetMolFrags(asMols=False)  # reactants complex
    _frags_p = p_complex.GetMolFrags(asMols=False)  # products complex

    frags_r = [None] * len(_frags_r)
    frags_p = [None] * len(_frags_p)

    _reactants, _products = split_rxn_smi(rxn_smi=rxn_smi)
    _reactants_atom_map = [get_ordered_integers(x, sorted=True) for x in _reactants]
    _products_atom_map = [get_ordered_integers(x, sorted=True) for x in _products]

    r_complex_atom_map = r_complex.GetAtomMapNumbers()
    _frags_r_atom_map = [[r_complex_atom_map[i] for i in x] for x in _frags_r]

    for i, x in enumerate(_frags_r_atom_map):
        idx = _reactants_atom_map.index(x)
        frags_r[idx] = _frags_r[i]
    else:
        frags_r = tuple(frags_r)

    p_complex_atom_map = p_complex.GetAtomMapNumbers()
    _frags_p_atom_map = [[p_complex_atom_map[i] for i in x] for x in _frags_p]

    for i, x in enumerate(_frags_p_atom_map):
        idx = _products_atom_map.index(x)
        frags_p[idx] = _frags_p[i]
    else:
        frags_p = tuple(frags_p)

    # re-arrange pivot_atoms to match the reactants ordered in the smi;
    # pivot = R1 -- H(TS) -- R2; e.g., pivot = [15, 7] means atom with index 15 in R1
    # and index 7 in R2 are atoms in the reaction coordinate
    for i, x in enumerate(_pivot_atoms):
        idx = [x in f for f in frags_r].index(True)
        pivot_atoms[idx] = _pivot_atoms[i]

    # embed 3D geometry for reactants and products
    r_complex.EmbedConformer(
        FF
    )  # here we use ETKDGv3() defined on top of the notebook, but can be changed to
    # others if needed
    p_complex.EmbedConformer(FF)
    ts_complex = r_complex.AddRedundantBonds(
        fbond
    )  # we need to add redundant bond to the reactant complex graph to represent the TS
    # geometry ts_complex.GetConformer() # embed TS conformer

    # get indices for neighbouring atoms of pivot atoms
    # ordered by how "bulky" the molecular fragment that the atom is connected to
    _r1_neighbour_indices = list(
        get_neighbour_atom(
            r_complex, center_atom_idx=pivot_atoms[0], exlude_atom_idx_list=the_h_atom
        ).keys()
    )
    _r1_neighbour_indices_by_size = [
        (
            x,
            len(
                find_fragment(
                    r_complex, center_atom_idx=x, exlude_atom_idx_list=[pivot_atoms[0]]
                )
            ),
        )
        for x in _r1_neighbour_indices
    ]
    _r1_neighbour_indices_by_size.sort(key=lambda x: x[1], reverse=True)
    r1_neighbour_indices = [x[0] for x in _r1_neighbour_indices_by_size]

    _r2_neighbour_indices = list(
        get_neighbour_atom(
            r_complex, center_atom_idx=pivot_atoms[1], exlude_atom_idx_list=the_h_atom
        ).keys()
    )
    _r2_neighbour_indices_by_size = [
        (
            x,
            len(
                find_fragment(
                    r_complex, center_atom_idx=x, exlude_atom_idx_list=[pivot_atoms[1]]
                )
            ),
        )
        for x in _r2_neighbour_indices
    ]
    _r2_neighbour_indices_by_size.sort(key=lambda x: x[1], reverse=True)
    r2_neighbour_indices = [x[0] for x in _r2_neighbour_indices_by_size]

    output = dict()
    output["rxn_smi"] = rxn_smi
    output["r_complex"] = r_complex
    output["p_complex"] = p_complex
    output["ts_complex"] = ts_complex
    output["formed_bond"] = fbond  # R2OO
    output["broken_bond"] = bbond  # R1H
    output["ts_h_index"] = the_h_atom
    output["ts_pivot_indices"] = pivot_atoms
    output["r_fragment_indices"] = frags_r
    output["p_fragment_indices"] = frags_p
    output["r1_neighbour_indices"] = r1_neighbour_indices
    output["r2_neighbour_indices"] = r2_neighbour_indices

    return output
