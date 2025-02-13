import itertools as it
import re
import typing as t

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
from rdmc import RDKitMol
from rdmc.ts import get_formed_and_broken_bonds

FF = AllChem.ETKDGv3()
# this make sure we get different embedding each time
FF.randomSeed = np.random.randint(1, 10000000)


# SMARTS patterns for substructure match
ROO_GROUP = "[H,C,N,O]-[O;X2]-[O;X1+0]"  # match ROO radical, with R = H, C, O, N only
ROOH_GROUP = "[*]-[O;X2]-[OH]"  # match any ROOH group
# using Xe as a None group that will match nothing; use when you do not care what
# the species is
NONE_GROUP = "[Xe]"
RADICAL_GROUP = "[CX3+0,NX2+0,OX1+0]"  # match radical, with R = C, O, N only


def adjust_atom_map_smi_indexing(
    rxn_smi: str,
    mode: str,
) -> str:
    """
    Helper function to adjust the indexing of atom mapped smiles. Useful for converting
    the smi to zero- or one- indexing.

    Parameters
    ----------
    rxn_smi : str
        Input atom-mapped reaction smiles.
    mode : str
        plus_one: increase all indexing by 1
        minus_one: reduce all indexing by 1

    Returns
    -------
    str
        Re-numbered smi string.
    """

    def decrement(m):
        return str(int(m.group().rstrip("]")) - 1) + "]"

    def increment(m):
        return str(int(m.group().rstrip("]")) + 1) + "]"

    # Use a regular expression to find all the integers and decrement or increment each
    # one. Note we search for "{number}]" to ensure that the number corresponds to an
    # atom index. Sometimes, SMILES strings include numbers to indicate connectivity
    # i.e. in rings.
    if mode == "plus_one":
        new_smi = re.sub(r"\d+]", lambda m: increment(m), rxn_smi)
    elif mode == "minus_one":
        new_smi = re.sub(r"\d+]", lambda m: decrement(m), rxn_smi)
    else:
        raise ValueError(
            f"Specificed mode {mode} not recognized. "
            "Must be either plus_one or minus_one"
        )

    return new_smi


def determine_atom_map_smi_indexing(
    rxn_smi: str,
) -> int:
    """
    Helper function to determine the starting index of an atom mapped reaction smile.
    Expect either zero- or one- indexing.

    Parameters
    ----------
    rxn_smi : str
        Input atom-mapped reaction smiles.

    Returns
    -------
    int
        Starting index.
    """

    # Extract all integers from the string
    integers = [int(match.rstrip("]")) for match in re.findall(r"\d+]", rxn_smi)]

    # Return the smallest integer if there's any integer in the string
    return min(integers) if integers else None


def split_rxn_smi(
    rxn_smi: str,
) -> t.Tuple[t.List[str], t.List[str]]:
    """
    Split a given reaction smile into reactant and product smiles in lists.

    Parameters
    ----------
    rxn_smi : str
        Input reaction smiles.

    Returns
    -------
    Union[t.List[str], t.List[str]]
        Smiles of individual reactants and products, in separate lists.
    """

    reactants = rxn_smi.split(">>")[0].split(".")
    products = rxn_smi.split(">>")[1].split(".")

    return reactants, products


def reorder_reaction_smile(
    rxn_smi: str,
    r_pattern: t.Sequence[str] = (),
    p_pattern: t.Sequence[str] = (),
) -> str:
    """
    Preprocess a reaction smile to a specified order by user.

    This function will separate a reaction smile into species, then reorder them into
    reactants and products based on SMARTS patterns specified by user, and finally
    return the ordered new reaction smile.

    Note: use NONE_GROUP if you do not care about which species to be matched in a
    particular position. Do not use a wild card [*] for this purpose, for it will match
    any species and the logic in this function does not work well with it.
    Make sure to test this function. It is common to make mistakes in SMARTS.

    Parameters
    ----------
    rxn_smi : str
        Atom-mapped reaction smiles.
    r_pattern : Optional[t.List[str]], optional
        SMARTS pattern for matching reactants, length must match number of reactants in
        rxn_smi.
    p_pattern : Optional[t.List[str]], optional
        SMARTS pattern for matching products, length must match number of products in
        rxn_smi.

    Returns
    -------
    str
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
        # turn pattern into rdkit molecule for substructure matching
        patt = Chem.MolFromSmarts(pattern)
        try:
            # return species index of the first match
            matched_idx = [
                bool(x) for x in [mol.GetSubstructMatch(patt) for mol in _r_mols]
            ].index(True)
        except ValueError:
            # NONE_group means the user does not care about which species ge
            if NONE_GROUP in r_pattern:
                # matched in the current index
                # place holder for index/order keeping, will be replaced by leftover
                reactants.append(None)
                # species later
                continue
            else:
                raise ValueError(
                    f"Pattern {pattern} not found in provided reaction species."
                )

        # add matched species to new reactants list
        reactants.append(_reactants[matched_idx])
        _r_mols.pop(matched_idx)
        _reactants.pop(matched_idx)
    # this means the user does not care about species order
    if reactants == [None, None]:
        # so we leave order unchanged
        reactants = _reactants
    # one of the species can be any left over species
    elif None in reactants:
        # replace the place holder with left over species
        reactants[reactants.index(None)] = _reactants[0]

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
    if products == [None, None]:
        products = _products
    elif None in products:
        products[products.index(None)] = _products[0]

    ordered_rxn_smi = ".".join(reactants) + ">>" + ".".join(products)
    return ordered_rxn_smi


def isomorphic_check(
    mol1: RDKitMol,
    mol2: RDKitMol,
) -> bool:
    """
    Compare if two rdkit molecules are the same.

    Parameters
    ----------
    mol1 : RDKitMol
    mol2 : RDKitMol

    Returns
    -------
    bool
        True if the same.
    """

    return mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1)


def get_ordered_integers(
    rxn_smi: str,
    sorted: bool = False,
) -> t.List[int]:
    """
    Get integers ordered by occurrence from a string. Useful to extract atom mapping
    from smiles.

    Parameters
    ----------
    rxn_smi : str
        Atom-mapped reaction smiles.
    sorted : bool, optional
        Whether to sort the integers, by default False.

    Returns
    -------
    t.List[int]
        Extracted integer list.
    """

    # Extract all integers from the string
    int_list = [int(match.rstrip("]")) for match in re.findall(r"\d+\]", rxn_smi)]
    if sorted:
        int_list.sort()

    return int_list


def get_neighbour_atom(mol, center_atom_idx, exlude_atom_idx_list=None):
    """
    Get neighboring atoms of a given atom in a molecule.

    Parameters
    ----------
    mol : RDKitMol
        The molecule.
    center_atom_idx : int
        Index of the center atom.
    exlude_atom_idx_list : list, optional
        t.List of atom indices to exclude, by default None.

    Returns
    -------
    dict
        Dictionary of neighboring atom indices and their symbols.
    """
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
    """
    Find the fragment of a molecule centered around a given atom.

    Parameters
    ----------
    mol : RDKitMol
        The molecule.
    center_atom_idx : int
        Index of the center atom.
    exlude_atom_idx_list : list
        t.List of atom indices to exclude.

    Returns
    -------
    dict
        Dictionary of fragment atom indices and their symbols.
    """
    frag = {}
    max_frag_size = len(mol.GetAtomicNumbers())
    center_atom = {center_atom_idx: mol.GetAtomWithIdx(center_atom_idx).GetSymbol()}

    frag = get_neighbour_atom(mol, center_atom_idx, exlude_atom_idx_list)

    search_list = list(frag.keys())
    to_exlude = []
    to_exlude.append(center_atom_idx)

    while search_list:
        for k in search_list:
            if mol.GetAtomWithIdx(k).GetSymbol() == "H":
                to_exlude.append(k)
                search_list = list({x for x in search_list if x not in to_exlude})
            neighbour = get_neighbour_atom(mol, k, to_exlude)
            search_list.extend(list(neighbour.keys()))
            to_exlude.append(k)
            search_list = list({x for x in search_list if x not in to_exlude})
            frag.update(neighbour)

            if len(frag.keys()) >= max_frag_size:
                frag.update(center_atom)
                return frag
    frag.update(center_atom)
    return frag


def perceive_rxn_generate_complex(
    rxn_smi: str,
) -> dict:
    """
    Perceive reaction and generate complex.

    Parameters
    ----------
    rxn_smi : str
        Atom-mapped reaction smiles.

    Returns
    -------
    dict
        Dictionary containing reaction information and complexes.
    """

    # generate reactant and product complex RDkitMOl from smiles, atoms are always
    # zero-indexed, use mol.GetAtomMapNumbers() to get atom map specified in the smiles
    r_complex, p_complex = [RDKitMol.FromSmiles(smi) for smi in rxn_smi.split(">>")]

    # perceive reaction center
    # formed, broken bonds indices e.g., fbond = [(1, 3)] means a bond forms between
    fbond, bbond = get_formed_and_broken_bonds(r_complex, p_complex)
    # atom with index 1 and 3, notice that atoms are zero-indexed and reaction is
    # analyzed in the forward direction
    # the H atom index in the TS
    the_h_atom = list(set(it.chain(*fbond)).intersection(it.chain(*bbond)))

    # TS pivot atom indices, not yet sorted by given reactants order
    _pivot_atoms = list(set(it.chain(*(fbond + bbond))).difference(the_h_atom))
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
    frags_r = tuple(frags_r)

    p_complex_atom_map = p_complex.GetAtomMapNumbers()
    _frags_p_atom_map = [[p_complex_atom_map[i] for i in x] for x in _frags_p]

    for i, x in enumerate(_frags_p_atom_map):
        idx = _products_atom_map.index(x)
        frags_p[idx] = _frags_p[i]
    frags_p = tuple(frags_p)

    # re-arrange pivot_atoms to match the reactants ordered in the smi;
    # pivot = R1 -- H(TS) -- R2; e.g., pivot = [15, 7] means atom with index 15 in R1
    # and index 7 in R2 are atoms in the reaction coordinate
    for x in _pivot_atoms:
        idx = [x in f for f in frags_r].index(True)
        pivot_atoms[idx] = x

    # embed 3D geometry for reactants and products
    # here we use ETKDGv3() defined on top of the notebook, but can be changed to
    r_complex.EmbedConformer(FF)
    # others if needed
    p_complex.EmbedConformer(FF)
    # we need to add redundant bond to the reactant complex graph to represent the TS
    ts_complex = r_complex.AddRedundantBonds(fbond)
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

    output = {}
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
