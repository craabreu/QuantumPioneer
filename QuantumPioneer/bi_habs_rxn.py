import itertools as it

import numpy as np
from rdkit.Chem import AllChem
from rdmc import RDKitMol
from rdmc import forcefield as rdmc_ff
from rdmc import ts as rdmc_ts

from QuantumPioneer import utils as qp_utils

from rdmc import RDKitMol

from rdmc.forcefield import RDKitFF

import numpy as np
import copy


FF = AllChem.ETKDGv3()
# this make sure we get different embedding each time
FF.randomSeed = np.random.randint(1, 10000000)


# Empirical parameters
_DEFAULT_TS_GUESS_PARAMETERS = {
    # initial angle for TS pivot. Do not make it too close to 180.
    "angle_X_H_Y": 160,
    # initial dihedral angle for TS, defined using r1, does not really matter most of
    # the time for bi-molecular H abstraction.
    "dihedral_r1": 0,
    # the other diehdral angle, defined using reactant 2 or r2
    "dihedral_r2": 0,
    # how much to scale the TS bond length based on reactant bond length for r1; can be
    # modified to scale each bond differently, but usually 1.2-1.25 is good guess for
    # bi-molecular H abstraction with C-H and 1.15 - 1.2 for O-H.
    "bond_length_scale_factor_r1": 1.22,
    # how much to scale the TS bond length based on reactant bond length for r1; can be
    # modified to scale each bond differently, but usually 1.2-1.25 is good guess for
    # bi-molecular H abstraction with C-H and 1.15 - 1.2 for O-H.
    "bond_length_scale_factor_r2": 1.19,
    # directly specifiy the TS bond length for r1, useful if another bond length
    # estimator e.g., TS-EGNN/Chemprop/Kinbot can provide this information
    "bond_length_X_H": None,
    # directly specifiy the TS bond length for r2
    "bond_length_H_Y": None,
}



# %%
# this function calcualte R-H bond distances for closed-shell reactant/product species based on averaging of N conformers embedded from force field
# a helper function used in gen_ts
def return_opt_spc_bond_distance(
    spc_smi,
    pivot_atom,
    the_h_atom,
    atom_map_idx,
    averaged = 5,
    ):

    """
    Return the optimized bond distance of reacting R-H from reactant/product species. Only use on non-radical RH species.
    """

    bd_list = list()

    while averaged:
        r = RDKitMol.FromSmiles(spc_smi)
        ff = RDKitFF('mmff94s')
        r.EmbedConformer(qp_utils.FF)
        ff.setup(r)
        ff.optimize()
        m = ff.get_optimized_mol()

        r_conf = r.GetConformer()
        r_conf.SetPositions(m.GetPositions())

        h_idx = atom_map_idx.index(the_h_atom)
        pivot_idx = atom_map_idx.index(pivot_atom)

        bd = r_conf.GetBondLength([h_idx, pivot_idx])
        bd_list.append(bd)
        averaged -= 1

    return sum(bd_list)/len(bd_list)

# %%
def initialize_ts_guess_geometry(
    empirical_para_dict,
    species_complex_dict,
):

    _reactants, _products = qp_utils.split_rxn_smi(rxn_smi=species_complex_dict['rxn_smi'])
    ts_conformer = species_complex_dict['ts_complex'].GetConformer()

    # set TS bond distances
    # update bond distance of r1 -- H(ts)
    # step 1: find the reacting bond distance of r1 when it's not in the TS based on averaging of 10 conformers
    r1_dist = return_opt_spc_bond_distance(spc_smi = _reactants[0],
                            pivot_atom = species_complex_dict['ts_pivot_indices'][0],
                            the_h_atom = species_complex_dict['ts_h_index'][0],
                            atom_map_idx = species_complex_dict['r_fragment_indices'][0],
                            )

    # step 2: scale the reacting bond with a scaling factor, the factor is provided by user currently, but it can be learned since it depends on reactant/product type (usually 25% is good starting guess for C-H, 15% for O-H etc)
    bond_length_X_H = r1_dist * empirical_para_dict['bond_length_scale_factor_r1']

    # step 3: set bond length for the TS complex
    ts_conformer.SetBondLength([species_complex_dict['ts_pivot_indices'][0],
                                                      species_complex_dict['ts_h_index'][0],
                                                      ],
                                                     bond_length_X_H)

    # update bond distance of H(ts) -- r2
    r2_dist = return_opt_spc_bond_distance(spc_smi = _products[1],
                            pivot_atom = species_complex_dict['ts_pivot_indices'][1],
                            the_h_atom = species_complex_dict['ts_h_index'][0],
                            atom_map_idx = species_complex_dict['p_fragment_indices'][1],
                            )


    bond_length_H_Y = r2_dist * empirical_para_dict['bond_length_scale_factor_r2']

    ts_conformer.SetBondLength([species_complex_dict['ts_pivot_indices'][1],
                                                      species_complex_dict['ts_h_index'][0],
                                                      ],
                                                     bond_length_H_Y)


    # set TS angle
    ts_conformer.SetAngleDeg([species_complex_dict['ts_pivot_indices'][0],
                                                    species_complex_dict['ts_h_index'][0],
                                                    species_complex_dict['ts_pivot_indices'][1]],
                                                    empirical_para_dict['angle_X_H_Y']) # angle_X_H_Y can be learned, but usually 160 deg is good for most bi-mol H abs family

    # set TS dihedral
    # note: there are two sets of dihedrals for the TS, since a dihedral is defined by 4 connecting atoms, and there are 5 atoms connected in the TS region, thus 2 sets of dihedrals

    # set dihedrals
    # note: the dihedrals can actually be searched or learned, but not implemented yet, currently we set to user provided value
    ts_conformer.SetTorsionDeg([species_complex_dict['r1_neighbour_indices'][0],
                                                      species_complex_dict['ts_pivot_indices'][0],
                                                      species_complex_dict['ts_h_index'][0],
                                                      species_complex_dict['ts_pivot_indices'][1],
                                                      ],
                                                     empirical_para_dict['dihedral_r1'])

    ts_conformer.SetTorsionDeg([species_complex_dict['ts_pivot_indices'][0],
                                                      species_complex_dict['ts_h_index'][0],
                                                      species_complex_dict['ts_pivot_indices'][1],
                                                      species_complex_dict['r2_neighbour_indices'][0],
                                                      ],
                                                     empirical_para_dict['dihedral_r2'])

    return species_complex_dict

# %%
# this function check if the TS guess has atoms colliding with each other
# 0.4 anstrom is an empirical parameter
def check_hard_collision(ts, threshold=0.4):
    if ts.HasCollidingAtoms(threshold=threshold):
        raise ValueError('Atom collision detected.')

# %%
# a helper function to optimize the transition state using force field
# note: this is a constrained optimization that fixes the two TS bond distances first,
# then, we optimize the geometry to a stable point to relax it (not to a saddle point as in typical ts search)
# the idea here is to relax each fragment of the ts to make the geometry more reasonable
def opt_ts_ff(species_complex_dict):

    ts_complex = species_complex_dict['ts_complex']
    ts_conformer = ts_complex.GetConformer()

    bond_length_X_H = ts_conformer.GetBondLength(species_complex_dict['broken_bond'][0])
    bond_length_H_Y = ts_conformer.GetBondLength(species_complex_dict['formed_bond'][0])

    ff = RDKitFF('mmff94s')
    fake_ts = RDKitMol.FromSmiles(species_complex_dict['rxn_smi'].split('>>')[0])
    fake_ts.EmbedConformer()
    fake_ts.SetPositions(ts_complex.GetPositions())
    ff.setup(fake_ts)

    the_h_atom = species_complex_dict['ts_h_index'][0]
    pivot_atom_r1 = species_complex_dict['ts_pivot_indices'][0]
    pivot_atom_r2 = species_complex_dict['ts_pivot_indices'][1]

    ff.add_distance_constraint(atoms=[the_h_atom,pivot_atom_r1], value=bond_length_X_H)
    ff.add_distance_constraint(atoms=[the_h_atom,pivot_atom_r2], value=bond_length_H_Y)

    ff.optimize()
    m = ff.get_optimized_mol()

    ts_new = copy.deepcopy(ts_complex)
    ts_new.SetPositions(m.GetPositions())

    return ts_new

# %%
# this function check if two parts of the TS are too close to each other
# since the TS bond distance is about 1.3 anstrom for C-H, if two fragements are closer than this, additional bonds may form (collision)
def check_fragment_collision(ts, h_idx, threshold=1.3):
    distance_matrix = np.triu(ts.GetDistanceMatrix())
    index = h_idx # atom index of the ts h
    fragment_bond_distance_matrix = distance_matrix[0:index, index:] # check if two fragments are too close to each other
    if np.any(fragment_bond_distance_matrix < threshold):
        raise ValueError('Reactants collision detected.')
    else:
        small_distances = fragment_bond_distance_matrix[fragment_bond_distance_matrix < 2]
        relax_score = np.sum((small_distances-2)**2) + np.min(small_distances) + np.max(fragment_bond_distance_matrix) + np.median(fragment_bond_distance_matrix) + np.average(fragment_bond_distance_matrix)
        relax_score = relax_score / (5 * np.max(fragment_bond_distance_matrix))
        return relax_score # notice that we return a score where the larger the score, the more seperated the two fragments of the TS are (ideal for initial guess)

# %%
def generate_bi_habs_ts_guess(
           rxn_smi,
           empirical_para_dict,
):
    """
    generate_bi_habs_ts_guess _summary_

    _extended_summary_

    Args:
        rxn_smi (_type_): _description_
        angle_X_H_Y (_type_): _description_
        dihedral_r1 (_type_): _description_
        dihedral_r2 (_type_): _description_
        bond_length_scale_factor_r1 (_type_): _description_
        bond_length_scale_factor_r2 (_type_): _description_
        bond_length_X_H (_type_, optional): _description_. Defaults to None.
        bond_length_H_Y (_type_, optional): _description_. Defaults to None.
    """

    # perceive reaction and generate reactants, products, and TS complexes
    # species_complex_dict = qp_utils.perceive_rxn_generate_complex(rxn_smi=rxn_smi)
    rxn = BimolecularHydrogenAbstractionReaction(rxn_smi)
    species_complex_dict = rxn.__dict__

    # initialize ts guess geometry
    species_complex_dict = initialize_ts_guess_geometry(
        empirical_para_dict = empirical_para_dict,
        species_complex_dict = species_complex_dict,)

    # optimize ts guess
    ts_new = opt_ts_ff(species_complex_dict=species_complex_dict)

    check_hard_collision(ts_new)

    bond_length_X_H = ts_new.GetConformer().GetBondLength(species_complex_dict['broken_bond'][0])
    bond_length_H_Y = ts_new.GetConformer().GetBondLength(species_complex_dict['formed_bond'][0])

    threshold = min([bond_length_X_H, bond_length_H_Y]) * 0.98
    relax_score = check_fragment_collision(ts_new, species_complex_dict['ts_h_index'][0], threshold=threshold)

    return relax_score, ts_new, species_complex_dict

# %%
# a helper function to attempt to generate N valid TS guesses, each with a score upto some max iteration
# expensive step, can be optimized
def gen_n_ts_confs(
           rxn_smi,
           empirical_para_dict,
           num_confs = 10,
           max_total_iter = 50,
):

    result = list()
    result_count = len(result)

    iter_counter = 0
    while result_count < num_confs and iter_counter < max_total_iter:
        try:
            relax_score, ts_new, species_complex_dict = generate_bi_habs_ts_guess(rxn_smi=rxn_smi, empirical_para_dict=empirical_para_dict)

            if all([relax_score, ts_new]):
                xyz = ts_new.ToXYZ()
                g_xyz = "\n".join([l for l in xyz.splitlines()[2:]]) + "\n\n"
                result.append(tuple([relax_score, g_xyz]))
        except:
            pass
        finally:
            iter_counter += 1
            result_count = len(result)

    if not result:
        raise ValueError('Failed to generate TS conformers.')
    else:
        result.sort(key=lambda y:y[0])

    output = dict()
    output = copy.deepcopy(species_complex_dict)
    del output['r_complex']
    del output['p_complex']
    del output['ts_complex']
    result_g_xyz = [x[-1] for x in result]
    score_dist = [x[0] for x in result]
    ts_conformers_coord = tuple([(k, v) for k, v in zip(score_dist, result_g_xyz)])
    output['ts_conformers_coord'] = ts_conformers_coord

    return output



# # this function calcualte R-H bond distances for closed-shell reactant/product species
# # based on averaging of N conformers embedded from force field
# # a helper function used in gen_ts
# def return_opt_spc_bond_distance(
#     spc_smi,
#     pivot_atom,
#     the_h_atom,
#     atom_map_idx,
#     averaged=5,
# ):
#     """
#     Return the optimized bond distance of reacting R-H from reactant/product species.
#     Only use on non-radical RH species.
#     """

#     bd_list = []

#     while averaged:
#         r = RDKitMol.FromSmiles(spc_smi)
#         ff = rdmc_ff.RDKitFF("mmff94s")
#         r.EmbedConformer(qp_utils.FF)
#         ff.setup(r)
#         ff.optimize()
#         m = ff.get_optimized_mol()

#         r_conf = r.GetConformer()
#         r_conf.SetPositions(m.GetPositions())

#         h_idx = atom_map_idx.index(the_h_atom)
#         pivot_idx = atom_map_idx.index(pivot_atom)

#         bd = r_conf.GetBondLength([h_idx, pivot_idx])
#         bd_list.append(bd)
#         averaged -= 1

#     return sum(bd_list) / len(bd_list)


class BimolecularHydrogenAbstractionReaction:
    """
    A bi-molecular hydrogen abstraction reaction.

    Parameters
    ----------
    rxn_smi : str
        Atom-mapped reaction smiles.

    Keyword Arguments
    ----------------
    **ts_guess_parameters : dict
        Keyword arguments for transition state guess parameters.


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

    def __init__(self, rxn_smi: str, **ts_guess_parameters):
        self._ts_guess_parameters = _DEFAULT_TS_GUESS_PARAMETERS.copy()
        self._ts_guess_parameters.update(ts_guess_parameters)

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

    # def initialize_ts_guess_geometry(self):
    #     species_complex_dict = self.__dict__
    #     empirical_para_dict = self._ts_guess_parameters

    #     _reactants, _products = qp_utils.split_rxn_smi(
    #         rxn_smi=species_complex_dict["rxn_smi"]
    #     )
    #     ts_conformer = species_complex_dict["ts_complex"].GetConformer()

    #     # set TS bond distances
    #     # update bond distance of r1 -- H(ts)
    #     # step 1: find the reacting bond distance of r1 when it's not in the TS based
    #     # on averaging of 10 conformers
    #     r1_dist = return_opt_spc_bond_distance(
    #         spc_smi=_reactants[0],
    #         pivot_atom=species_complex_dict["ts_pivot_indices"][0],
    #         the_h_atom=species_complex_dict["ts_h_index"][0],
    #         atom_map_idx=species_complex_dict["r_fragment_indices"][0],
    #     )

    #     # step 2: scale the reacting bond with a scaling factor, the factor is provided
    #     # by user currently, but it can be learned since it depends on reactant/product
    #     # type (usually 25% is good starting guess for C-H, 15% for O-H etc)
    #     bond_length_X_H = r1_dist * empirical_para_dict["bond_length_scale_factor_r1"]

    #     # step 3: set bond length for the TS complex
    #     ts_conformer.SetBondLength(
    #         [
    #             species_complex_dict["ts_pivot_indices"][0],
    #             species_complex_dict["ts_h_index"][0],
    #         ],
    #         bond_length_X_H,
    #     )

    #     # update bond distance of H(ts) -- r2
    #     r2_dist = return_opt_spc_bond_distance(
    #         spc_smi=_products[1],
    #         pivot_atom=species_complex_dict["ts_pivot_indices"][1],
    #         the_h_atom=species_complex_dict["ts_h_index"][0],
    #         atom_map_idx=species_complex_dict["p_fragment_indices"][1],
    #     )

    #     bond_length_H_Y = r2_dist * empirical_para_dict["bond_length_scale_factor_r2"]

    #     ts_conformer.SetBondLength(
    #         [
    #             species_complex_dict["ts_pivot_indices"][1],
    #             species_complex_dict["ts_h_index"][0],
    #         ],
    #         bond_length_H_Y,
    #     )

    #     # set TS angle
    #     # angle_X_H_Y can be learned, but usually 160 deg is good for most bi-mol
    #     # H abs family
    #     ts_conformer.SetAngleDeg(
    #         [
    #             species_complex_dict["ts_pivot_indices"][0],
    #             species_complex_dict["ts_h_index"][0],
    #             species_complex_dict["ts_pivot_indices"][1],
    #         ],
    #         empirical_para_dict["angle_X_H_Y"],
    #     )

    #     # set TS dihedral
    #     # note: there are two sets of dihedrals for the TS, since a dihedral is defined
    #     # by 4 connecting atoms, and there are 5 atoms connected in the TS region, thus
    #     # 2 sets of dihedrals

    #     # set dihedrals
    #     # note: the dihedrals can actually be searched or learned, but not implemented
    #     # yet, currently we set to user provided value
    #     ts_conformer.SetTorsionDeg(
    #         [
    #             species_complex_dict["r1_neighbour_indices"][0],
    #             species_complex_dict["ts_pivot_indices"][0],
    #             species_complex_dict["ts_h_index"][0],
    #             species_complex_dict["ts_pivot_indices"][1],
    #         ],
    #         empirical_para_dict["dihedral_r1"],
    #     )

    #     ts_conformer.SetTorsionDeg(
    #         [
    #             species_complex_dict["ts_pivot_indices"][0],
    #             species_complex_dict["ts_h_index"][0],
    #             species_complex_dict["ts_pivot_indices"][1],
    #             species_complex_dict["r2_neighbour_indices"][0],
    #         ],
    #         empirical_para_dict["dihedral_r2"],
    #     )

    #     return species_complex_dict
