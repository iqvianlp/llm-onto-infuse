import os.path
import pickle
import re
from collections import Counter

from pronto import Ontology
from tqdm import tqdm

import constant_config
from onto.model import OntoConcept


def load_mondo_onto_dictionary(pickle_file_path=constant_config.MONDO_ONTO_PICKLE_FILE_PATH):
    """
    Load MONDO ontology from the OBO library into an onto-dictionary where each entry describes an ontology concept: the key
    is the unique identifier of the ontology concept, while the value is the onto.model.OntoConcept class instance
    gathering the information that describes that ontology concept.
    If a MONDO ontology pickle object file is already available at pickle_file_path, the onto-dictionary will be loaded
    from that pickle_file_path (quicker), otherwise it will be loaded by relying on the pronto python package and then
    serialized/stored as MONDO ontology pickle object file at pickle_file_path.
    :param pickle_file_path: full local path of the MONDO ontology pickle object file (exploited to store the contents
    of the ontology to support ontological knowledge infusion)
    :return: ontology concept dictionary
    """
    if os.path.isfile(pickle_file_path):
        print(f"Loading MONDO ontology onto_dictionary from pickle file {pickle_file_path}...")
        file = open(pickle_file_path, 'rb')
        onto_dictionary = pickle.load(file)
        file.close()
    else:
        print(f"Loading MONDO ontology, then storing onto_dictionary to pickle file {pickle_file_path}...")
        mondo_obo = Ontology.from_obo_library("mondo.owl")

        print(f'Ontology MONDO loaded: {mondo_obo}, post processing contents...')

        onto_dictionary = dict()
        for term in mondo_obo.terms():
            if term.obsolete:
                continue

            # Add term and main label
            if term.id in onto_dictionary:
                print(f"Attention, not adding concept with id {term.id} since it has already been added.")
            else:
                onto_dictionary[term.id] = OntoConcept(term.id, term.name)

            # Add definition
            if isinstance(term.definition, str) and len(term.definition.strip()) > 0:
                onto_dictionary[term.id].add_definition(term.definition.strip())

            # Add synonyms
            for syn in term.synonyms:
                syn_dict = dict()
                syn_dict['scope'] = syn.scope
                syn_dict['type'] = syn.type
                syn_dict['rel_concepts'] = [(xr.id, xr.description) for xr in syn.xrefs]
                onto_dictionary[term.id].add_synonym(syn.description, syn_dict)

            # Add hyponym relations
            subc_set = term.subclasses(distance=1).to_set()
            if len(subc_set) > 0:
                for subc in subc_set:
                    if term.id != subc.id and not subc.obsolete:
                        # print(f'TERM {term.id} ({term.name}) HAS AS SUB-CLASS THE TERM: {subc.id} ({subc.name})')
                        rel_dict = dict()
                        rel_dict['rel_name'] = 'hyponym'
                        rel_dict['target_name'] = subc.name
                        onto_dictionary[term.id].add_relation('hyponym', subc.id, rel_dict)

            # Add non-hyponym relations
            rel_keys = term.relationships.keys()
            if len(rel_keys) > 0:
                for rel_type in rel_keys:
                    rel_targets = term.relationships[mondo_obo.get_relationship(rel_type.id)]
                    for rel_target in [rel_t for rel_t in rel_targets if not rel_t.obsolete]:
                        # print(f"TERM {term.id} ({term.name}) HAS {rel_type.id} ({rel_type.name}) RELATIONSHIP WITH TERM {rel_target.id} ({rel_target.name})")
                        rel_dict = dict()
                        rel_dict['rel_id'] = rel_type.id
                        rel_dict['rel_name'] = rel_type.name
                        rel_dict['target_name'] = rel_target.name
                        onto_dictionary[term.id].add_relation(rel_type.name, rel_target.id, rel_dict)

        # Store onto-
        file = open(pickle_file_path, 'wb')
        pickle.dump(onto_dictionary, file)
        file.close()

    print(f"\n--- LOADED MONDO ONTOLOGY DICTIONARY ---")

    return onto_dictionary


def get_parent_ids(onto_d, concept_id):
    """
    Get the list of parent concept IDs of a specific concept
    :param onto_d: the MONDO onto-dictionary of reference
    :param concept_id: the id of the concept to retrieve parents of
    :return: list of parent concept IDs
    """
    parents_cid = [cid for cid, c_obj in onto_d.items() for r_name, r_targets_dict in c_obj.rel_dict.items() if cid.startswith('MONDO:') and r_name == 'hyponym' and concept_id in list(r_targets_dict.keys())]
    return parents_cid


def get_child_ids(onto_d, concept_id):
    """
    Get the list of child concept IDs of a specific concept
    :param onto_d: the MONDO onto-dictionary of reference
    :param concept_id: the id of the concept to retrieve child(ren) of
    :return: list of child(ren) concept IDs
    """
    children_cid = [cid_target for r_name, r_targets_dict in onto_d[concept_id].rel_dict.items() for cid_target in list(r_targets_dict.keys()) if r_name == 'hyponym' and cid_target.startswith('MONDO:')]
    return children_cid


def get_ancestors_of(onto_d, concept_id):
    """
    Get the list of ancestor concept IDs of a specific concept
    :param onto_d: the MONDO onto-dictionary of reference
    :param concept_id: the id of the concept to retrieve ancestors of
    :return: list of ancestor concept IDs
    """
    ancestors_cids = set()
    visited_cids = set()
    ancestors_cids.add(concept_id)
    while True:
        initial_count_ancestors_cids = len(ancestors_cids)
        new_ancestors_cids = set()
        for ancestors_cid in ancestors_cids:
            if ancestors_cid not in visited_cids:
                parent_cids = get_parent_ids(onto_d, ancestors_cid)
                visited_cids.add(ancestors_cid)
                for cid in parent_cids:
                    new_ancestors_cids.add(cid)
        ancestors_cids.update(new_ancestors_cids)
        if len(ancestors_cids) == initial_count_ancestors_cids:
            break

    if concept_id in ancestors_cids:
        ancestors_cids.remove(concept_id)

    return ancestors_cids


def get_descendants_of(onto_d, concept_id):
    """
    Get the list of descendent concept IDs of a specific concept
    :param onto_d: the MONDO onto-dictionary of reference
    :param concept_id: the id of the concept to retrieve descendents of
    :return: list of descendent concept IDs
    """
    descendants_cids = set()
    visited_cids = set()
    descendants_cids.add(concept_id)
    while True:
        initial_count_descendants_cids = len(descendants_cids)
        new_descendants_cids = set()
        for descendants_cid in descendants_cids:
            if descendants_cid not in visited_cids:
                children_cids = get_child_ids(onto_d, descendants_cid)
                visited_cids.add(descendants_cid)
                for cid in children_cids:
                    new_descendants_cids.add(cid)
        descendants_cids.update(new_descendants_cids)
        if len(descendants_cids) == initial_count_descendants_cids:
            break

    if concept_id in descendants_cids:
        descendants_cids.remove(concept_id)

    return descendants_cids


def get_siblings_of(onto_d, concept_id):
    """
    Get the list of sibling concept IDs of a specific concept
    :param onto_d: the MONDO onto-dictionary of reference
    :param concept_id: the id of the concept to retrieve siblings of
    :return: list of sibling concept IDs
    """
    siblings_cids = set()
    parent_cids = get_parent_ids(onto_d, concept_id)
    for parent_cid in parent_cids:
        child_ids = get_child_ids(onto_d, parent_cid)
        siblings_cids.update(child_ids)

    if concept_id in siblings_cids:
        siblings_cids.remove(concept_id)

    return siblings_cids


def generate_mondo_concept_synonym_set(onto_d):
    """
    Retrieve the set of synonyms associated to each MONDO concept. Include synonym expansion heuristics.
    :param onto_d: the MONDO onto-dictionary of reference
    :return: dictionary where each entry has a MONDO concept IDs as key and the set of associated synonyms as value.
    Exclusively EXACT synonyms are considered.
    """
    concept_synonym_set = dict()

    for cid, onto_concept in tqdm(onto_d.items(), desc=f"Retrieving / generating concept synonyms from ontology contents..."):

        # Skip concepts that are not directly defined by MONDO ontology
        if not cid.startswith("MONDO:"):
            continue

        # Retrieve all concept synonyms
        local_synonyms_set = set()

        if isinstance(onto_d[cid].synonyms, dict):
            local_synonyms_set = set([syn for syn, syn_dic in onto_d[cid].synonyms.items() if 'scope' in syn_dic and syn_dic['scope'] == 'EXACT'])
            local_synonyms_set = set([el.strip() for el in local_synonyms_set if isinstance(el, str) and len(el.strip()) > 0])

        # Add main name (i.e. preferred term)
        local_synonyms_set.add(onto_concept.main_label)

        # Generate new synonyms from synonyms including commas
        new_comma_reordered_synonyms = set()
        for concept_synonym in local_synonyms_set:
            if ', ' in concept_synonym:
                concept_synonym_split = [el.strip() for el in concept_synonym.split(', ') if len(el.strip()) > 0]
                if len(concept_synonym_split) > 0:

                    # Adding new synonym: normal order
                    new_syn = " ".join([concept_synonym_split[idx] for idx in range(len(concept_synonym_split))])
                    new_comma_reordered_synonyms.add(new_syn)
                    # print(f"  FROM '{concept_synonym}' --> NEW SYNONYM: {new_syn}")

                    # Adding new synonym: reverse order
                    new_syn = " ".join([concept_synonym_split[idx] for idx in reversed(range(len(concept_synonym_split)))])
                    new_comma_reordered_synonyms.add(new_syn)
                    # print(f"  FROM '{concept_synonym}' --> NEW SYNONYM: {new_syn}")

                    # Adding new synonym: last split at beginning
                    if len(concept_synonym_split) == 3:
                        new_syn = " ".join([concept_synonym_split[1], concept_synonym_split[2], concept_synonym_split[0]])
                        new_comma_reordered_synonyms.add(new_syn)
                        # print(f"  FROM '{concept_synonym}' --> NEW SYNONYM: {new_syn}")

        if len(new_comma_reordered_synonyms) > 0:
            local_synonyms_set.update(new_comma_reordered_synonyms)

        # Generate new synonyms from synonyms including round brackets
        new_parenthesis_extended_synonyms = set()
        for concept_synonym in local_synonyms_set:
            if '(' in concept_synonym and ')' in concept_synonym:
                removed_txt_between_parenthesis = re.sub("\(.*?\)", "", concept_synonym).strip()
                if concept_synonym.strip()[-1] == ')' and len(removed_txt_between_parenthesis) > 0:
                    new_parenthesis_extended_synonyms.add(removed_txt_between_parenthesis)
                    # print(f"  FROM '{concept_synonym}' --> NEW SYNONYM: {removed_txt_between_parenthesis}")

                removed_parenthesis = concept_synonym.replace("(", " ").replace(")", " ").strip()
                if concept_synonym.strip()[0] == '(' and len(removed_parenthesis) > 0:
                    new_parenthesis_extended_synonyms.add(removed_parenthesis)
                    # print(f"  FROM '{concept_synonym}' --> NEW SYNONYM: {removed_parenthesis}")

        if len(new_parenthesis_extended_synonyms) > 0:
            local_synonyms_set.update(new_parenthesis_extended_synonyms)

        concept_synonym_set[cid] = local_synonyms_set

    print(f"----------------------------------------")
    print(f" Distribution of concepts by number of synonyms: {Counter([len(syn_set) for cid, syn_set in concept_synonym_set.items()])}.")
    print(f"----------------------------------------")

    return concept_synonym_set


if __name__ == '__main__':

    mondo_onto_dict = load_mondo_onto_dictionary()
    concept_syn_set = generate_mondo_concept_synonym_set(mondo_onto_dict)

    # Print for each MONDO-ontology concept the list of siblings, parents, ancestors, children and descendents
    # concept IDs
    printed_concepts_counter = 0
    for cid, cid_obj in mondo_onto_dict.items():

        printed_concepts_counter = printed_concepts_counter + 1
        if printed_concepts_counter > 100:
            break

        # Skip concepts that are not directly defined by MONDO ontology
        if not cid.startswith("MONDO:"):
            continue

        sibling_ids = get_siblings_of(mondo_onto_dict, cid)
        parents_ids = get_parent_ids(mondo_onto_dict, cid)
        ancestors_ids = get_ancestors_of(mondo_onto_dict, cid)
        children_ids = get_child_ids(mondo_onto_dict, cid)
        descendants_ids = get_descendants_of(mondo_onto_dict, cid)

        print(f"------------------------------")
        print(f"CONCEPT: {cid_obj.main_label} ({cid})")
        print(f"   --> SIBLINGS: {[f'{mondo_onto_dict[el].main_label} ({el}) - ' for el in sibling_ids]}")
        print(f"   --> PARENTS: {[f'{mondo_onto_dict[el].main_label} ({el}) - ' for el in parents_ids]}")
        print(f"   --> ANCESTORS: {[f'{mondo_onto_dict[el].main_label} ({el}) - ' for el in ancestors_ids]}")
        print(f"   --> CHILDREN: {[f'{mondo_onto_dict[el].main_label} ({el}) - ' for el in children_ids]}")
        print(f"   --> DESCENDANTS: {[f'{mondo_onto_dict[el].main_label} ({el}) - ' for el in descendants_ids]}")
