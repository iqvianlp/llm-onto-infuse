from collections.abc import Iterable


class OntoConcept:
    """
    Concept from ontology
    """

    def __init__(self, identifier, main_label):
        """
        Constructor of ontology concepts
        :param identifier: identifier of the ontology concept (non-empty string)
        :param main_label: main label of the ontoklogy concept (non-empty string)
        """
        if identifier is None or not isinstance(identifier, str) or len(identifier.strip()) == 0:
            raise Exception("Attention, you're creating an ontology concept by providing a None or empty-string "
                            "identifier.")
        if identifier is None or not isinstance(identifier, str) or len(identifier.strip()) == 0:
            print(f"Attention, you're creating an ontology concept by providing a None or empty-string main label "
                  f"for concept with id {identifier}.")

        # Concept identifier
        self.id = identifier

        # Main label (i.e. preferred term) of the concept
        self.main_label = main_label

        # Definition of the concept
        self.definition = None

        # Dictionary of concept synonyms where: the key is the synonym and value is a dictionary of metadate (key-
        # value pairs) associated to the considered concept synonym (metadata could include the type or the source
        # of the synonym)
        self.synonyms = dict()

        # Dictionary describing the relations of the concept with other concepts: the key is the identifier of the
        # relation while the value is in turns a dictionary. For each considered relation, such dictionary will include:
        # as keys the concept identifiers that are relation targets and as values a dictionary of metadate (key-value
        # pairs) associated to the considered relation-target-concept-id pair (metadata could include the source of the
        # relation for instance)
        self.rel_dict = dict()

    def add_synonym(self, synonym, metadata_dict=None):
        """
        Add a synonym of the concept. Existing concept synonyms and metadata associated to that concept synonym are
        overwritten.
        :param synonym: non-empty string representing the synonym to add (case-sensitive)
        :param metadata_dict: metadata associated to the synonym
        :return: boolean, True of the synonym has been correctly added to the concept
        """
        added_synonym = False
        if isinstance(synonym, str) and len(synonym.strip()) > 0:
            added_synonym = True
            synonym = synonym.strip()
            if synonym not in self.synonyms:
                self.synonyms[synonym] = dict()
            if isinstance(metadata_dict, dict):
                for k, v in metadata_dict.items():
                    if isinstance(k, str) and len(k.strip()) > 0:
                        self.synonyms[synonym][k] = v

        return added_synonym

    def del_synonyms(self, synonyms):
        """
        Delerte
        :param synonyms: list of non-empty strings representing the synonyms to delete (case-sensitive)
        :return: number of synonyms deleted form that concept
        """
        deleted_syns_counter = 0
        if isinstance(synonyms, Iterable) and len(synonyms) > 0:
            for syn in [s.strip() for s in synonyms if isinstance(s, str) and len(s.strip()) > 0]:
                if syn in self.synonyms:
                    self.synonyms.remove(syn)
                    deleted_syns_counter = deleted_syns_counter + 1

        return deleted_syns_counter

    def add_definition(self, definition):
        """
        Add definition of the concept
        :param definition: non-empty string representing the definition of the concept
        :return: boolean, True if the definition has been correctly associated to the concept
        """
        added_definition = False
        if isinstance(definition, str) and len(definition.strip()) > 0:
            added_definition = True
            self.definition = definition.strip()

        return added_definition

    def add_relation(self, rel_type, target_id, metadata_dict=None):
        """
        Add a relation-target-concept-id pair to concept relations
        :param rel_type: identifier of the relation type (non-empty string)
        :param target_id: identifier of the concept that represents the target of the relation (non-empty string)
        :param metadata_dict: dictionary of metadate (key-value pairs) associated to the considered
        relation-target-concept-id pair (metadata could include the source of the relation for instance)
        :return: boolean, True if the relation-target-concept-id pair has been correctly added to the concept
        """
        added_relation = False
        if isinstance(rel_type, str) and len(rel_type.strip()) > 0 and isinstance(target_id, str) and len(target_id.strip()) > 0:
            added_relation = True
            rel_type = rel_type.strip()
            target_id = target_id.strip()
            if rel_type not in self.rel_dict:
                self.rel_dict[rel_type] = dict()
            if target_id not in self.rel_dict[rel_type]:
                self.rel_dict[rel_type][target_id] = dict()
            if isinstance(metadata_dict, dict):
                for k, v in metadata_dict.items():
                    if isinstance(k, str) and len(k.strip()) > 0:
                        self.rel_dict[rel_type][target_id][k] = v

        return added_relation

    def del_relation(self, rel_type, target_id=None):
        """
        Remove a relation-target-concept-id pair from concept relations
        :param rel_type: identifier of the relation type (non-empty string)
        :param target_id: identifier of the concept that represents the target of the relation (non-empty string)
        :return: boolean, True if the relation-target-concept-id pair has been correctly added to the concept
        """
        deleted_relation = False
        if isinstance(rel_type, str) and len(rel_type.strip()) > 0:
            if rel_type in self.rel_dict:
                if target_id is None:
                    del self.rel_dict[rel_type]
                    deleted_relation = True
                if isinstance(target_id, str) and len(target_id.strip()) > 0:
                    del self.rel_dict[rel_type][target_id]
                    deleted_relation = True

        return deleted_relation
