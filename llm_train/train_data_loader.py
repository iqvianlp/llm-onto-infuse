import csv


def load_definition_spreadsheet(definition_csv_full_path, skip_first_row=True):
    """
    Given the path of a synthetic definitions CSV generated by means of the synth_data_gen.generate_synth_data script,
    read and load such data in memory
    :param definition_csv_full_path: full local path of the CSV with synthetic definitions generated by means of the
                                     synth_data_gen.generate_synth_data script
    :param skip_first_row: (default True) skip the first row when reading data from the CSV
    :return: two dictionaries:
        id_def: concept_ID --> synonym --> dict with the following keys:
        'label_type', 'main_name', 'synth_text', 'real_def', 'llm_id', 'prompt', 'error'
        label_concept_set: synonym --> set of associated concept IDs
    """
    id_def = dict()
    label_concept_set = dict()
    with open(definition_csv_full_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        first_row = True
        for row in reader:

            if skip_first_row is True and first_row is True:
                first_row = False
                continue

            concept_id = row[0]
            label = row[2]

            if len(concept_id.strip()) > 0 and len(label.strip()) > 0:

                # Populate concept definition
                if concept_id.strip() not in id_def:
                    id_def[concept_id.strip()] = dict()
                if label.strip() not in id_def[concept_id.strip()]:
                    id_def[concept_id.strip()][label.strip()] = dict()

                id_def[concept_id.strip()][label.strip()]['label_type'] = row[3]
                id_def[concept_id.strip()][label.strip()]['main_name'] = row[1]
                id_def[concept_id.strip()][label.strip()]['synth_text'] = row[4]
                id_def[concept_id.strip()][label.strip()]['real_def'] = row[5]
                id_def[concept_id.strip()][label.strip()]['llm_id'] = row[6]
                id_def[concept_id.strip()][label.strip()]['prompt'] = row[7]
                id_def[concept_id.strip()][label.strip()]['error'] = row[8]

                if label.strip() not in label_concept_set:
                    label_concept_set[label.strip()] = set()
                label_concept_set[label.strip()].add(concept_id)

    return id_def, label_concept_set
