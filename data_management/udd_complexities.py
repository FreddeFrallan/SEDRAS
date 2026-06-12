from data_management.underlying_data.output_properties import CategoricalOutputProperty

COMPLEXITY_LEVELS = {
    1: {
        "udd_num_output_labels": 2,
        "udd_num_rules": 5,

        "udd_num_variables": 4,
        "udd_min_categories": 3,
        "udd_max_categories": 4,

        "udd_num_numerical_variables": 0,
        "udd_numerical_min_spans": 4,
        "udd_numerical_max_spans": 4,

        'output_properties': [],
    },
    2: {
        "udd_num_output_labels": 3,
        "udd_num_rules": 6,

        "udd_num_variables": 5,
        "udd_min_categories": 3,
        "udd_max_categories": 4,

        "udd_num_numerical_variables": 1,
        "udd_numerical_min_spans": 4,
        "udd_numerical_max_spans": 4,

        'output_properties': [],
    },
    3: {
        "udd_num_output_labels": 4,
        "udd_num_rules": 8,

        "udd_num_variables": 6,
        "udd_min_categories": 3,
        "udd_max_categories": 5,

        "udd_num_numerical_variables": 1,
        "udd_numerical_min_spans": 4,
        "udd_numerical_max_spans": 4,

        'output_properties': [],
    },
    4: {
        "udd_num_output_labels": 4,
        "udd_num_rules": 8,

        "udd_num_variables": 6,
        "udd_min_categories": 3,
        "udd_max_categories": 5,

        "udd_num_numerical_variables": 1,
        "udd_numerical_min_spans": 4,
        "udd_numerical_max_spans": 4,

        'output_properties': [CategoricalOutputProperty('property_1', 3)],
    },
    5: {
        "udd_num_output_labels": 4,
        "udd_num_rules": 10,

        "udd_num_variables": 7,
        "udd_min_categories": 3,
        "udd_max_categories": 5,

        "udd_num_numerical_variables": 2,
        "udd_numerical_min_spans": 4,
        "udd_numerical_max_spans": 4,

        'output_properties': [CategoricalOutputProperty('property_1', 3),
                              CategoricalOutputProperty('property_2', 3)],
    },
}
