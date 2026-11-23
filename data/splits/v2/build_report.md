# splits/v2 build report

Seed 20261120. Built by `GenIA-code/02_labeling/relation_v2/build_splits_v2.py`.

| split_v2 | task_kind | source | synthetic | images | people (prefill, new only) |
|---|---|---|---|---|---|
| cs10k_dev | new | cs10k | 0 | 150 |  |
| test_v1 | frozen_test | cs10k | 0 | 159 |  |
| test_v1 | frozen_test | rf_harness_main | 0 | 36 |  |
| test_v1 | frozen_test | rf_harness_mks | 0 | 34 |  |
| test_v1 | frozen_test | rf_scaffold_archikhu | 0 | 18 |  |
| test_v1 | frozen_test | rf_scaffold_climb | 0 | 2 |  |
| test_v1 | frozen_test | rf_scaffold_esco | 0 | 22 |  |
| test_v1 | frozen_test | rf_scaffold_hook | 0 | 1 |  |
| test_v1 | frozen_test | rf_scaffold_mobile | 0 | 64 |  |
| train_v2 | new | rf_harness_main | 0 | 300 | 997 |
| train_v2 | new | rf_scaffold_archikhu | 0 | 261 | 734 |
| train_v2 | new | rf_scaffold_climb | 0 | 17 | 29 |
| train_v2 | new | rf_scaffold_platform | 0 | 20 | 35 |
| train_v2 | new | rf_scaffold_sparks | 0 | 52 | 119 |
| train_v2 | relabel_v1 | rf_scaffold_archikhu | 0 | 11 |  |
| train_v2 | relabel_v1 | rf_scaffold_climb | 0 | 2 |  |
| train_v2 | relabel_v1 | rf_scaffold_esco | 0 | 11 |  |
| train_v2 | relabel_v1 | rf_scaffold_esco | 1 | 113 |  |
| train_v2 | relabel_v1 | rf_scaffold_hook | 0 | 1 |  |
| train_v2 | relabel_v1 | rf_scaffold_mobile | 0 | 30 |  |
| val_v2 | new | rf_harness_main | 0 | 100 | 355 |
| val_v2 | new | rf_scaffold_archikhu | 0 | 56 | 158 |
| val_v2 | new | rf_scaffold_climb | 0 | 2 | 3 |
| val_v2 | new | rf_scaffold_platform | 0 | 4 | 5 |
| val_v2 | new | rf_scaffold_sparks | 0 | 9 | 22 |
| val_v2 | relabel_v1 | rf_scaffold_archikhu | 0 | 3 |  |
| val_v2 | relabel_v1 | rf_scaffold_esco | 0 | 12 |  |
| val_v2 | relabel_v1 | rf_scaffold_esco | 1 | 35 |  |

| split_v2 | images |
|---|---|
| cs10k_dev | 150 |
| test_v1 | 336 |
| train_v2 | 818 |
| val_v2 | 221 |

test_v1 by subset: {'test_external': 159, 'test_indist': 143, 'test_heldout': 34} (expected 143 / 159 / 34 = 336)
CS10k test categories: {'at_height_no_scaffold': 66, 'scaffold_workers': 60, 'lookalike': 30, 'cap_hood': 3}
cs10k_dev quotas: {'at_height_no_scaffold': 62, 'scaffold_workers': 57, 'lookalike': 28, 'cap_hood': 3}
cs10k_dev candidates per category: {'at_height_no_scaffold': 2285, 'lookalike': 1832, 'scaffold_workers': 1021, 'cap_hood': 15}
cs10k_dev exclusions: {'hamming<=10_to_test': 5}
