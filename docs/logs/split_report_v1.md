# Split report, splits/v1 (FROZEN 2026-11-18, see SHA256SUMS)

Seed 20261118. Held out: `rf_harness_mks` -> test_heldout, `cs10k` -> test_external. Groups = pHash duplicate clusters (threshold 6). Only dedup keepers are listed.

Check: clusters spanning two splits = **0**. Leak-guard exclusions: 456.

## Images per split x source

| split | cs10k | hardhat_wkrs | rf_harness_fypcv | rf_harness_main | rf_harness_mks | rf_scaffold_archikhu | rf_scaffold_climb | rf_scaffold_esco | rf_scaffold_hook | rf_scaffold_mobile | rf_scaffold_platform | rf_scaffold_sparks | shwd | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 0 | 4726 | 474 | 2758 | 0 | 551 | 35 | 1010 | 9 | 348 | 125 | 70 | 3491 | **13597** |
| val | 0 | 1017 | 101 | 591 | 0 | 119 | 5 | 215 | 2 | 73 | 26 | 15 | 840 | **3004** |
| test_indist | 0 | 1015 | 101 | 609 | 0 | 117 | 5 | 215 | 1 | 74 | 26 | 20 | 857 | **3040** |
| test_heldout | 0 | 0 | 0 | 0 | 2692 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **2692** |
| test_external | 9394 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **9394** |

## Images per split x tag (selection hints from original classes, not labels)

| split | person | helmet | head | harness | scaffold | empty_scaffold |
|---|---|---|---|---|---|---|
| train | 12564 | 8490 | 3765 | 779 | 1942 | 546 |
| val | 2784 | 1819 | 908 | 176 | 414 | 117 |
| test_indist | 2811 | 1833 | 918 | 170 | 419 | 122 |
| test_heldout | 179 | 0 | 0 | 511 | 1701 | 1701 |
| test_external | 0 | 0 | 0 | 0 | 0 | 0 |

## Boxes per split x original class

| class | train | val | test_indist | test_heldout | test_external |
|---|---|---|---|---|---|
| cs10k:excavator | 0 | 0 | 0 | 0 | 4359 |
| cs10k:rebar | 0 | 0 | 0 | 0 | 2750 |
| cs10k:rule_1_violation | 0 | 0 | 0 | 0 | 1602 |
| cs10k:rule_2_violation | 0 | 0 | 0 | 0 | 141 |
| cs10k:rule_3_violation | 0 | 0 | 0 | 0 | 199 |
| cs10k:rule_4_violation | 0 | 0 | 0 | 0 | 67 |
| cs10k:worker_with_white_hard_hat | 0 | 0 | 0 | 0 | 1340 |
| hardhat_wkrs:head | 4545 | 972 | 1062 | 0 | 0 |
| hardhat_wkrs:helmet | 13213 | 2955 | 2843 | 0 | 0 |
| hardhat_wkrs:person | 460 | 57 | 92 | 0 | 0 |
| rf_harness_fypcv:human | 311 | 50 | 57 | 0 | 0 |
| rf_harness_fypcv:safety-harness | 456 | 99 | 102 | 0 | 0 |
| rf_harness_main:No harness | 6822 | 1385 | 1454 | 0 | 0 |
| rf_harness_main:No helmet | 1436 | 322 | 276 | 0 | 0 |
| rf_harness_main:No safety shoes | 870 | 246 | 172 | 0 | 0 |
| rf_harness_main:Safety shoes | 5736 | 1074 | 1298 | 0 | 0 |
| rf_harness_main:gloves | 1009 | 235 | 247 | 0 | 0 |
| rf_harness_main:harness | 586 | 161 | 130 | 0 | 0 |
| rf_harness_main:helmet | 6843 | 1374 | 1430 | 0 | 0 |
| rf_harness_main:ladder | 153 | 30 | 38 | 0 | 0 |
| rf_harness_main:lifeline | 310 | 65 | 68 | 0 | 0 |
| rf_harness_main:no gloves | 2914 | 630 | 667 | 0 | 0 |
| rf_harness_main:person | 8411 | 1752 | 1758 | 0 | 0 |
| rf_harness_main:table | 6 | 1 | 3 | 0 | 0 |
| rf_harness_mks:Guardrail | 0 | 0 | 0 | 2201 | 0 |
| rf_harness_mks:Suspension_scaffold | 0 | 0 | 0 | 2067 | 0 |
| rf_harness_mks:human | 0 | 0 | 0 | 320 | 0 |
| rf_harness_mks:safety-harness | 0 | 0 | 0 | 643 | 0 |
| rf_scaffold_archikhu:-scaffold | 82 | 17 | 20 | 0 | 0 |
| rf_scaffold_archikhu:mobile_scaffold_no_outtrigger | 130 | 28 | 26 | 0 | 0 |
| rf_scaffold_archikhu:mobile_scaffold_outtrigger | 215 | 45 | 48 | 0 | 0 |
| rf_scaffold_archikhu:vertical_ladder | 206 | 43 | 37 | 0 | 0 |
| rf_scaffold_archikhu:worker_with_helmet | 366 | 76 | 68 | 0 | 0 |
| rf_scaffold_archikhu:worker_with_helmet_climbing | 8 | 1 | 2 | 0 | 0 |
| rf_scaffold_archikhu:worker_without_helmet | 323 | 75 | 65 | 0 | 0 |
| rf_scaffold_archikhu:worker_without_helmet_climbing | 5 | 1 | 0 | 0 | 0 |
| rf_scaffold_climb:mobile_scaffold_no_outtrigger | 16 | 1 | 2 | 0 | 0 |
| rf_scaffold_climb:mobile_scaffold_outtrigger | 18 | 4 | 3 | 0 | 0 |
| rf_scaffold_climb:vertical_ladder | 10 | 3 | 1 | 0 | 0 |
| rf_scaffold_climb:worker_with_helmet | 18 | 1 | 2 | 0 | 0 |
| rf_scaffold_climb:worker_with_helmet_climbing | 1 | 0 | 0 | 0 | 0 |
| rf_scaffold_climb:worker_without_helmet | 11 | 1 | 5 | 0 | 0 |
| rf_scaffold_esco:scaffold | 1068 | 245 | 238 | 0 | 0 |
| rf_scaffold_esco:worker | 1266 | 271 | 270 | 0 | 0 |
| rf_scaffold_hook:box | 9 | 0 | 0 | 0 | 0 |
| rf_scaffold_hook:hardhat | 7 | 2 | 1 | 0 | 0 |
| rf_scaffold_hook:hook | 9 | 2 | 0 | 0 | 0 |
| rf_scaffold_hook:scaffolds | 9 | 2 | 1 | 0 | 0 |
| rf_scaffold_hook:worker | 9 | 2 | 1 | 0 | 0 |
| rf_scaffold_mobile:Missing Guardrail | 43 | 11 | 9 | 0 | 0 |
| rf_scaffold_mobile:mobile_scaffold_outtrigger | 332 | 71 | 72 | 0 | 0 |
| rf_scaffold_mobile:vertical_ladder | 192 | 42 | 39 | 0 | 0 |
| rf_scaffold_mobile:worker_with_helmet | 77 | 16 | 12 | 0 | 0 |
| rf_scaffold_mobile:worker_with_helmet_climbing | 143 | 31 | 35 | 0 | 0 |
| rf_scaffold_mobile:worker_without_helmet | 71 | 15 | 15 | 0 | 0 |
| rf_scaffold_mobile:worker_without_helmet_climbing | 86 | 18 | 19 | 0 | 0 |
| rf_scaffold_platform:base_support | 21 | 0 | 3 | 0 | 0 |
| rf_scaffold_platform:bracing | 294 | 39 | 55 | 0 | 0 |
| rf_scaffold_platform:guardrail | 77 | 24 | 16 | 0 | 0 |
| rf_scaffold_platform:modular_scaffold | 61 | 9 | 14 | 0 | 0 |
| rf_scaffold_platform:prefabricated_frame_scaffold | 57 | 13 | 15 | 0 | 0 |
| rf_scaffold_platform:toe_board | 85 | 18 | 16 | 0 | 0 |
| rf_scaffold_platform:tube_and_coupler_scaffold | 18 | 6 | 3 | 0 | 0 |
| rf_scaffold_platform:working_platform | 136 | 32 | 29 | 0 | 0 |
| rf_scaffold_sparks:scaffold | 94 | 18 | 21 | 0 | 0 |
| shwd:dog | 0 | 3 | 0 | 0 | 0 |
| shwd:hat | 5520 | 1234 | 1140 | 0 | 0 |
| shwd:person | 32638 | 9238 | 8294 | 0 | 0 |
