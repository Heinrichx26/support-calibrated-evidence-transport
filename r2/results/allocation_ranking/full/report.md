# Frozen allocation and ranking decomposition

No models, transforms, thresholds, or per-query counts were fitted or selected in this analysis. The four cells cross the saved sentence ranks and saved per-query admitted counts. The cross-encoder uses raw scores and every rank uses the original fixed tie keys.

| Dataset | Seed | Rank | Count allocation | Queries | Sentences | Correct | Complete sets | Precision | Set recall |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| fever | 13 | ce | ce | 13999 | 13999 | 7553 | 6076 | 0.53954 | 0.65319 |
| fever | 13 | scet | ce | 13999 | 13999 | 7324 | 5933 | 0.52318 | 0.63782 |
| fever | 13 | ce | scet | 13999 | 13999 | 8175 | 6461 | 0.58397 | 0.69458 |
| fever | 13 | scet | scet | 13999 | 13999 | 8155 | 6437 | 0.58254 | 0.69200 |
| fever | 17 | ce | ce | 13999 | 13999 | 7539 | 6051 | 0.53854 | 0.65156 |
| fever | 17 | scet | ce | 13999 | 13999 | 7243 | 5874 | 0.51739 | 0.63250 |
| fever | 17 | ce | scet | 13999 | 13999 | 8153 | 6385 | 0.58240 | 0.68752 |
| fever | 17 | scet | scet | 13999 | 13999 | 8106 | 6366 | 0.57904 | 0.68547 |
| fever | 29 | ce | ce | 13999 | 13999 | 7587 | 6113 | 0.54197 | 0.65018 |
| fever | 29 | scet | ce | 13999 | 13999 | 7345 | 5977 | 0.52468 | 0.63572 |
| fever | 29 | ce | scet | 13999 | 13999 | 8262 | 6502 | 0.59019 | 0.69155 |
| fever | 29 | scet | scet | 13999 | 13999 | 8242 | 6500 | 0.58876 | 0.69134 |
| fever | 7 | ce | ce | 13999 | 13999 | 7583 | 6094 | 0.54168 | 0.65492 |
| fever | 7 | scet | ce | 13999 | 13999 | 7358 | 5958 | 0.52561 | 0.64030 |
| fever | 7 | ce | scet | 13999 | 13999 | 8211 | 6442 | 0.58654 | 0.69232 |
| fever | 7 | scet | scet | 13999 | 13999 | 8218 | 6452 | 0.58704 | 0.69339 |
| hover | 13 | ce | ce | 2800 | 2800 | 1876 | 124 | 0.67000 | 0.04429 |
| hover | 13 | scet | ce | 2800 | 2800 | 1897 | 147 | 0.67750 | 0.05250 |
| hover | 13 | ce | scet | 2800 | 2800 | 1807 | 159 | 0.64536 | 0.05679 |
| hover | 13 | scet | scet | 2800 | 2800 | 1893 | 195 | 0.67607 | 0.06964 |
| hover | 17 | ce | ce | 2800 | 2800 | 1874 | 125 | 0.66929 | 0.04464 |
| hover | 17 | scet | ce | 2800 | 2800 | 1874 | 125 | 0.66929 | 0.04464 |
| hover | 17 | ce | scet | 2800 | 2800 | 1874 | 125 | 0.66929 | 0.04464 |
| hover | 17 | scet | scet | 2800 | 2800 | 1874 | 125 | 0.66929 | 0.04464 |
| hover | 29 | ce | ce | 2800 | 2800 | 1910 | 113 | 0.68214 | 0.04036 |
| hover | 29 | scet | ce | 2800 | 2800 | 1915 | 139 | 0.68393 | 0.04964 |
| hover | 29 | ce | scet | 2800 | 2800 | 1856 | 138 | 0.66286 | 0.04929 |
| hover | 29 | scet | scet | 2800 | 2800 | 1927 | 196 | 0.68821 | 0.07000 |
| hover | 7 | ce | ce | 2800 | 2800 | 1866 | 124 | 0.66643 | 0.04429 |
| hover | 7 | scet | ce | 2800 | 2800 | 1882 | 142 | 0.67214 | 0.05071 |
| hover | 7 | ce | scet | 2800 | 2800 | 1840 | 143 | 0.65714 | 0.05107 |
| hover | 7 | scet | scet | 2800 | 2800 | 1915 | 189 | 0.68393 | 0.06750 |

All original diagonals reproduce admitted counts, correct counts, complete-set outcomes, coverage, and original evidence counts exactly, for every evaluated query. Sorting and label invariance are checked in the smoke. Every full-run cell retains the original query denominators.

| Dataset | Metric | Group | Net SCET minus cross-encoder | Ranking contribution | Count-allocation contribution |
|---|---|---|---:|---:|---:|
| fever | precision | all | 0.04391 [0.03633, 0.05156] | -0.00957 [-0.01175, -0.00735] | 0.05349 [0.04637, 0.06064] |
| fever | evidence_recall | all | 0.02543 [0.01775, 0.03327] | -0.01016 [-0.01229, -0.00805] | 0.03558 [0.02812, 0.04286] |
| fever | set_recall | all | 0.03809 [0.02946, 0.04636] | -0.00841 [-0.01061, -0.00622] | 0.04650 [0.03839, 0.05432] |
| fever | complete_per100 | all | 2.53768 [1.96560, 3.08574] | -0.55986 [-0.70631, -0.41707] | 3.09754 [2.55750, 3.62518] |
| fever | coverage | all | -0.01823 [-0.02540, -0.01109] | 0.00000 [0.00000, 0.00000] | -0.01823 [-0.02540, -0.01109] |
| fever | no_evidence_admission | all | -0.12433 [-0.13936, -0.11084] | 0.00000 [0.00000, 0.00000] | -0.12433 [-0.13936, -0.11084] |
| fever | no_evidence_items_per100 | all | -5.15930 [-5.98417, -4.31665] | 0.00000 [0.00000, 0.00000] | -5.15930 [-5.98417, -4.31665] |
| fever | precision | singleton | 0.00936 [0.00049, 0.01809] | -0.01002 [-0.01262, -0.00752] | 0.01939 [0.01118, 0.02763] |
| fever | precision | multiple | 0.02215 [-0.00918, 0.05441] | -0.03765 [-0.05345, -0.02179] | 0.05980 [0.03177, 0.08629] |
| fever | set_recall | singleton | 0.04477 [0.03549, 0.05358] | -0.00863 [-0.01107, -0.00634] | 0.05340 [0.04465, 0.06182] |
| fever | set_recall | multiple | -0.02856 [-0.04290, -0.01540] | -0.00617 [-0.01252, -0.00012] | -0.02240 [-0.03461, -0.01086] |
| hover | precision | all | 0.00741 [-0.00676, 0.01966] | 0.01223 [0.00555, 0.01904] | -0.00482 [-0.01627, 0.00472] |
| hover | evidence_recall | all | 0.00723 [0.00156, 0.01272] | 0.00529 [0.00276, 0.00791] | 0.00194 [-0.00312, 0.00669] |
| hover | set_recall | all | 0.01955 [0.01405, 0.02497] | 0.00924 [0.00556, 0.01317] | 0.01031 [0.00655, 0.01425] |
| hover | complete_per100 | all | 1.95536 [1.40513, 2.49657] | 0.92411 [0.55622, 1.31665] | 1.03125 [0.65544, 1.42527] |
| hover | coverage | all | -0.02946 [-0.04194, -0.01737] | 0.00000 [0.00000, 0.00000] | -0.02946 [-0.04194, -0.01737] |
| hover | no_evidence_items_per100 | all | 0.00000 [0.00000, 0.00000] | 0.00000 [0.00000, 0.00000] | 0.00000 [0.00000, 0.00000] |
| hover | precision | multiple | 0.00741 [-0.00676, 0.01966] | 0.01223 [0.00555, 0.01904] | -0.00482 [-0.01627, 0.00472] |
| hover | set_recall | multiple | 0.01955 [0.01405, 0.02497] | 0.00924 [0.00556, 0.01317] | 0.01031 [0.00655, 0.01425] |

Ranking contribution averages the ranking change under each fixed count allocation. Count-allocation contribution averages the count change under each fixed ranking. They sum exactly to the original net difference. Conditional effects and their interaction are retained in the decomposition CSV files. Negative contributions and intervals spanning zero are reported without changing the scorer or analysis specification.

Intervals use 1,000 paired query-cluster resamples with a common weight for each query wherever it occurs across the four overlapping partitions. Decisions remain frozen inside resamples. Intervals are pointwise and conditional on the observed partitions. The diagnostic hybrid cells do not define newly selected deployment methods.

Coverage, admission on queries without annotated evidence, and the number of items assigned to those queries depend entirely on the count allocation. HoVer contains no query without annotated evidence, so its corresponding admission proportion is unavailable.
