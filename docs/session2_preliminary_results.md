> Preliminary: 10 paired seeds per scenario, generated in Session 2. Not final results.

# Results summary

Mean [95% bootstrap CI] over seeds.

## baseline_150_no_fire

| Method | n | Last out (s) | Mean time (s) | Exposed agents | Max queue | Route changes |
| --- | --- | --- | --- | --- | --- | --- |
| nearest | 10 | 196.3 [193.3, 199.3] | 113.3 [112.5, 114.2] | 0.0 [0.0, 0.0] | 2.8 [2.4, 3.2] | 0.0 [0.0, 0.0] |
| reactive | 10 | 196.3 [193.3, 199.3] | 113.3 [112.5, 114.2] | 0.0 [0.0, 0.0] | 2.8 [2.4, 3.2] | 0.0 [0.0, 0.0] |
| safe_shortest | 10 | 196.3 [193.3, 199.3] | 113.3 [112.5, 114.2] | 0.0 [0.0, 0.0] | 2.8 [2.4, 3.2] | 0.0 [0.0, 0.0] |

Paired difference vs `safe_shortest` in last-out time (negative = faster):

| Method | Median diff (s) | 95% CI | Wilcoxon p |
| --- | --- | --- | --- |
| nearest | +0.0 | [+0.0, +0.0] | - |
| reactive | +0.0 | [+0.0, +0.0] | - |

## crowd_250_fire_j19

| Method | n | Last out (s) | Mean time (s) | Exposed agents | Max queue | Route changes |
| --- | --- | --- | --- | --- | --- | --- |
| nearest | 10 | 235.8 [227.8, 244.5] | 126.0 [124.2, 128.2] | 170.1 [166.2, 174.5] | 5.1 [4.0, 6.3] | 0.0 [0.0, 0.0] |
| reactive | 10 | 275.2 [256.9, 296.6] | 154.0 [148.6, 160.2] | 2.8 [2.0, 3.6] | 8.2 [6.6, 10.2] | 1.0 [0.0, 3.0] |
| safe_shortest | 10 | 271.1 [256.6, 286.2] | 154.0 [148.6, 160.3] | 2.8 [2.0, 3.6] | 8.3 [6.6, 10.4] | 0.0 [0.0, 0.0] |

Paired difference vs `safe_shortest` in last-out time (negative = faster):

| Method | Median diff (s) | 95% CI | Wilcoxon p |
| --- | --- | --- | --- |
| nearest | -33.8 | [-63.0, -11.0] | 0.00195 |
| reactive | +0.0 | [+0.0, +0.0] | 1 |

## fire_j19_150

| Method | n | Last out (s) | Mean time (s) | Exposed agents | Max queue | Route changes |
| --- | --- | --- | --- | --- | --- | --- |
| nearest | 10 | 196.3 [193.3, 199.3] | 114.2 [113.5, 115.0] | 102.1 [99.8, 104.4] | 2.7 [2.3, 3.1] | 0.0 [0.0, 0.0] |
| reactive | 10 | 210.2 [206.2, 214.8] | 127.2 [125.7, 129.5] | 1.7 [1.2, 2.2] | 3.1 [2.5, 3.7] | 0.0 [0.0, 0.0] |
| safe_shortest | 10 | 210.2 [206.2, 214.8] | 127.2 [125.7, 129.5] | 1.7 [1.2, 2.2] | 3.1 [2.5, 3.7] | 0.0 [0.0, 0.0] |

Paired difference vs `safe_shortest` in last-out time (negative = faster):

| Method | Median diff (s) | 95% CI | Wilcoxon p |
| --- | --- | --- | --- |
| nearest | -11.5 | [-20.5, -7.8] | 0.00195 |
| reactive | +0.0 | [+0.0, +0.0] | - |

## fire_j6_east_exit_150

| Method | n | Last out (s) | Mean time (s) | Exposed agents | Max queue | Route changes |
| --- | --- | --- | --- | --- | --- | --- |
| nearest | 10 | 207.4 [201.6, 213.4] | 124.8 [123.7, 125.8] | 11.5 [8.8, 13.9] | 2.9 [2.6, 3.2] | 0.0 [0.0, 0.0] |
| reactive | 10 | 216.3 [206.0, 226.2] | 125.5 [124.4, 126.6] | 1.0 [0.3, 1.7] | 3.1 [2.8, 3.4] | 0.0 [0.0, 0.0] |
| safe_shortest | 10 | 216.3 [206.0, 226.2] | 125.5 [124.4, 126.6] | 1.0 [0.3, 1.7] | 3.1 [2.8, 3.4] | 0.0 [0.0, 0.0] |

Paired difference vs `safe_shortest` in last-out time (negative = faster):

| Method | Median diff (s) | 95% CI | Wilcoxon p |
| --- | --- | --- | --- |
| nearest | -11.8 | [-15.0, +0.0] | 0.0312 |
| reactive | +0.0 | [+0.0, +0.0] | - |

## lecture_hall_f1_120

| Method | n | Last out (s) | Mean time (s) | Exposed agents | Max queue | Route changes |
| --- | --- | --- | --- | --- | --- | --- |
| nearest | 10 | 304.6 [301.2, 308.1] | 204.3 [201.8, 206.8] | 0.0 [0.0, 0.0] | 81.1 [81.0, 81.3] | 0.0 [0.0, 0.0] |
| reactive | 10 | 239.9 [236.9, 243.3] | 173.2 [171.3, 175.1] | 0.0 [0.0, 0.0] | 78.2 [78.0, 78.5] | 1003.4 [998.6, 1006.9] |
| safe_shortest | 10 | 304.6 [301.2, 308.1] | 204.3 [201.8, 206.8] | 0.0 [0.0, 0.0] | 81.1 [81.0, 81.3] | 0.0 [0.0, 0.0] |

Paired difference vs `safe_shortest` in last-out time (negative = faster):

| Method | Median diff (s) | 95% CI | Wilcoxon p |
| --- | --- | --- | --- |
| nearest | +0.0 | [+0.0, +0.0] | - |
| reactive | -65.0 | [-68.2, -62.5] | 0.00195 |
