# Dicke-state submission head-to-head

**Simulator proxies only; these are not hardware results.** Higher scores are better.

Snapshot: 2026-10-25T08:01:40+00:00. Eligible circuits with both models scored: **196/288**; recorded circuits: 288/288; circuits with reported failures: 1.

Candidates:

- **A** — `team_eval/will_dicke_submission.py (reference)`
- **B** — `submission.py`
- **C** — `columnforge/noise_aware/rewrites/submission_scheduled.py`
- **D** — `astra/submission.py`

Each calibration seed uses the same dictionaries for every candidate. Requested settings are K=N; 10,000 shots per basis; at least 2,000 Pauli trajectories and 1,000 HW trajectories; at least 400 shot resamplings. Actual settings and score-formula spot-check results are recorded below when supplied by the runner.

Eligibility requires recorded noiseless exactness and validity passes plus construction time ≤60 seconds. A missing check is pending. Non-finite medians are not ranked. All score sums below sum instance medians, not pooled shots or per-resampling sums. Raw means, fidelity, build time, validation detail, and circuit paths are in `results.json`.

**Partial-data rules:** † marks a provisional winner with fewer than four eligible scores. × marks an unranked score from an ineligible circuit. ⚠ points to the failures table. Bold scores are per-instance maxima among eligible available results; ties are retained. Small-state scores can be heavy-tailed, so reported means and medians need not agree.

## Score sums and coverage

Available sums can have different coverage and must not be compared directly. Common sums use only instances with an eligible score for **all four** candidates under that seed/model. A complete all-instance sum requires 36/36.

| Seed | Candidate | Pauli available sum | Pauli coverage | Pauli common sum | Common coverage | HW available sum | HW coverage | HW common sum | Common coverage |
|---:|:---|---:|:---:|---:|:---:|---:|:---:|---:|:---:|
| 101 | A | 35.32527 | 30/36 | 22.66784 | 24/36 | 32.04491 | 30/36 | 20.10692 | 23/36 |
| 101 | B | 22.45363 | 24/36 | 22.45363 | 24/36 | 20.42544 | 24/36 | 20.00887 | 23/36 |
| 101 | C | 27.21350 | 25/36 | 22.34687 | 24/36 | 23.22626 | 24/36 | 19.13616 | 23/36 |
| 101 | D | 30.98643 | 27/36 | 22.53180 | 24/36 | 27.71296 | 27/36 | 20.01546 | 23/36 |
| 202 | A | 21.88765 | 23/36 | 20.97082 | 22/36 | 19.91714 | 23/36 | 19.07758 | 22/36 |
| 202 | B | 21.84644 | 23/36 | 20.94043 | 22/36 | 20.00864 | 23/36 | 19.18394 | 22/36 |
| 202 | C | 20.62941 | 22/36 | 20.62941 | 22/36 | 18.26383 | 22/36 | 18.26383 | 22/36 |
| 202 | D | 22.50124 | 24/36 | 20.94423 | 22/36 | 19.67056 | 23/36 | 18.93803 | 22/36 |

## Per-instance results — seed 101

Each candidate cell: **CZ / moments / Pauli median / HW median**.

| N | n | A (reference) | B (root) | C (ColumnForge) | D (Astra) | Pauli winner | HW winner |
|---:|---:|:---|:---|:---|:---|:---|:---|
| 2 | 1 | 1 / 3 / **4.86662** / **4.09010** | 1 / 3 / — / — (constructed) | 1 / 3 / **4.86662** / **4.09010** | 1 / 3 / **4.86662** / **4.09010** | A (1 CZ), C (1 CZ), D (1 CZ) † | A (1 CZ), C (1 CZ), D (1 CZ) † |
| 3 | 1 | 3 / 6 / **2.21302** / **1.93475** | 3 / 7 / — / — (constructed) | 3 / 7 / — / — (constructed) | 3 / 6 / **2.21302** / **1.93475** | A (3 CZ), D (3 CZ) † | A (3 CZ), D (3 CZ) † |
| 4 | 1 | 5 / 11 / **1.69537** / **1.71028** | 5 / 7 / — / — (constructed) | 5 / 11 / — / — (constructed) | 5 / 10 / — / — (constructed) | A (5 CZ) † | A (5 CZ) † |
| 4 | 2 | 5 / 7 / **1.37499** / **1.26849** | 5 / 7 / — / — (constructed) | 5 / 7 / — / — (constructed) | 5 / 7 / **1.37499** / **1.26849** | A (5 CZ), D (5 CZ) † | A (5 CZ), D (5 CZ) † |
| 5 | 1 | 7 / 15 / **1.40993** / **1.37996** | 7 / 11 / — / — (constructed) | 7 / 15 / — / — (constructed) | 7 / 9 / — / — (constructed) | A (7 CZ) † | A (7 CZ) † |
| 5 | 2 | 10 / 15 / **1.09751** / **1.15025** | 9 / 15 / — / — (constructed) | 10 / 15 / — / — (constructed) | 13 / 18 / — / — (constructed) | A (10 CZ) † | A (10 CZ) † |
| 6 | 1 | 9 / 18 / 1.25459 / 1.29640 | 9 / 11 / 1.27017 / 1.29471 | 9 / 19 / **1.33419** / **1.34931** | 9 / 18 / 1.25459 / 1.29640 | C (9 CZ) | C (9 CZ) |
| 6 | 2 | 15 / 16 / **1.06496** / 1.00892 | 15 / 17 / 1.05457 / **1.02712** | 15 / 16 / 1.01689 / 1.01255 | 15 / 15 / 1.01529 / 1.02321 | A (15 CZ) | B (15 CZ) |
| 6 | 3 | 14 / 18 / 0.98739 / 0.92287 | 14 / 15 / 1.00802 / **0.97549** | 14 / 18 / **1.00983** / 0.94693 | 17 / 23 / 1.00560 / 0.95313 | C (14 CZ) | B (14 CZ) |
| 7 | 1 | 11 / 10 / 1.13283 / **1.27157** | 11 / 11 / 1.15179 / 1.12036 | 11 / 23 / **1.24305** / 1.19276 | 11 / 10 / 1.13283 / **1.27157** | C (11 CZ) | A (11 CZ), D (11 CZ) |
| 7 | 2 | 19 / 18 / 1.00965 / **0.96591** | 18 / 21 / 1.01174 / 0.95787 | 22 / 27 / **1.01394** / 0.95008 | 19 / 18 / 1.00965 / **0.96591** | C (22 CZ) | A (19 CZ), D (19 CZ) |
| 7 | 3 | 21 / 27 / 0.95098 / **0.90834** | 21 / 27 / 0.95104 / 0.87932 | 21 / 27 / **0.97562** / 0.88326 | 30 / 34 / 0.93436 / 0.86415 | C (21 CZ) | A (21 CZ) |
| 8 | 1 | 13 / 26 / **1.18262** / 1.05050 | 13 / 15 / 1.17671 / **1.14998** | 13 / 27 / 1.16380 / 1.05423 | 13 / 26 / **1.18262** / 1.05050 | A (13 CZ), D (13 CZ) | B (13 CZ) |
| 8 | 2 | 23 / 19 / **0.99555** / 0.89330 | 22 / 23 / 0.96500 / **0.91526** | 29 / 28 / 0.95479 / 0.89449 | 23 / 19 / **0.99555** / 0.89330 | A (23 CZ), D (23 CZ) | B (22 CZ) |
| 8 | 3 | 29 / 29 / **0.92657** / 0.82844 | 29 / 29 / 0.90530 / **0.84262** | 47 / 47 / 0.86195 / 0.73893 | 39 / 37 / 0.87184 / 0.78916 | A (29 CZ) | B (29 CZ) |
| 8 | 4 | 24 / 24 / **0.90954** / **0.79507** | 45 / 43 / 0.84735 / 0.72157 | 33 / 36 / 0.88909 / 0.78364 | 39 / 45 / 0.87631 / 0.74253 | A (24 CZ) | A (24 CZ) |
| 9 | 1 | 15 / 12 / **1.18684** / **1.09886** | 15 / 15 / 1.17274 / 1.08471 | 15 / 31 / 1.16735 / 1.02625 | 15 / 12 / **1.18684** / **1.09886** | A (15 CZ), D (15 CZ) | A (15 CZ), D (15 CZ) |
| 9 | 2 | 27 / 22 / **0.97991** / 0.88573 | 26 / 29 / 0.94824 / **0.88925** | 36 / 39 / 0.90577 / 0.85133 | 27 / 22 / **0.97991** / 0.88573 | A (27 CZ), D (27 CZ) | B (26 CZ) |
| 9 | 3 | 48 / 48 / 0.83564 / 0.70785 | 48 / 47 / **0.83707** / **0.72610** | 59 / 62 / 0.82553 / 0.64834 | 48 / 48 / 0.83564 / 0.70785 | B (48 CZ) | B (48 CZ) |
| 9 | 4 | 61 / 62 / 0.76723 / 0.59323 | 59 / 61 / **0.77954** / **0.59615** | 66 / 72 / 0.75487 / 0.57229 | 61 / 62 / 0.76723 / 0.59323 | B (59 CZ) | B (59 CZ) |
| 10 | 1 | 17 / 13 / 1.13618 / **1.08985** | 17 / 15 / 1.10979 / 1.08028 | 17 / 35 / **1.13857** / 1.02114 | 17 / 13 / 1.13618 / **1.08985** | C (17 CZ) | A (17 CZ), D (17 CZ) |
| 10 | 2 | 31 / 23 / 0.92139 / **0.89105** | 29 / 27 / **0.93592** / 0.88485 | 43 / 40 / 0.87541 / 0.80305 | 31 / 23 / 0.92139 / **0.89105** | B (29 CZ) | A (31 CZ), D (31 CZ) |
| 10 | 3 | 57 / 51 / 0.80577 / 0.71129 | 45 / 45 / **0.82319** / **0.74941** | 83 / 127 / 0.77395 / 0.58585 | 57 / 51 / 0.80577 / 0.71129 | B (45 CZ) | B (45 CZ) |
| 10 | 4 | 75 / 67 / **0.74848** / 0.56764 | 73 / 65 / 0.74628 / **0.57413** | 83 / 77 / 0.71489 / 0.52432 | 75 / 67 / **0.74848** / 0.56764 | A (75 CZ), D (75 CZ) | B (73 CZ) |
| 10 | 5 | 73 / 73 / **0.74285** / **0.55221** | 79 / 73 / 0.66170 / 0.48875 | 81 / 84 / 0.70855 / 0.54067 | 73 / 73 / **0.74285** / **0.55221** | A (73 CZ), D (73 CZ) | A (73 CZ), D (73 CZ) |
| 11 | 1 | 19 / 13 / 1.07919 / 1.02188 | 19 / 15 / **1.10806** / **1.02487** | 19 / 15 / 1.08505 / 0.98911 | 19 / 13 / 1.07919 / 1.02188 | B (19 CZ) | B (19 CZ) |
| 11 | 2 | 35 / 26 / **0.90084** / **0.85448** | 33 / 33 / 0.89506 / 0.83997 | 50 / 51 / 0.86162 / 0.76197 | 35 / 26 / **0.90084** / **0.85448** | A (35 CZ), D (35 CZ) | A (35 CZ), D (35 CZ) |
| 11 | 3 | 66 / 62 / **0.79690** / 0.66915 | 66 / 61 / 0.79017 / **0.67158** | 95 / 145 / 0.74784 / 0.53186 | 66 / 62 / **0.79690** / 0.66915 | A (66 CZ), D (66 CZ) | B (66 CZ) |
| 11 | 4 | 89 / 80 / **0.70174** / **0.52238** | 97 / 89 / 0.68338 / 0.51451 | 100 / 90 / 0.68239 / 0.47383 | 89 / 80 / **0.70174** / **0.52238** | A (89 CZ), D (89 CZ) | A (89 CZ), D (89 CZ) |
| 11 | 5 | 102 / 90 / **0.65020** / 0.40416 | 106 / 99 / 0.62081 / **0.41657** | 107 / 100 / 0.64192 / — (constructed) | 102 / 90 / **0.65020** / 0.40416 | A (102 CZ), D (102 CZ) | B (106 CZ) † |
| 12 | 1 | 21 / 13 / — / — (constructed) | 21 / 15 / — / — (constructed) | 21 / 15 / — / — (constructed) | 21 / 13 / — / — (constructed) | pending | pending |
| 12 | 2 | 39 / 27 / — / — (constructed) | 38 / 35 / — / — (constructed) | 68 / 117 / — / — (constructed) | 39 / 27 / — / — (constructed) | pending | pending |
| 12 | 3 | 75 / 65 / — / — (constructed) | 70 / 61 / — / — (constructed) | 107 / 163 / — / — (constructed) | 75 / 65 / — / — (constructed) | pending | pending |
| 12 | 4 | 103 / 85 / — / — (constructed) | 101 / 85 / — / — (constructed) | 117 / 95 / — / — (constructed) | 103 / 85 / — / — (constructed) | pending | pending |
| 12 | 5 | 143 / 165 / — / — (constructed) | 127 / 105 / — / — (constructed) | 129 / 107 / — / — (constructed) | 143 / 165 / — / — (constructed) | pending | pending |
| 12 | 6 | 117 / 101 / — / — (constructed) | 129 / 105 / — / — (constructed) | 126 / 114 / — / — (constructed) | 117 / 101 / — / — (constructed) | pending | pending |

## Per-instance results — seed 202

Each candidate cell: **CZ / moments / Pauli median / HW median**.

| N | n | A (reference) | B (root) | C (ColumnForge) | D (Astra) | Pauli winner | HW winner |
|---:|---:|:---|:---|:---|:---|:---|:---|
| 2 | 1 | 1 / 3 / — / — (constructed) | 1 / 3 / — / — (constructed) | 1 / 3 / — / — (constructed) | 1 / 3 / — / — (constructed) | pending | pending |
| 3 | 1 | 3 / 7 / — / — (constructed) | 3 / 7 / — / — (constructed) | 3 / 7 / — / — (constructed) | 3 / 7 / — / — (constructed) | pending | pending |
| 4 | 1 | 5 / 7 / — / — (constructed) | 5 / 7 / — / — (constructed) | 5 / 11 / — / — (constructed) | 5 / 7 / — / — (constructed) | pending | pending |
| 4 | 2 | 5 / 7 / — / — (constructed) | 5 / 7 / — / — (constructed) | 5 / 7 / — / — (constructed) | 5 / 7 / — / — (constructed) | pending | pending |
| 5 | 1 | 7 / 15 / — / — (constructed) | 7 / 11 / — / — (constructed) | 7 / 15 / — / — (constructed) | 7 / 15 / — / — (constructed) | pending | pending |
| 5 | 2 | 10 / 17 / — / — (constructed) | 10 / 17 / — / — (constructed) | 9 / 14 / — / — (constructed) | 13 / 18 / — / — (constructed) | pending | pending |
| 6 | 1 | 9 / 11 / 1.23273 / 1.26748 | 9 / 11 / **1.31070** / **1.30200** | 9 / 11 / 1.26487 / 1.26638 | 9 / 18 / 1.26030 / 1.21775 | B (9 CZ) | B (9 CZ) |
| 6 | 2 | 15 / 15 / 1.02661 / 0.94389 | 15 / 17 / **1.07378** / **0.98933** | 15 / 16 / 1.06226 / 0.97252 | 15 / 15 / 1.02661 / 0.94389 | B (15 CZ) | B (15 CZ) |
| 6 | 3 | 14 / 18 / 1.00939 / 0.90228 | 14 / 15 / 0.98449 / 0.94113 | 14 / 18 / **1.03485** / **0.96690** | 17 / 23 / 0.99163 / 0.89474 | C (14 CZ) | C (14 CZ) |
| 7 | 1 | 11 / 10 / **1.23966** / 1.22540 | 11 / 15 / 1.19069 / 1.17961 | 11 / 23 / 1.19118 / **1.26949** | 11 / 10 / **1.23966** / 1.22540 | A (11 CZ), D (11 CZ) | C (11 CZ) |
| 7 | 2 | 19 / 18 / **1.02384** / **0.98767** | 18 / 21 / 1.00953 / 0.97677 | 22 / 27 / 0.97168 / 0.94465 | 19 / 18 / **1.02384** / **0.98767** | A (19 CZ), D (19 CZ) | A (19 CZ), D (19 CZ) |
| 7 | 3 | 21 / 27 / 0.93410 / **0.88774** | 21 / 27 / **0.95884** / 0.88514 | 21 / 27 / 0.95884 / 0.87919 | 30 / 34 / 0.91659 / 0.86275 | B (21 CZ) | A (21 CZ) |
| 8 | 1 | 13 / 11 / 1.17326 / 1.11371 | 13 / 15 / 1.12222 / **1.17216** | 13 / 27 / **1.19537** / 1.04015 | 13 / 11 / 1.17326 / 1.11371 | C (13 CZ) | B (13 CZ) |
| 8 | 2 | 23 / 19 / 0.94275 / 0.91360 | 22 / 23 / **0.95963** / **0.92990** | 29 / 28 / 0.95344 / 0.87849 | 23 / 19 / 0.94275 / 0.91360 | B (22 CZ) | B (22 CZ) |
| 8 | 3 | 29 / 29 / **0.91683** / **0.83956** | 29 / 29 / 0.90601 / 0.82470 | 47 / 47 / 0.83742 × / 0.76183 × ⚠ | 47 / 69 / 0.89829 / 0.73253 | A (29 CZ) † | A (29 CZ) † |
| 8 | 4 | 24 / 24 / 0.87769 / **0.81230** | 45 / 43 / 0.80297 / 0.63116 | 33 / 36 / **0.88308** / 0.78757 | 39 / 45 / 0.85881 / 0.75501 | C (33 CZ) | A (24 CZ) |
| 9 | 1 | 15 / 12 / 1.11708 / 1.09685 | 15 / 15 / **1.13934** / **1.13513** | 15 / 31 / 1.07699 / 1.08420 | 15 / 12 / 1.11708 / 1.09685 | B (15 CZ) | B (15 CZ) |
| 9 | 2 | 27 / 22 / **0.95721** / **0.91896** | 26 / 29 / 0.95412 / 0.88485 | 36 / 39 / 0.87976 / 0.79073 | 27 / 22 / **0.95721** / **0.91896** | A (27 CZ), D (27 CZ) | A (27 CZ), D (27 CZ) |
| 9 | 3 | 48 / 48 / **0.85019** / **0.74460** | 48 / 47 / 0.83621 / 0.74324 | 71 / 109 / 0.83333 / 0.60781 | 48 / 48 / **0.85019** / **0.74460** | A (48 CZ), D (48 CZ) | A (48 CZ), D (48 CZ) |
| 9 | 4 | 61 / 62 / **0.77932** / **0.60183** | 59 / 61 / 0.75598 / 0.60051 | 66 / 72 / 0.76202 / 0.58106 | 61 / 62 / **0.77932** / **0.60183** | A (61 CZ), D (61 CZ) | A (61 CZ), D (61 CZ) |
| 10 | 1 | 17 / 34 / 1.03631 / 1.05670 | 17 / 15 / **1.15513** / **1.08986** | 17 / 35 / 1.06987 / 1.08565 | 17 / 34 / 1.03631 / 1.05670 | B (17 CZ) | B (17 CZ) |
| 10 | 2 | 33 / 38 / **0.91561** / 0.84335 | 29 / 27 / 0.90952 / **0.89661** | 43 / 40 / 0.85319 / 0.79526 | 33 / 38 / **0.91561** / 0.84335 | A (33 CZ), D (33 CZ) | B (29 CZ) |
| 10 | 3 | 57 / 51 / 0.83926 / 0.73960 | 45 / 45 / **0.85185** / **0.76556** | 83 / 127 / 0.80077 / 0.59649 | 57 / 51 / 0.83926 / 0.73960 | B (45 CZ) | B (45 CZ) |
| 10 | 4 | 89 / 119 / **0.74194** / 0.47555 | 73 / 65 / 0.72872 / **0.55698** | 104 / 137 / 0.72844 / 0.44675 | 89 / 119 / **0.74194** / 0.47555 | A (89 CZ), D (89 CZ) | B (73 CZ) |
| 10 | 5 | 73 / 73 / 0.70865 / **0.54508** | 79 / 73 / 0.66543 / 0.51393 | 81 / 84 / **0.71228** / 0.53772 | 73 / 73 / 0.70865 / **0.54508** | C (81 CZ) | A (73 CZ), D (73 CZ) |
| 11 | 1 | 19 / 13 / 1.13819 / **1.03759** | 19 / 15 / **1.16688** / 1.03389 | 19 / 38 / 1.07653 / 1.00430 | 19 / 13 / 1.13819 / **1.03759** | B (19 CZ) | A (19 CZ), D (19 CZ) |
| 11 | 2 | 37 / 42 / **0.91060** / 0.78567 | 33 / 33 / 0.87905 / **0.82052** | 61 / 105 / 0.87895 / 0.71797 | 37 / 42 / **0.91060** / 0.78567 | A (37 CZ), D (37 CZ) | B (33 CZ) |
| 11 | 3 | 66 / 62 / 0.79990 / 0.65158 | 66 / 61 / **0.80458** / **0.65574** | 83 / 80 / 0.76745 / 0.60073 | 66 / 62 / 0.79990 / 0.65158 | B (66 CZ) | B (66 CZ) |
| 11 | 4 | 89 / 80 / **0.71654** / **0.52615** | 97 / 89 / 0.68077 / 0.47992 | 121 / 155 / 0.67424 / 0.40981 | 89 / 80 / **0.71654** / **0.52615** | A (89 CZ), D (89 CZ) | A (89 CZ), D (89 CZ) |
| 11 | 5 | 124 / 147 / — / — (constructed) | 106 / 99 / — / — (constructed) | 107 / 100 / — / — (constructed) | 124 / 147 / **0.65872** / — (constructed) | D (124 CZ) † | pending |
| 12 | 1 | 21 / 13 / — / — (constructed) | 21 / 15 / — / — (constructed) | 21 / 42 / — / — (constructed) | 21 / 13 / — / — (constructed) | pending | pending |
| 12 | 2 | 41 / 46 / — / — (constructed) | 38 / 35 / — / — (constructed) | 68 / 117 / — / — (constructed) | 41 / 46 / — / — (constructed) | pending | pending |
| 12 | 3 | 75 / 65 / — / — (constructed) | 70 / 61 / — / — (constructed) | 107 / 163 / — / — (constructed) | 75 / 65 / — / — (constructed) | pending | pending |
| 12 | 4 | 117 / 155 / — / — (constructed) | 101 / 85 / — / — (constructed) | 138 / 173 / — / — (constructed) | 117 / 155 / — / — (constructed) | pending | pending |
| 12 | 5 | 143 / 165 / — / — (constructed) | 127 / 105 / — / — (constructed) | 161 / 183 / — / — (constructed) | 143 / 165 / — / — (constructed) | pending | pending |
| 12 | 6 | 117 / 101 / — / — (constructed) | 129 / 105 / — / — (constructed) | 126 / 114 / — / — (constructed) | 117 / 101 / — / — (constructed) | pending | pending |

## Per-instance winner across calibration seeds

A candidate enters this table only when both models are eligible and scored for every requested seed. Winners maximize the arithmetic mean of the seed medians. CZ counts follow seed order 101, 202. A † winner remains provisional until all four candidates have paired complete results.

| N | n | Paired candidates | Pauli winner (CZ by seed) | Mean Pauli median | HW winner (CZ by seed) | Mean HW median |
|---:|---:|:---|:---|---:|:---|---:|
| 2 | 1 | — | pending | — | pending | — |
| 3 | 1 | — | pending | — | pending | — |
| 4 | 1 | — | pending | — | pending | — |
| 4 | 2 | — | pending | — | pending | — |
| 5 | 1 | — | pending | — | pending | — |
| 5 | 2 | — | pending | — | pending | — |
| 6 | 1 | A,B,C,D | C (9/9) | 1.29953 | C (9/9) | 1.30784 |
| 6 | 2 | A,B,C,D | B (15/15) | 1.06417 | B (15/15) | 1.00823 |
| 6 | 3 | A,B,C,D | C (14/14) | 1.02234 | B (14/14) | 0.95831 |
| 7 | 1 | A,B,C,D | C (11/11) | 1.21712 | A (11/11), D (11/11) | 1.24848 |
| 7 | 2 | A,B,C,D | A (19/19), D (19/19) | 1.01675 | A (19/19), D (19/19) | 0.97679 |
| 7 | 3 | A,B,C,D | C (21/21) | 0.96723 | A (21/21) | 0.89804 |
| 8 | 1 | A,B,C,D | C (13/13) | 1.17959 | B (13/13) | 1.16107 |
| 8 | 2 | A,B,C,D | A (23/23), D (23/23) | 0.96915 | B (22/22) | 0.92258 |
| 8 | 3 | A,B,D | A (29/29) † | 0.92170 | A (29/29) † | 0.83400 |
| 8 | 4 | A,B,C,D | A (24/24) | 0.89362 | A (24/24) | 0.80369 |
| 9 | 1 | A,B,C,D | B (15/15) | 1.15604 | B (15/15) | 1.10992 |
| 9 | 2 | A,B,C,D | A (27/27), D (27/27) | 0.96856 | A (27/27), D (27/27) | 0.90234 |
| 9 | 3 | A,B,C,D | A (48/48), D (48/48) | 0.84291 | B (48/48) | 0.73467 |
| 9 | 4 | A,B,C,D | A (61/61), D (61/61) | 0.77327 | B (59/59) | 0.59833 |
| 10 | 1 | A,B,C,D | B (17/17) | 1.13246 | B (17/17) | 1.08507 |
| 10 | 2 | A,B,C,D | B (29/29) | 0.92272 | B (29/29) | 0.89073 |
| 10 | 3 | A,B,C,D | B (45/45) | 0.83752 | B (45/45) | 0.75748 |
| 10 | 4 | A,B,C,D | A (75/89), D (75/89) | 0.74521 | B (73/73) | 0.56556 |
| 10 | 5 | A,B,C,D | A (73/73), D (73/73) | 0.72575 | A (73/73), D (73/73) | 0.54864 |
| 11 | 1 | A,B,C,D | B (19/19) | 1.13747 | A (19/19), D (19/19) | 1.02973 |
| 11 | 2 | A,B,C,D | A (35/37), D (35/37) | 0.90572 | B (33/33) | 0.83024 |
| 11 | 3 | A,B,C,D | A (66/66), D (66/66) | 0.79840 | B (66/66) | 0.66366 |
| 11 | 4 | A,B,C,D | A (89/89), D (89/89) | 0.70914 | A (89/89), D (89/89) | 0.52427 |
| 11 | 5 | — | pending | — | pending | — |
| 12 | 1 | — | pending | — | pending | — |
| 12 | 2 | — | pending | — | pending | — |
| 12 | 3 | — | pending | — | pending | — |
| 12 | 4 | — | pending | — | pending | — |
| 12 | 5 | — | pending | — | pending | — |
| 12 | 6 | — | pending | — | pending | — |

Complete four-candidate paired comparisons: 22/36. Winner counts (ties count for each candidate): Pauli A=11, B=6, C=5, D=10; HW A=8, B=13, C=1, D=6.

## Failures and validation

| Seed | N | n | Candidate | Construction seconds | Reasons |
|---:|---:|---:|:---|---:|:---|
| 202 | 8 | 3 | C | 67.273 | construction_over_60_seconds; construction >60 s (67.273 s) |

## Run configuration and verification

Recorded run configuration:

```json
{
  "created_unix": 1792912727.5865963,
  "seeds": [
    101,
    202
  ],
  "instances_per_seed": 36,
  "expected_records": 288,
  "python": "C:/Users/kaisa/miniforge3/python.exe",
  "max_workers": 4,
  "noise_models": {
    "pauli": {
      "T1": 1000000000.0,
      "Tphi_white": 0,
      "T2star": 0,
      "n_traj": 2000
    },
    "hw": {
      "defaults": "tools.sim.HW()",
      "n_traj": 1000
    }
  },
  "shots_per_basis": 10000,
  "shot_resamplings": 400,
  "max_correlator_size": "N",
  "candidates": {
    "A": "team_eval/will_dicke_submission.py",
    "B": "submission.py",
    "C": "columnforge/noise_aware/rewrites/submission_scheduled.py",
    "D": "astra/submission.py"
  },
  "simulator_proxy_only": true,
  "exactness_threshold": 0.99999999,
  "construction_limit_seconds": 60,
  "tools_sim_sha256": "39a53de0428c2503ab65407b85af3f7d9d72b8f10716b7030758758f95c8cd3a",
  "deduplication": "Reuse only byte-identical circuit JSON, same qubits, calibration, N,n,seed.",
  "updated_unix": 1792915299.1021483,
  "elapsed_seconds": 2571.515552043915,
  "active_workers": [
    {
      "name": "score_pool_8088_1",
      "pid": 39012
    },
    {
      "name": "score_pool_8088_2",
      "pid": 32872
    },
    {
      "name": "score_pool_8088_3",
      "pid": 19136
    },
    {
      "name": "score_pool_8088_4",
      "pid": 26280
    }
  ],
  "pool_mode": "persistent CPU workers; imports and JIT reused",
  "score_parity": {
    "python": "C:\\Users\\kaisa\\miniforge3\\python.exe",
    "pythonhashseed": "0",
    "calibrations": {
      "101": {
        "path": "team_eval\\calibrations\\seed_101.json",
        "fresh_sha256": "3e78cf23ad04e115b3dfacc7071885dc63f1d668394387076ec750f2195decac",
        "saved_sha256": "3e78cf23ad04e115b3dfacc7071885dc63f1d668394387076ec750f2195decac",
        "equal": true,
        "n_qubits": 105,
        "n_couplers": 181
      },
      "202": {
        "path": "team_eval\\calibrations\\seed_202.json",
        "fresh_sha256": "bc3b6d6ad048a333cfa6a07cafbf30268039ae1f621ea3e449e26a17489a1f42",
        "saved_sha256": "bc3b6d6ad048a333cfa6a07cafbf30268039ae1f621ea3e449e26a17489a1f42",
        "equal": true,
        "n_qubits": 103,
        "n_couplers": 173
      }
    },
    "generators_verbatim": true,
    "cases": [
      {
        "N": 2,
        "n": 1,
        "K": 2,
        "shots": 10000,
        "distribution": "ideal_dicke",
        "fast_score": 296.8239833778569,
        "official_score": 296.82398337785713,
        "fast_Z": 0.0032,
        "official_Z": 0.0032,
        "fast_X": 0.000169,
        "official_X": 0.0001689999999999974,
        "shell_fraction": 1.0,
        "max_abs_difference": 2.2737367544323206e-13,
        "passed": true
      },
      {
        "N": 2,
        "n": 1,
        "K": 2,
        "shots": 10000,
        "distribution": "noisy_nonuniform",
        "fast_score": 0.10656294425811355,
        "official_score": 0.10656294425811355,
        "fast_Z": 9.358319450539922,
        "official_Z": 9.358319450539922,
        "fast_X": 0.025805650000000003,
        "official_X": 0.025805650000000006,
        "shell_fraction": 0.8876,
        "max_abs_difference": 3.469446951953614e-18,
        "passed": true
      },
      {
        "N": 3,
        "n": 1,
        "K": 3,
        "shots": 10000,
        "distribution": "ideal_dicke",
        "fast_score": 1.288936730655864,
        "official_score": 1.2889367306558637,
        "fast_Z": 0.7754666666666666,
        "official_Z": 0.7754666666666667,
        "fast_X": 0.00036660444444444574,
        "official_X": 0.00036660444444444606,
        "shell_fraction": 1.0,
        "max_abs_difference": 2.220446049250313e-16,
        "passed": true
      },
      {
        "N": 3,
        "n": 1,
        "K": 3,
        "shots": 10000,
        "distribution": "noisy_nonuniform",
        "fast_score": 0.04417446114598871,
        "official_score": 0.04417446114598872,
        "fast_Z": 22.60621979310552,
        "official_Z": 22.606219793105517,
        "fast_X": 0.031294599999999985,
        "official_X": 0.03129459999999997,
        "shell_fraction": 0.8791,
        "max_abs_difference": 3.552713678800501e-15,
        "passed": true
      },
      {
        "N": 4,
        "n": 2,
        "K": 4,
        "shots": 10000,
        "distribution": "ideal_dicke",
        "fast_score": 1.2147106316333323,
        "official_score": 1.2147106316333323,
        "fast_Z": 0.8225333333333333,
        "official_Z": 0.8225333333333333,
        "fast_X": 0.0007080000000000034,
        "official_X": 0.0007080000000000072,
        "shell_fraction": 1.0,
        "max_abs_difference": 3.7947076036992655e-18,
        "passed": true
      },
      {
        "N": 4,
        "n": 2,
        "K": 4,
        "shots": 10000,
        "distribution": "noisy_nonuniform",
        "fast_score": 0.07304461520758317,
        "official_score": 0.07304461520758315,
        "fast_Z": 13.620752976850216,
        "official_Z": 13.62075297685022,
        "fast_X": 0.06951012000000001,
        "official_X": 0.06951012,
        "shell_fraction": 0.8725,
        "max_abs_difference": 3.552713678800501e-15,
        "passed": true
      },
      {
        "N": 6,
        "n": 2,
        "K": 6,
        "shots": 10000,
        "distribution": "ideal_dicke",
        "fast_score": 1.1036550349887344,
        "official_score": 1.1036550349887344,
        "fast_Z": 0.9057333333333335,
        "official_Z": 0.9057333333333335,
        "fast_X": 0.0003468894694444439,
        "official_X": 0.00034688946944444384,
        "shell_fraction": 1.0,
        "max_abs_difference": 5.421010862427522e-20,
        "passed": true
      },
      {
        "N": 6,
        "n": 2,
        "K": 6,
        "shots": 10000,
        "distribution": "noisy_nonuniform",
        "fast_score": 0.1254990957373175,
        "official_score": 0.12549909573731752,
        "fast_Z": 7.9269759225903025,
        "official_Z": 7.9269759225903,
        "fast_X": 0.04120898045555557,
        "official_X": 0.04120898045555561,
        "shell_fraction": 0.8475,
        "max_abs_difference": 2.6645352591003757e-15,
        "passed": true
      },
      {
        "N": 8,
        "n": 4,
        "K": 8,
        "shots": 10000,
        "distribution": "ideal_dicke",
        "fast_score": 1.259072789163883,
        "official_score": 1.2590727891638833,
        "fast_Z": 0.7940571428571428,
        "official_Z": 0.7940571428571426,
        "fast_X": 0.00017811391786343956,
        "official_X": 0.00017811391786343736,
        "shell_fraction": 1.0,
        "max_abs_difference": 2.220446049250313e-16,
        "passed": true
      },
      {
        "N": 8,
        "n": 4,
        "K": 8,
        "shots": 10000,
        "distribution": "noisy_nonuniform",
        "fast_score": 0.3471736658110572,
        "official_score": 0.3471736658110573,
        "fast_Z": 2.74345428887812,
        "official_Z": 2.743454288878119,
        "fast_X": 0.1369485137475592,
        "official_X": 0.13694851374755923,
        "shell_fraction": 0.8523,
        "max_abs_difference": 8.881784197001252e-16,
        "passed": true
      },
      {
        "N": 12,
        "n": 6,
        "K": 12,
        "shots": 10000,
        "distribution": "ideal_dicke",
        "fast_score": 1.0008035289687371,
        "official_score": 1.0008035289687358,
        "fast_Z": 0.9990891774891774,
        "official_Z": 0.9990891774891786,
        "fast_X": 0.00010793868250015553,
        "official_X": 0.00010793868250016166,
        "shell_fraction": 1.0,
        "max_abs_difference": 1.3322676295501878e-15,
        "passed": true
      },
      {
        "N": 12,
        "n": 6,
        "K": 12,
        "shots": 10000,
        "distribution": "noisy_nonuniform",
        "fast_score": 0.7259022504297651,
        "official_score": 0.7259022504297649,
        "fast_Z": 1.255360051614674,
        "official_Z": 1.2553600516146746,
        "fast_X": 0.12223589798867465,
        "official_X": 0.12223589798867439,
        "shell_fraction": 0.8432,
        "max_abs_difference": 6.661338147750939e-16,
        "passed": true
      }
    ],
    "edge_case": {
      "case": "zero_correct_weight_z_shots",
      "N": 4,
      "n": 2,
      "K": 4,
      "official_score": "NaN",
      "fast_score": 0.0,
      "note": "The official notebook divides by zero; tools.sim returns zero by convention."
    },
    "all_sampled_cases_pass": true,
    "maximum_absolute_difference": 2.2737367544323206e-13
  }
}
```
