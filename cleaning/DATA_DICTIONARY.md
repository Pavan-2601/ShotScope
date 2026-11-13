# Cleaned ShuttleSet and ShuttleSet22: data dictionary

Produced by `cleaning/clean_shuttleset.py`. **The two datasets are cleaned separately and never combined**: ShuttleSet (44 matches, the **training** data) goes to `data/clean/`, and ShuttleSet22 (the **held-out test** data, 50 matches) goes to `data/clean_shuttleset22/`. Both use the same columns, so one analysis script can read either, but every row carries `source` and `match_uid` and no file mixes them. Checks are in `cleaning/test_clean.py` (96 checks, including separation checks). Counts and findings are in each folder's `cleaning_report.md`.

| File | Rows | What it is |
|---|---|---|
| `data/clean/strokes_clean.csv` | 36,482 | Every stroke, with labels and court positions in metres |
| `data/clean/smashes_clean.csv` | 4,234 | The strokes that are `smash` or `wrist smash` (same columns) |
| `data/clean/matches_clean.csv` | 44 | One row per match: names, gender, date, court frame |
| `data/clean/players.csv` | 27 | One row per player: gender and smash counts |
| `data/external/player_attributes.csv` | 27 | Height, handedness and birth date from outside sources (built by `cleaning/build_player_attributes.py`; see the last section) |
| `data/clean_shuttleset22/strokes_clean.csv` | 45,455 | **Test set.** Every stroke of the 50 new 2022 matches, same columns as above |
| `data/clean_shuttleset22/smashes_clean.csv` | 5,608 | Test-set strokes that are `smash` (3,470) or `wrist smash` (2,138) |
| `data/clean_shuttleset22/matches_clean.csv` | 50 | The 50 test matches |
| `data/clean_shuttleset22/matches_all_shuttleset22.csv` | 58 | All 58 ShuttleSet22 matches with `is_duplicate_of_shuttleset`, `stroke_level_identical` and `in_test_set`; records the 8 matches that were removed |
| `data/clean_shuttleset22/players.csv` | 35 | Players in the test matches with gender and smash counts |
| `data/clean_shuttleset22/player_crosswalk.csv` | 35 | Maps each test player to the same person in ShuttleSet (`seen_in_shuttleset`). Names and counts only; this is the only link between the datasets |
| `data/external/player_attributes_shuttleset22.csv` | 35 | Attributes for the 35 test-set players (19 shared with ShuttleSet carry identical values; 16 are new) |

## Decisions made while cleaning

- **Duplicates.** 1 exact duplicate row and 1 repeated placeholder row (same video, set, rally and ball round) were dropped. Neither was a real stroke.
- **Placeholders.** 1,406 rows have type `unknown` (`未知球種`, almost all flagged `flaw = 1` in the raw data). 1,239 of them are the first stroke of a rally, an unannotated serve. They are kept (`is_placeholder = True`) so rally structure is intact, but they never count as a real stroke.
- **The "smash" definition (provisional, to confirm; the EDA weakens the first choice).** `smash_kind` is `smash` (2,586), `wrist smash` (1,648) or empty. The two behave differently (a smash wins the point directly 18.4% of the time, a wrist smash 8.3%, and within the same match smashes win more often in 36 of 41 matches). **But the split between them depends on the match**: the wrist-smash share of smash-type strokes runs from 4% to 96%, and in 4 matches (Antonsen v Christie, Marin v Katethong, Axelsen v Christie, Axelsen v Liew) 84% to 96% are filed as wrist smashes, while the two kinds together are steady across matches. The EDA therefore recommends using **both kinds together as the main group, with `smash_kind` as a feature**, and reporting `smash` only as a sensitivity check.
- **Which clock.** `frame_sec = frame_num / 30`. The raw `time` column (hh:mm:ss) disagrees with it by more than 1.5 s on about 16,800 strokes and has 1-second resolution, so it is not used.
- **Scores.** The raw `roundscore_A/B` are the score **after** the rally, so using them would leak the result. The `*_before` columns rebuild the score before the rally (post score minus the winner's point). They match the previous rally's score in 99.0% of rallies; the rest are rallies missing from the data.
- **Court frame.** Positions go through each match's homography into court units (about 49.2 units per metre, net at y = 480), then into metres (below).
- **Orientation.** Each stroke is rotated 180° if needed so that **the shooter is always on the top half** (small `y_m`), and the opponent on the bottom half (`y_m` above the net at 6.7 m). After rotation this holds for 100% of the 4,234 smash-type strokes.
- **Missing values.** Smashes missing `hit` coordinates: 71 smash, 43 wrist smash (all have landing, shooter and opponent positions). They stay in the table with `has_reach_inputs = False` and are excluded only from analyses that need the hit point.
- **Outcome rules** are described under the label columns below. Rallies with no recorded winner (174), and 29 with an ending that contradicts the winner or is flagged flawed, get empty S1/S2.

## Identifiers and context

| Column | Meaning |
|---|---|
| `source`, `match_uid` | `ShuttleSet` or `ShuttleSet22`, and a unique match id such as `SS-12` or `SS22-12`. **`match_id` alone is not unique across datasets** (both start at 1) |
| `match_id`, `video` | Match number within its dataset and the folder name in `set/` |
| `tournament`, `round`, `date`, `gender` | Match context. `gender` is `men` or `women` (16 men, 11 women; checked that each match is single-gender) |
| `set`, `rally`, `ball_round`, `rally_id` | Position in the match. `rally_id` is `video|set|rally` |
| `player` | `A` (the match winner) or `B`. `player_name` is the shooter's name; `opponent_name` is the other player |
| `type`, `type_zh` | Shot type in English and the original Chinese label (`passive drop` merges both Chinese spellings) |
| `is_placeholder`, `is_smash`, `is_wrist_smash`, `smash_kind` | See decisions above |
| `aroundhead`, `backhand` | 1 if the stroke was hit around the head / backhand, else 0 |
| `hit_height` | Height of the shuttle when this stroke is hit: **1 = above net height, 2 = below** (checked: smashes, clears and drops are 1; net shots, lobs and serves are 2). Empty for 42 strokes |
| `landing_height` | Raw code, 1 or 2, that equals the **next** stroke's `hit_height` in 97% of cases, so it describes the opponent's reply, not this stroke. Do not use as a feature |
| `hit_area`, `landing_area`, `player_location_area`, `opponent_location_area` | Court grid cell codes (1–9 in court, 10–16 outside) |
| `stroke_status` | `serve`, `rally` or `last` (the rally-ending stroke). See the leakage list |

## Timing and score (known before the stroke)

| Column | Meaning |
|---|---|
| `frame_num`, `frame_sec` | Frame number and seconds (frames / 30) |
| `score_A_before`, `score_B_before` | Score in the set before this rally |
| `shooter_score_before`, `opponent_score_before`, `score_diff_before` | Same, from the shooter's point of view; `score_diff_before` is shooter minus opponent |

## Court positions

Raw court units (`*_court_x`, `*_court_y`) are kept for tracing. The metre columns are in the rotated frame: `x_m` runs across the court (0 to about 6.1), `y_m` along it (0 at the shooter's baseline, 6.7 at the net, about 13.4 at the opponent's baseline).

| Metre columns | What they locate |
|---|---|
| `hit_x_m`, `hit_y_m` | Where the shuttle was hit (the shooter's contact point) |
| `land_x_m`, `land_y_m` | Where the stroke landed |
| `shooter_x_m`, `shooter_y_m` | Shooter's position when hitting |
| `opp_x_m`, `opp_y_m` | Opponent's position at that moment |
| `rotated` | 1 if this stroke was rotated 180° |
| `coord_suspect` | True if a player or opponent position is more than 3 m outside the court (0 cases in both datasets) |
| `geometry_ok` | False for every stroke of a rally in which most strokes put the opponent on the shooter's half of the court after rotation (a positions problem in the data). 0 rallies in ShuttleSet; 27 in ShuttleSet22 |
| `has_reach_inputs` | True if hit, landing, shooter and opponent positions are all present **and** `geometry_ok` |

## Outcome and label columns (these use information from after the stroke)

| Column | Meaning |
|---|---|
| `rally_winner` | `A` or `B`; empty if the rally has no recorded winner |
| `end_reason`, `end_category` | Why the rally ended (`landed`, `opponent misjudged landing`, `out`, `net`, `not over net`, `fault`); category `direct` (winner's stroke landed) or `error` (loser's own error) |
| `outcome_consistent` | 1 if the ending agrees with the winner and is not flagged flawed; 0 if it contradicts |
| `S1_rally_won` | **S1.** 1 if the shooter won the rally, else 0; empty if winner unknown or ending inconsistent. Only for smash-type strokes |
| `smash_result` | `returned` (the opponent hit it back), `won_directly` (last stroke, shooter won, shuttle landed or was misjudged), `shooter_error` (last stroke, shooter lost on out / net / fault), `ambiguous` (last stroke but the ending is inconsistent), `unknown` |
| `S2_won_directly` | **S2.** 1 if `won_directly`, 0 if `returned`, empty otherwise (shooter errors and unclear endings are reported separately, not as zeros) |
| `later_real_strokes` | Real strokes after this one in the rally (0 for the last) |
| `next_stroke_type`, `t_next_s`, `t_next_valid`, `t_next_for_smash_s` | Next stroke's type and the time to it. Empty when there is no next stroke or the gap is not positive (42 strokes) |
| `rally_len`, `rally_real_len`, `rally_has_placeholder` | Rally length (all rows, real strokes) and whether it has a placeholder |
| `match_winner_name` | The match winner. Description only |

## ShuttleSet22 (the test set): what is different

- **Same files, same code.** The 58 ShuttleSet22 matches have the same columns, shot types, end reasons and 30 fps clock as ShuttleSet. Its `time` column agrees with `frame_num / 30`; ShuttleSet's does not (neither is used).
- **8 matches removed.** They are in both datasets (same folder name, same date, same winner and loser) and the two copies are identical row for row. They stay in the training data only. All 58 are listed in `matches_all_shuttleset22.csv`.
- **Test set size.** 50 matches, 45,455 strokes, 4,205 rallies (92 with no recorded winner), 3,470 smashes and 2,138 wrist smashes. Smash outcomes: 2,734 returned, 449 won directly, 285 shooter errors, 2 ambiguous. Smashes per player: min 14, median 85, max 328.
- **Positions problem in one match.** In *Chen Yu Fei vs He Bing Jiao* (Indonesia Masters 2022 semi-final), set 2, from rally 9 to the end of the set both players are recorded on the same side of the net. The court calibration is fine (every position is inside the court), so it looks like an annotation or camera-view problem. The 27 rallies (423 strokes, 59 smash-type) are flagged `geometry_ok = False` and have `has_reach_inputs = False`. Their rally outcomes and shot types are still usable; their positions are not.
- **Swapped corner labels.** In 1 test match (*Nishimoto vs Antonsen*, Japan Open 2022 semi-final) the bottom court corners are listed left/right in swapped order (`corner_labels_swapped`). The homography itself is correct. An early version of the cleaning got this wrong (it shifted that match's x positions by 3 m); corners are now ordered by pixel position and the match is at 99.9% inside the court like the rest.
- **Many players are unseen.** 16 of the 35 test players never appear in ShuttleSet. They hit 1,716 of the 3,470 test smashes (49%) and are the opponent in 1,749. Of the 50 test matches, 14 have both players known, 25 one, 11 neither. Use `player_crosswalk.csv` to split results by known and unseen players.
- **Different gender mix.** Women hit 1,457 of the test smashes (42%) against 594 of 2,586 (23%) in ShuttleSet. Report results by gender.
- **Attributes.** The 16 new players have no height from BWF (it lists none, or a 0.00 placeholder), so their heights rest on Wikipedia alone; Wang Zhi Yi has no height or handedness at all, and Aakarshi Kashyap's BWF birth date is blank.
- **Gender of the new players** was inferred from their opponents (everyone in a match shares a gender, and each group of opponents contains a player whose gender is known from ShuttleSet).

## Do not use as model features

These columns describe how the rally or smash turned out, so using them would leak the answer:

`rally_winner`, `match_winner_name`, `end_reason`, `end_category`, `outcome_consistent`, `S1_rally_won`, `S2_won_directly`, `smash_result`, `later_real_strokes`, `next_stroke_type`, `t_next_s`, `t_next_valid`, `t_next_for_smash_s`, `rally_len`, `rally_real_len`, `stroke_status`, `landing_height`.

Also keep out of the model any player field computed from match results (for example win percentage). Player attributes used as features must come from outside sources (height, handedness, age) and are added separately.

Note on flight time: `t_next_s` exists only for returned smashes, so it would leak S2. The main model uses a distance-based flight-time estimate instead (Phase 5); `t_next_s` is only a check on that estimate.

## Player attributes (`data/external/player_attributes*.csv`)

Height, handedness and birth date for the 27 players, collected on 2026-11-05 from two public sources per player. None of it comes from match results.

| Column | Meaning |
|---|---|
| `player_name`, `gender`, `birth_year` | Same names as `players.csv`. `birth_year` is what to use for an age feature (both sources agree on it for all 27 players) |
| `height_cm` | Height used: the second source if it lists one, otherwise Wikipedia |
| `height_status` | `two sources agree` (7 players), `single source (Wikipedia)` (19), `single source (BWF)` (1: Rasmus Gemke) |
| `handedness`, `handedness_status` | `right` / `left` (3 left-handed: Marin, Momota, Katethong); 26 confirmed by two sources, Lee Zii Jia from BWF only |
| `dob_wikipedia`, `dob_other`, `dob_status` | Full birth dates from each source. Two conflict, both within the same year: Ginting (20 Oct vs 11 May 1996) and Kosetskaya (16 Dec vs 16 Nov 1994) |
| `other_source`, `other_url`, `wikipedia_url` | Provenance. The second source is the official BWF profile for 26 players and Olympics.com for Kento Momota |

Caveats: the BWF profile pages block direct reads (HTTP 403), so the BWF values were read through a web-search summary of the pages, and BWF lists height for only 7 of the 27 players. Heights for the other 19 come from Wikipedia alone. Treat opponent height (RQ2) with that in mind, and re-check any height that turns out to matter.
