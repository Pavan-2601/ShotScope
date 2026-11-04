# ShotScope: Shot Clusters and Opponent Reach in Badminton

DS-GA 3001, Fall 2026. Team: Ankur Mittal, Ishan Gupta, Pavan Kulkarni.

We study badminton **smashes** with the public ShuttleSet data: does the opponent's reach (how far they must move, and how much time they have) explain whether a smash wins the point, beyond where it landed? The analysis is retrospective (it explains outcomes given where the shot went; it does not predict at hit time). Results are meant to be shown in a D3 interface with the uncertainty behind each number.

## Status

Early stage. Be aware of what is and is not here:

| Part | State |
|---|---|
| Data (ShuttleSet, 44 matches) | In `set/`, loads fine |
| Cleaning | One-match merge script works; the other 43 matches are not merged yet |
| EDA | Done for **one match** (Axelsen vs. Ng, Thailand Open 2021 final, sets 1-2), including a first version of the smash reach features |
| Smash model, baselines, clusters | Not started |
| Shot Influence code (optional baseline) | Added **unmodified from the original authors** and **not yet run** (see below) |
| D3 interface | Not started |
| ShuttleSet22 (later-season test) | Not downloaded yet |

The full task list is in `plan.md`.

## Folder layout

| Path | What it is |
|---|---|
| `set/` | ShuttleSet files: per-match folders with `set1.csv`, `set2.csv`, (`set3.csv`), plus `match.csv` and `homography.csv` |
| `MergedSet/merge.py` | Merges one match into a single file with court coordinates in metres (using the homography) and translated shot types |
| `MergedSet/*.csv` | Merged output for the one match used in the EDA |
| `EDA/eda.py` | EDA script. Writes figures and CSVs into `EDA/` |
| `EDA/*.png`, `EDA/*.csv` | EDA output (winning shots, hit locations by shot type, stroke timing, smash reach features) |
| `Shot Influence/` | Code from the Shot Influence papers (MIT license). Not run yet |
| `plan.md` | Project plan and checklist |
| `feedback_response.md` | How we are applying the proposal feedback |
| `project_update.pdf` | The project update |

## How to run

Python 3 with `matplotlib` is enough for the EDA.

```
pip install matplotlib
python EDA/eda.py          # run from the repo root, not from inside EDA/
```

To merge a different match (the output is what `eda.py` reads):

```
python MergedSet/merge.py set <match_folder_name> <output.csv> [1,2]
```

`<match_folder_name>` is a folder name inside `set/`. The optional last argument picks which sets to include (default: set 1). `merge.py` reads its shot-type table from `README.md` in the repo root (see the note at the bottom).

## Data

- **ShuttleSet** (Wang et al., KDD 2023): 44 matches, 104 sets, 36,484 strokes in the files we have, including 2,586 smashes and 1,648 wrist smashes. Source and license: https://github.com/wywyWang/CoachAI-Projects (paper: https://arxiv.org/abs/2306.04948).
- Counts here come from the CSV files. Table 1 of the paper sums to far fewer strokes (about 24.5k), so we do not use it for sample sizes.
- **ShuttleSet22** (for the later-season test) is not in the repo yet.

## Shot Influence code

`Shot Influence/` is the original authors' implementation of the ICDM 2021 / ACM TIST 2022 papers (MIT license, see `Shot Influence/LICENSE`). We have not modified or run it. Known gaps before it can run on our data:

- It needs a `new_data/dataset.csv` that is not included; we must build it from ShuttleSet, including a choice of "target player" for each rally.
- Its pinned versions (Python 3.6, TensorFlow 2.0) do not install on current machines.
- `evaluate.py` has bugs (for example `load_weight` instead of `load_weights`), and the train/test split is hard-coded.

Details are in Phase 8 of `plan.md`. It is an optional extension; the main project does not depend on it.

## Note on the shot-type table

`MergedSet/merge.py` reads the shot-type translation table below from `README.md` at the repo root. If this file is renamed to `README.md`, keep the table in this exact format so `merge.py` keeps working.

| English | Chinese |
| ------- | ------- |
| net shot | 放小球 |
| return net | 擋小球 |
| smash | 殺球 |
| wrist smash | 點扣 |
| lob | 挑球 |
| defensive return lob | 防守回挑 |
| clear | 長球 |
| drive | 平球 |
| driven flight | 小平球 |
| back-court drive | 後場抽平球 |
| drop | 切球 |
| passive drop | 過渡切球 |
| push | 推球 |
| rush | 撲球 |
| defensive return drive | 防守回抽 |
| cross-court net shot | 勾球 |
| short service | 發短球 |
| long service | 發長球 |

## References

- W.-Y. Wang, Y.-C. Huang, T.-U. Ik, W.-C. Peng. *ShuttleSet: A Human-Annotated Stroke-Level Singles Dataset for Badminton Tactical Analysis.* KDD 2023.
- W.-Y. Wang, T.-F. Chan, H.-K. Yang, C.-C. Wang, Y.-C. Fan, W.-C. Peng. *Exploring the Long Short-Term Dependencies to Infer Shot Influence in Badminton Matches.* ICDM 2021; extended in ACM TIST 2022.
