"""Checks on the EDA outputs.  Run after eda_all.py:  python EDA/test_eda.py
1. The tables add up against the cleaned data.  2. The test-set EDA is outcome-blind."""
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
A, T = REPO / 'EDA' / 'all_matches', REPO / 'EDA' / 'test_set'
FAILED = []


def check(name, cond, detail=''):
    print(('PASS  ' if cond else 'FAIL  ') + name + (f'  [{detail}]' if detail else ''))
    if not cond:
        FAILED.append(name)


sm = pd.read_csv(REPO / 'data' / 'clean' / 'smashes_clean.csv')
sm22 = pd.read_csv(REPO / 'data' / 'clean_shuttleset22' / 'smashes_clean.csv')
s = sm[sm.smash_kind == 'smash']

pp = pd.read_csv(A / 'counts_per_player.csv')
check('counts per player: smashes and wrist smashes add up', pp.smashes.sum() == (sm.smash_kind == 'smash').sum() and pp.wrist_smashes.sum() == (sm.smash_kind == 'wrist smash').sum(), f'{pp.smashes.sum()} / {pp.wrist_smashes.sum()}')
zt = pd.read_csv(A / 'counts_per_zone.csv')
check('counts per zone add up to the smashes with a landing zone', zt.smashes.sum() == s.landing_area.notna().sum(), str(zt.smashes.sum()))
check('counts per zone: labelled counts add up to the smashes with an S2 label and a zone', zt.smashes_with_S2.sum() == s.dropna(subset=['landing_area', 'S2_won_directly']).shape[0])
pz = pd.read_csv(A / 'counts_player_zone.csv')
check('player x zone counts add up to the labelled smashes with a zone', pz.smashes_with_S2.sum() == zt.smashes_with_S2.sum())
rz = pd.read_csv(A / 'outcome_rates_by_zone.csv')
check('outcome rates by zone: S2 counts add up and every rate lies inside its interval',
      rz.n_S2.sum() == zt.smashes_with_S2.sum() and ((rz.S2_rate.isna()) | ((rz.S2_lo <= rz.S2_rate) & (rz.S2_rate <= rz.S2_hi))).all())
check('outcome rates by zone: the three mid-court zones hold 89% of smashes', abs(rz.head(3).smashes.sum() / rz.smashes.sum() - .89) < .01)
mh = pd.read_csv(A / 'match_heterogeneity.csv')
check('match heterogeneity has one row per match', len(mh) == 44 and mh.video.is_unique)
check('match heterogeneity: smash counts add up', mh.smashes.sum() == len(s))
check('wrist-smash share is between 0 and 1', mh.wrist_share_of_smash_type.between(0, 1).all())
check('distribution table covers smash and wrist smash for six features', len(pd.read_csv(A / 'distributions_reach_features.csv')) == 12)
pb = pd.read_csv(A / 'selection_smash_propensity.csv')
check('selection: shares are between 0 and 1 and intervals contain them', pb.share.between(0, 1).all() and ((pb.lo <= pb.share) & (pb.share <= pb.hi)).all())

# ---------------------------------------------------------------------------------------------- the test-set EDA is outcome-blind
outcome_cols = {'S1_rally_won', 'S2_won_directly', 'smash_result', 'rally_winner', 'end_reason', 'match_winner_name', 't_next_for_smash_s',
                'later_real_strokes', 'outcome_consistent', 'S1_rate', 'S2_rate', 'shooter_error_rate', 'win_rate'}
bad = []
for f in sorted(T.glob('*.csv')):
    cols = set(pd.read_csv(f, nrows=0).columns)
    if cols & outcome_cols or any(re.search(r'(S1|S2|win|won|error)', c, re.I) for c in cols):
        bad.append(f.name)
check('no test-set EDA table has an outcome column', not bad, str(bad))
tc = pd.read_csv(T / 'test_counts_per_player.csv')
check('test counts per player add up to the test smashes', tc.smashes.sum() == (sm22.smash_kind == 'smash').sum() and tc.wrist_smashes.sum() == (sm22.smash_kind == 'wrist smash').sum())
check('test set has 50 matches in the quality table', len(pd.read_csv(T / 'test_match_quality.csv')) == 50)
shift = pd.read_csv(T / 'shift_check_inputs.csv')
check('shift check uses input features only', set(shift.feature) <= {'d_m', 't_hat_s', 'flight_m', 'hit_to_opp_m', 'opp_home_m'}, str(list(shift.feature)))
report = (A / 'eda_report.md').read_text(encoding='utf-8')
test_lines = [l for l in report.splitlines() if l.startswith('- TEST') or l.startswith('- SHIFT CHECK')]
words = re.compile(r'(direct-win|rally-won|S1|S2|won directly|shooter error|win rate)', re.I)
check('test-set lines in the report contain no outcome wording', test_lines and not any(words.search(l) for l in test_lines), f'{len(test_lines)} lines checked')
src = (REPO / 'EDA' / 'eda_all.py').read_text(encoding='utf-8')
check('the test-set loader drops outcome columns before any analysis', 'load(TEST_DIR, outcome_blind=True)' in src and 'assert not (set(OUTCOME_COLS) & set(sm.columns))' in src)

print(f'\n{len(FAILED)} check(s) failed' if FAILED else '\nall checks passed')
sys.exit(1 if FAILED else 0)
