"""Checks on the cleaned tables for BOTH datasets, and on the separation between them.
Run after clean_shuttleset.py and build_player_attributes.py:  python cleaning/test_clean.py
Each check recomputes something directly from the raw files and compares it with data/clean*/ ."""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean_shuttleset as cs  # noqa: E402  (only used for paths and the name key)

REPO = cs.REPO
DIR = {'ShuttleSet': cs.DATASETS['shuttleset']['out'], 'ShuttleSet22': cs.DATASETS['shuttleset22']['out']}
FAILED = []
NET_M = cs.NET_M


def check(name, cond, detail=''):
    print(('PASS  ' if cond else 'FAIL  ') + name + (f'  [{detail}]' if detail else ''))
    if not cond:
        FAILED.append(name)


def raw_strokes(set_dir, drop_videos=()):
    frames = []
    for f in sorted(glob.glob(str(set_dir / '*' / 'set*.csv'))):
        v = os.path.basename(os.path.dirname(f))
        if v in drop_videos:
            continue
        d = pd.read_csv(f, encoding='utf-8-sig')
        d['video'] = v
        d['set'] = int(os.path.basename(f)[3])
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def dataset_checks(label, set_dir, drop_videos, n_matches, n_dropped_rows):
    print(f'\n--- {label} ---')
    out = DIR[label]
    raw = raw_strokes(set_dir, drop_videos)
    st = pd.read_csv(out / 'strokes_clean.csv')
    sm = pd.read_csv(out / 'smashes_clean.csv')
    ma = pd.read_csv(out / 'matches_clean.csv')
    pl = pd.read_csv(out / 'players.csv')
    p = f'{label}: '

    check(p + f'{n_matches} matches', st.video.nunique() == n_matches and len(ma) == n_matches)
    check(p + 'every row carries source and match_uid of this dataset',
          (st.source == label).all() and (sm.source == label).all() and (ma.source == label).all() and
          st.match_uid.str.startswith('SS22-' if label == 'ShuttleSet22' else 'SS-').all())
    check(p + 'only duplicate placeholder rows were dropped', len(raw) - len(st) == n_dropped_rows, f'{len(raw)} raw -> {len(st)}')
    check(p + 'smash count matches raw 殺球', (st.type == 'smash').sum() == (raw.type == '殺球').sum(), str((st.type == 'smash').sum()))
    check(p + 'wrist smash count matches raw 點扣', (st.type == 'wrist smash').sum() == (raw.type == '點扣').sum(), str((st.type == 'wrist smash').sum()))
    check(p + 'smash table = smash + wrist smash rows', len(sm) == ((st.type == 'smash') | (st.type == 'wrist smash')).sum())
    check(p + 'every stroke has an English type', st.type.notna().all() and not st.type.str.contains('[一-鿿]').any())
    check(p + 'unique (video, set, rally, ball_round)', not st.duplicated(['video', 'set', 'rally', 'ball_round']).any())
    check(p + 'each match has a single gender and players of one match share it',
          st.groupby('video').gender.nunique().max() == 1)
    check(p + 'player table gender matches stroke gender',
          (st.drop_duplicates('player_name').set_index('player_name').gender.reindex(pl.player_name).values == pl.gender.values).all())

    # outcomes against raw getpoint_player
    end = raw[raw.getpoint_player.notna()].copy()
    end['rally_id'] = end.video + '|' + end['set'].astype(str) + '|' + end.rally.astype(str)
    rw = end.drop_duplicates('rally_id').set_index('rally_id').getpoint_player
    rid = st.drop_duplicates('rally_id').set_index('rally_id')
    check(p + 'rally winner matches raw getpoint_player',
          (rid.rally_winner.dropna() == rw.reindex(rid.rally_winner.dropna().index)).all() and rid.rally_winner.notna().sum() == len(rw))
    fin = raw.sort_values(['video', 'set', 'rally']).groupby('video').tail(1)
    check(p + 'A is the match winner (A leads at the end of every match)', (fin.roundscore_A > fin.roundscore_B).all(), f'{len(fin)} matches')

    s1 = sm.dropna(subset=['S1_rally_won'])
    check(p + 'S1 equals (shooter == rally winner)', (s1.S1_rally_won == (s1.player == s1.rally_winner).astype(float)).all())
    check(p + 'S1 empty exactly when winner unknown or ending inconsistent',
          sm.S1_rally_won.isna().sum() == (sm.rally_winner.isna() | (sm.outcome_consistent != 1)).sum())
    check(p + 'returned smashes have a later real stroke', (sm[sm.smash_result == 'returned'].later_real_strokes > 0).all())
    check(p + 'won_directly / shooter_error / ambiguous / unknown are last real strokes',
          (sm[sm.smash_result.isin(['won_directly', 'shooter_error', 'ambiguous', 'unknown'])].later_real_strokes == 0).all())
    check(p + 'won_directly implies the shooter won the rally', (sm[sm.smash_result == 'won_directly'].S1_rally_won == 1).all())
    check(p + 'shooter_error implies the shooter lost the rally', (sm[sm.smash_result == 'shooter_error'].S1_rally_won == 0).all())
    check(p + 'S2 is 1 only for won_directly, 0 only for returned',
          (sm.S2_won_directly.dropna().isin([0, 1])).all() and ((sm.S2_won_directly == 1) == (sm.smash_result == 'won_directly')).all() and
          ((sm.S2_won_directly == 0) == (sm.smash_result == 'returned')).all())
    check(p + 'every smash has exactly one known result category',
          sm.smash_result.isin(['returned', 'won_directly', 'shooter_error', 'ambiguous', 'unknown']).all())

    # pre-rally score never contains the rally's own point
    post = raw.assign(rally_id=raw.video + '|' + raw['set'].astype(str) + '|' + raw.rally.astype(str)).drop_duplicates('rally_id') \
        .set_index('rally_id')[['roundscore_A', 'roundscore_B']]
    r = rid.join(post)
    known = r.rally_winner.notna()
    a_won = (r.rally_winner == 'A')[known]
    check(p + "score_before = post score minus the winner's point", (r[known].score_A_before == r[known].roundscore_A - a_won).all())
    check(p + 'score_before never above the post score', ((r.score_A_before <= r.roundscore_A) & (r.score_B_before <= r.roundscore_B)).all())
    firsts = rid.sort_values('rally').groupby(['video', 'set']).head(1)
    check(p + 'score before the first rally of each set is 0-0 (or 1 point if the first rally is missing)',
          (firsts.score_A_before + firsts.score_B_before <= 1).all(), f'{len(firsts)} sets')

    # geometry
    ok = sm.dropna(subset=['shooter_y_m', 'opp_y_m'])
    check(p + 'after rotation the shooter is on the top half', (ok.shooter_y_m < NET_M).all(), f'{len(ok)} strokes')
    good = ok[ok.geometry_ok]
    check(p + 'after rotation the opponent is on the other half wherever geometry_ok', (good.opp_y_m > NET_M).all(), f'{len(good)} strokes')
    flagged = sm[~sm.geometry_ok]
    check(p + 'flagged strokes are whole rallies (no rally is half flagged)',
          (st.groupby('rally_id').geometry_ok.nunique() == 1).all())
    check(p + 'flagged smash-type strokes never count as having reach inputs', not flagged.has_reach_inputs.any(), f'{len(flagged)} flagged')
    check(p + 'player and opponent positions are within the court area + 3 m', (~sm.coord_suspect).all())
    check(p + 'court coordinates in metres are inside a sane range',
          sm[['shooter_x_m', 'opp_x_m']].stack().between(-1.5, 7.7).all() and sm[['shooter_y_m', 'opp_y_m']].stack().between(-1.5, 15).all())
    d = st.dropna(subset=['shooter_x_m', 'opp_x_m'])
    inside = (d.shooter_x_m.between(-0.3, 6.4) & d.shooter_y_m.between(-0.3, 13.7) & d.opp_x_m.between(-0.3, 6.4) & d.opp_y_m.between(-0.3, 13.7))
    per_match = inside.groupby(d.video).mean()
    check(p + 'at least 95% of player positions are inside the court in every match', (per_match > 0.95).all(),
          f'worst match {per_match.min():.3f}')
    check(p + 'flight time for returned smashes is plausible (0.1 to 3 s)', sm.t_next_for_smash_s.dropna().between(0.1, 3).all(),
          f'n={sm.t_next_for_smash_s.notna().sum()}')
    check(p + 'court width is about 300 units in every match', (ma.court_x_right - ma.court_x_left).sub(300).abs().max() < 6)
    return st, sm, ma, pl


# ---------------------------------------------------------------------------------------------- ShuttleSet (training)
st1, sm1, ma1, pl1 = dataset_checks('ShuttleSet', cs.SET1, (), 44, 2)
check('ShuttleSet: 27 players (16 men, 11 women)', len(pl1) == 27 and (pl1.gender == 'men').sum() == 16 and (pl1.gender == 'women').sum() == 11)
check('ShuttleSet: no stroke is flagged for bad geometry', st1.geometry_ok.all())

eda = pd.read_csv(REPO / 'EDA' / 'smash_reach_A_sets1-2.csv')
sub = sm1[sm1.video.str.contains('Axelsen') & sm1.video.str.contains('Ng_Ka_Long') & sm1.video.str.contains('Finals') &
          (sm1.player == 'A') & sm1.set.isin([1, 2])]
check('ShuttleSet: EDA match found once', sub.video.nunique() == 1, str(sub.video.unique().tolist()))
mm = eda.merge(sub, on=['set', 'rally', 'ball_round'], how='left')
flight = np.hypot(mm.land_x_m - mm.hit_x_m, mm.land_y_m - mm.hit_y_m)
check('ShuttleSet: matches the hand-built EDA table (same strokes)', mm.smash_result.notna().all() and len(mm) == len(eda))
check('ShuttleSet: matches the EDA table: flight distance within 0.01 m', (flight - mm.flight_m).abs().max() < 0.01)
check('ShuttleSet: matches the EDA table: flight time within 0.01 s', (mm.t_obs_s - mm.t_next_for_smash_s).abs().max() < 0.01)

# ---------------------------------------------------------------------------------------------- ShuttleSet22 (test)
set22 = cs.find_set_dir('shuttleset22')
all22 = pd.read_csv(DIR['ShuttleSet22'] / 'matches_all_shuttleset22.csv')
dup_videos = set(all22[all22.is_duplicate_of_shuttleset].video)
st22, sm22, ma22, pl22 = dataset_checks('ShuttleSet22', set22, dup_videos, 50, 0)
check('ShuttleSet22: 35 players (13 women, 22 men)', len(pl22) == 35 and (pl22.gender == 'women').sum() == 13 and (pl22.gender == 'men').sum() == 22)
check('ShuttleSet22: 8 of 58 matches are duplicates of ShuttleSet and every one is identical row for row',
      len(all22) == 58 and len(dup_videos) == 8 and all22[all22.is_duplicate_of_shuttleset].stroke_level_identical.all())
check('ShuttleSet22: duplicate matches are absent from every test table',
      not (set(st22.video) & dup_videos) and not (set(sm22.video) & dup_videos) and not (set(ma22.video) & dup_videos))
check('ShuttleSet22: test matches = the 50 not flagged as duplicates', set(ma22.video) == set(all22[~all22.is_duplicate_of_shuttleset].video))
bad22 = st22[~st22.geometry_ok]
check('ShuttleSet22: only the Chen v He Indonesia Masters rallies are flagged for bad geometry',
      bad22.video.nunique() == 1 and 'CHEN_Yu_Fei_HE_Bing_Jiao' in bad22.video.iloc[0], f'{bad22.rally_id.nunique()} rallies')

# ---------------------------------------------------------------------------------------------- the two datasets stay separate
print('\n--- separation ---')
check('no match appears in both cleaned datasets (by folder name)', not (set(st1.video) & set(st22.video)))
check('no match_uid is shared', not (set(st1.match_uid) & set(st22.match_uid)))
key = lambda m: set(zip(m.date, m.winner_name.map(cs.norm_name), m.loser_name.map(cs.norm_name)))
check('no match appears in both datasets (by date, winner and loser)', not (key(ma1) & key(ma22)))
check('no rally_id is shared', not (set(st1.rally_id) & set(st22.rally_id)))
mixed = [f for f in glob.glob(str(REPO / 'data' / '**' / '*.csv'), recursive=True)
         if 'source' in pd.read_csv(f, nrows=0).columns and pd.read_csv(f, usecols=['source']).source.nunique() > 1]
check('no output file mixes the two datasets', not mixed, str(mixed))
cw = pd.read_csv(DIR['ShuttleSet22'] / 'player_crosswalk.csv')
names1 = {cs.norm_name(n) for n in pl1.player_name}
check('crosswalk: seen_in_shuttleset agrees with the name match', all((cs.norm_name(n) in names1) == s for n, s in zip(cw.player_name_shuttleset22, cw.seen_in_shuttleset)))
check('crosswalk: 19 of 35 test players are known, 16 are not', cw.seen_in_shuttleset.sum() == 19 and (~cw.seen_in_shuttleset).sum() == 16)
check('crosswalk contains names only (no outcome or position columns)', not {'S1_rally_won', 'hit_x_m', 'smash_result'} & set(cw.columns))

# ---------------------------------------------------------------------------------------------- player attributes
print('\n--- player attributes ---')
ext = REPO / 'data' / 'external'
pa1 = pd.read_csv(ext / 'player_attributes.csv')
pa22 = pd.read_csv(ext / 'player_attributes_shuttleset22.csv')
check('attributes (ShuttleSet): exactly the 27 players', set(pa1.player_name) == set(pl1.player_name) and len(pa1) == 27)
check('attributes (ShuttleSet22): exactly the 35 test-set players', set(pa22.player_name) == set(pl22.player_name) and len(pa22) == 35)
check('attributes: labelled with their dataset', (pa1.source == 'ShuttleSet').all() and (pa22.source == 'ShuttleSet22').all())
check('attributes (ShuttleSet): every player has height 150 to 200 cm and a handedness',
      pa1.height_cm.between(150, 200).all() and pa1.handedness.isin(['left', 'right']).all())
has_h = pa22.height_cm.notna()
check('attributes (ShuttleSet22): heights present are 150 to 200 cm; missing ones are only Wang Zhi Yi',
      pa22[has_h].height_cm.between(150, 200).all() and list(pa22[~has_h].player_name) == ['WANG Zhi Yi'], str(list(pa22[~has_h].player_name)))
check('attributes (ShuttleSet22): handedness present is left/right; missing only for Wang Zhi Yi',
      pa22.handedness.dropna().isin(['left', 'right']).all() and list(pa22[pa22.handedness.isna()].player_name) == ['WANG Zhi Yi'])
shared = pa22[pa22.seen_in_shuttleset].merge(cw[['player_name_shuttleset22', 'player_name_shuttleset']], left_on='player_name',
                                             right_on='player_name_shuttleset22').merge(pa1, left_on='player_name_shuttleset',
                                                                                         right_on='player_name', suffixes=('_22', '_1'))
cols = ['height_cm', 'handedness', 'birth_year', 'dob_wikipedia']
check('attributes: the 19 shared players have identical values in both tables', len(shared) == 19 and all(
    ((shared[c + '_22'] == shared[c + '_1']) | (shared[c + '_22'].isna() & shared[c + '_1'].isna())).all() for c in cols))
for lab, pa in (('ShuttleSet', pa1), ('ShuttleSet22', pa22)):
    check(f'attributes ({lab}): birth years are plausible (1985 to 2005) and both sources agree on the year',
          pa.birth_year.between(1985, 2005).all() and (pa.dob_other.isna() | (pa.dob_wikipedia.str[:4] == pa.dob_other.str[:4])).all())
    check(f'attributes ({lab}): women are shorter than men on average (sanity check on the height column)',
          pa[pa.gender == 'women'].height_cm.mean() < pa[pa.gender == 'men'].height_cm.mean())
    check(f'attributes ({lab}): no attribute column is derived from match results', not {'win_pct', 'win_rate', 'wins', 'ranking'} & set(pa.columns))

print(f'\n{len(FAILED)} check(s) failed' if FAILED else '\nall checks passed')
sys.exit(1 if FAILED else 0)
