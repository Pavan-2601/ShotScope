"""Clean ShuttleSet (training data) and ShuttleSet22 (held-out test data) into analysis-ready tables, kept SEPARATE.

Run from anywhere:  python cleaning/clean_shuttleset.py [shuttleset | shuttleset22 | all]   (default: all)
  shuttleset   reads set/                       writes data/clean/
  shuttleset22 reads ShuttleSet22/set/          writes data/clean_shuttleset22/
Each folder gets strokes_clean, smashes_clean, matches_clean, players and cleaning_report.md. No file ever mixes the two
datasets: every row carries `source` and `match_uid`. For ShuttleSet22 the 8 matches that are also in ShuttleSet are
confirmed at stroke level and then excluded, so the test set contains no training match. The only link between the
datasets is a player-name crosswalk (data/clean_shuttleset22/player_crosswalk.csv), used to tell known from unseen players.
ShuttleSet22 is untracked, so it is looked up in ShuttleSet22/ or, from a git worktree, in the main checkout.
Column meanings are in cleaning/DATA_DICTIONARY.md.

Findings about the raw files that this script relies on (checked on the data, see the report):
  * roundscore_A / roundscore_B are the score AFTER the rally, so they leak the rally result.
    Pre-rally scores are rebuilt here (score_*_before).
  * `server` is a stroke-status code: 1 = serve, 2 = rally stroke, 3 = rally-ending stroke.
  * `time` (hh:mm:ss) disagrees with frame_num/30 for about half the strokes; frame_num is the clock used here.
  * Type '未知球種' (unknown) rows are placeholders, mostly an unannotated serve at the start of a rally.
  * Court coordinates come from the per-match homography (units of ~1/49.2 m; net at y = 480).
"""
import glob
import json
import os
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SET1 = REPO / 'set'
DATASETS = {
    'shuttleset': dict(label='ShuttleSet', short='SS', out=REPO / 'data' / 'clean'),
    'shuttleset22': dict(label='ShuttleSet22', short='SS22', out=REPO / 'data' / 'clean_shuttleset22'),
}

FPS = 30.0
UNITS_PER_M = 49.2          # court units per metre (same constant as EDA/eda.py)
NET_Y = 480.0               # net position in court units
Y_BACK = 150.0              # far baseline (top) in court units
Y_SUM = 960.0               # y -> 960 - y rotates the court by 180 degrees
NET_M = (NET_Y - Y_BACK) / UNITS_PER_M

TYPE_EN = {
    '放小球': 'net shot', '擋小球': 'return net', '殺球': 'smash', '點扣': 'wrist smash', '挑球': 'lob',
    '防守回挑': 'defensive return lob', '長球': 'clear', '平球': 'drive', '小平球': 'driven flight',
    '後場抽平球': 'back-court drive', '切球': 'drop', '過渡切球': 'passive drop', '過度切球': 'passive drop',
    '推球': 'push', '撲球': 'rush', '防守回抽': 'defensive return drive', '勾球': 'cross-court net shot',
    '發短球': 'short service', '發長球': 'long service', '未知球種': 'unknown',
}
UNKNOWN_ZH = '未知球種'
# lose_reason on the rally-ending stroke. DIRECT: the stroke was the winner's and the shuttle landed or was misjudged.
# ERROR: the stroke was the loser's own error.
DIRECT_ZH = {'對手落地致勝': 'landed', '對手落地判斷失誤': 'opponent misjudged landing', '落點判斷失誤': 'opponent misjudged landing'}
ERROR_ZH = {'出界': 'out', '掛網': 'net', '未過網': 'not over net', '犯規': 'fault'}
REASON_EN = {**DIRECT_ZH, **ERROR_ZH}

WOMEN = {'an se young', 'busanan ongbamrungphan', 'carolina marin', 'evgeniya kosetskaya', 'michelle li',
         'pusarla v sindhu', 'pornpawee chochuwong', 'ratchanok intanon', 'mia blichfeldt',
         'supanida katethong', 'neslihan yigit'}


def norm(name):
    return ' '.join(str(name).strip().lower().split())


def norm_name(name):
    """Key for matching the same person across datasets: ASCII, lower case, hyphens and dots as spaces."""
    t = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode()
    return ' '.join(t.lower().replace('-', ' ').replace('.', ' ').split())


def find_set_dir(name):
    if name == 'shuttleset':
        return SET1
    for cand in (REPO / 'ShuttleSet22' / 'set', REPO.parents[2] / 'ShuttleSet22' / 'set'):
        if (cand / 'match.csv').exists():
            return cand
    raise FileNotFoundError('ShuttleSet22/set not found in the repo folder or the main checkout')


def infer_gender(m, seeds):
    """Gender per norm_name key. `seeds` maps known keys to 'men'/'women'; opponents in a match share a gender, so the
    rest is filled in by walking the opponent graph. Raises if a component has no seed or two seeds disagree."""
    adj = defaultdict(set)
    for w, l in zip(m.winner_name, m.loser_name):
        a, b = norm_name(w), norm_name(l)
        adj[a].add(b); adj[b].add(a)
    out, unresolved = {}, []
    for start in adj:
        if start in out:
            continue
        comp, stack = [], [start]
        seen = {start}
        while stack:
            k = stack.pop(); comp.append(k)
            for n in adj[k]:
                if n not in seen:
                    seen.add(n); stack.append(n)
        genders = {seeds[k] for k in comp if k in seeds}
        if len(genders) > 1:
            raise AssertionError(f'players in one opponent group have different known genders: {sorted(comp)}')
        if not genders:
            unresolved.append(sorted(comp)); continue
        gender = next(iter(genders))
        for k in comp:
            out[k] = gender
    if unresolved:
        raise AssertionError(f'no known gender for these players (add a seed): {unresolved}')
    return out


def shuttleset_gender(match1):
    keys = {norm_name(n) for n in pd.concat([match1.winner, match1.loser]).str.strip()}
    return {k: ('women' if k in WOMEN else 'men') for k in keys}


def load_raw(set_dir):
    match = pd.read_csv(set_dir / 'match.csv')
    hom = pd.read_csv(set_dir / 'homography.csv')
    assert (match.video.values == hom.video.values).all(), 'match.csv and homography.csv are in different order'
    frames = []
    for video in match.video:
        for f in sorted(glob.glob(str(set_dir / video / 'set*.csv'))):
            d = pd.read_csv(f, encoding='utf-8-sig')
            d['video'] = video
            d['set'] = int(os.path.basename(f)[3])
            frames.append(d)
    return match, hom, pd.concat(frames, ignore_index=True)


def court_transform(df, hom, xcol, ycol):
    """Camera pixels -> court units, using each match's own homography."""
    cx = pd.Series(np.nan, index=df.index)
    cy = pd.Series(np.nan, index=df.index)
    ok = df[xcol].notna() & df[ycol].notna()
    for video, g in df[ok].groupby('video'):
        H = np.array(json.loads(hom.loc[video, 'homography_matrix']))
        P = np.vstack([g[xcol].values, g[ycol].values, np.ones(len(g))])
        Q = H @ P
        cx.loc[g.index] = Q[0] / Q[2]
        cy.loc[g.index] = Q[1] / Q[2]
    return cx, cy


def as_text(df):
    """Same value, same text: numbers become floats first so 3 and 3.0 compare equal; missing values become ''."""
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].astype(float) if pd.api.types.is_numeric_dtype(out[c]) else out[c]
    return out.fillna('').astype(str)


def drop_shuttleset_duplicates(match, hom, raw, log):
    """ShuttleSet22 repeats 8 ShuttleSet matches. Identify them two ways (same folder name; same date, winner and loser),
    require the two to agree, compare their strokes row by row with the ShuttleSet copy, then remove them."""
    m1, _, raw1 = load_raw(SET1)
    k = lambda df: list(zip(df.year, df.month, df.day, df.winner.map(norm_name), df.loser.map(norm_name)))
    by_key = pd.Series([x in set(k(m1)) for x in k(match)], index=match.index)
    by_folder = match.video.isin(set(m1.video))
    assert (by_key == by_folder).all(), 'duplicate detection by folder name and by date+players disagree'
    flags = match[['id', 'video', 'tournament', 'round', 'year', 'month', 'day', 'winner', 'loser']].copy()
    flags['is_duplicate_of_shuttleset'] = by_folder.values
    identical, n22, n1 = [], [], []
    for v in match.video:
        if not flags.loc[flags.video == v, 'is_duplicate_of_shuttleset'].iloc[0]:
            identical.append(None); n22.append(None); n1.append(None); continue
        cols = [c for c in raw1.columns if c != 'video']
        a = as_text(raw1[raw1.video == v][cols]).sort_values(cols).reset_index(drop=True)
        b = as_text(raw[raw.video == v][cols]).sort_values(cols).reset_index(drop=True)
        identical.append(bool(a.equals(b))); n22.append(len(b)); n1.append(len(a))
    flags['stroke_level_identical'] = identical
    flags['rows_in_shuttleset22'] = n22
    flags['rows_in_shuttleset'] = n1
    flags['in_test_set'] = ~flags.is_duplicate_of_shuttleset
    dups = flags[flags.is_duplicate_of_shuttleset]
    log(f'Duplicates with ShuttleSet: {len(dups)} of {len(match)} matches share a folder name AND the same date, winner and loser; '
        f'{int(dups.stroke_level_identical.sum())} of {len(dups)} are identical row for row. They are excluded from the test set.')
    keep = ~by_folder.values
    drop_videos = set(match.video[~keep])
    return match[keep].reset_index(drop=True), hom[~hom.video.isin(drop_videos)].reset_index(drop=True), \
        raw[~raw.video.isin(drop_videos)].reset_index(drop=True), flags


def run(dataset):
    cfg = DATASETS[dataset]
    label, uid_prefix, OUT = cfg['label'], cfg['short'], cfg['out']
    report = []
    log = report.append
    match, hom_df, raw = load_raw(find_set_dir(dataset))
    log(f'Raw {label}: {len(raw)} stroke rows, {raw.video.nunique()} matches, {match.shape[0]} rows in match.csv.')
    flags = None
    if dataset == 'shuttleset22':
        match, hom_df, raw, flags = drop_shuttleset_duplicates(match, hom_df, raw, log)
        log(f'After removing them: {len(raw)} stroke rows, {raw.video.nunique()} matches.')
    hom = hom_df.set_index('video')

    # ---- 1. duplicates --------------------------------------------------------------------------------------------
    n0 = len(raw)
    d = raw.drop_duplicates().copy()
    n_exact = n0 - len(d)
    d = d.sort_values(['video', 'set', 'rally', 'ball_round'], kind='stable').reset_index(drop=True)
    key = ['video', 'set', 'rally', 'ball_round']
    n_key = int(d.duplicated(key).sum())
    dup_types = d[d.duplicated(key, keep=False)].type.unique().tolist()
    d = d.drop_duplicates(key, keep='first').reset_index(drop=True)
    log(f'Dropped {n_exact} exact duplicate row(s) and {n_key} further row(s) repeating (video, set, rally, ball_round); '
        f'the latter were type(s) {dup_types}. Kept {len(d)} rows.')

    # ---- 2. match-level table: names, gender, court frame --------------------------------------------------------
    m = match.merge(hom_df[['video', 'upleft_x', 'upright_x', 'downleft_x', 'downright_x',
                            'upleft_y', 'upright_y', 'downleft_y', 'downright_y', 'homography_matrix']], on='video')
    m['winner_name'] = m.winner.str.strip()
    m['loser_name'] = m.loser.str.strip()
    # The four annotated court corners are named upleft/upright/downleft/downright, but in some matches the two bottom
    # corners are listed in swapped order (downleft_x > downright_x). The stored homography is still right, so the
    # corners are ordered by their pixel position here, never by their column names.
    xl, xr, swapped = [], [], []
    for r in m.itertuples():
        H = np.array(json.loads(r.homography_matrix))
        pix = np.array([(r.upleft_x, r.upleft_y), (r.upright_x, r.upright_y),
                        (r.downleft_x, r.downleft_y), (r.downright_x, r.downright_y)], float)
        q = np.c_[pix, np.ones(4)] @ H.T
        q = q[:, :2] / q[:, 2:3]
        order = np.argsort(pix[:, 1])
        tl, tr = sorted(order[:2], key=lambda i: pix[i, 0])
        bl, br = sorted(order[2:], key=lambda i: pix[i, 0])
        xl.append((q[tl, 0] + q[bl, 0]) / 2)
        xr.append((q[tr, 0] + q[br, 0]) / 2)
        swapped.append((tl, tr, bl, br) != (0, 1, 2, 3))
    m['court_x_left'], m['court_x_right'] = xl, xr
    m['corner_labels_swapped'] = swapped
    assert (np.abs(m.court_x_right - m.court_x_left - 300) < 6).all(), 'court width is not about 300 units in every match'
    if any(swapped):
        log(f'Court corners: {sum(swapped)} of {len(m)} matches list the bottom corners in swapped order; '
            f'their homographies map the corners correctly, so only the column names are off (handled by ordering corners by position).')
    m1_for_seeds = pd.read_csv(SET1 / 'match.csv')
    seeds = shuttleset_gender(m1_for_seeds)
    gmap = infer_gender(m, seeds)
    if dataset == 'shuttleset':
        assert gmap == seeds, 'gender walk disagrees with the known ShuttleSet list'
    m['gender'] = [gmap[norm_name(w)] for w in m.winner_name]
    bad_gender = m[[gmap[norm_name(w)] != gmap[norm_name(l)] for w, l in zip(m.winner_name, m.loser_name)]]
    assert bad_gender.empty, f'mixed-gender matches: {bad_gender.video.tolist()}'
    new_keys = sorted(k for k in gmap if k not in seeds)
    m['date'] = pd.to_datetime(dict(year=m.year, month=m.month, day=m.day)).dt.strftime('%Y-%m-%d')
    names = pd.concat([m.winner_name, m.loser_name])
    log(f'Players: {names.map(norm_name).nunique()} distinct (names are already consistent); '
        f'women {sum(1 for k in {norm_name(n) for n in names} if gmap[k] == "women")}, '
        f'men {sum(1 for k in {norm_name(n) for n in names} if gmap[k] == "men")}'
        + (f'; gender of {len(new_keys)} players not in ShuttleSet taken from their opponents: ' + ', '.join(f'{k} ({gmap[k]})' for k in new_keys) if new_keys else '') + '.')
    mi = m.set_index('video')

    # ---- 3. stroke-level basics ----------------------------------------------------------------------------------
    d['match_id'] = d.video.map(mi.id)
    d['source'] = label
    d['match_uid'] = uid_prefix + '-' + d.match_id.astype(str)
    for c in ('tournament', 'round', 'date', 'gender'):
        d[c] = d.video.map(mi[c])
    d['rally_id'] = d.video + '|' + d['set'].astype(str) + '|' + d.rally.astype(str)
    d['ball_round'] = d.ball_round.astype(int)
    win_n, los_n = d.video.map(mi.winner_name), d.video.map(mi.loser_name)
    d['player_name'] = np.where(d.player == 'A', win_n, los_n)
    d['opponent_name'] = np.where(d.player == 'A', los_n, win_n)
    d['match_winner_name'] = win_n
    unknown_types = sorted(set(d.type) - set(TYPE_EN))
    assert not unknown_types, f'untranslated shot types: {unknown_types}'
    d['is_placeholder'] = d.type == UNKNOWN_ZH
    d['type_zh'] = d.type
    d['type'] = d.type.map(TYPE_EN)
    d['is_smash'] = d.type == 'smash'
    d['is_wrist_smash'] = d.type == 'wrist smash'
    d['stroke_status'] = d.server.map({1: 'serve', 2: 'rally', 3: 'last'})
    d['frame_sec'] = d.frame_num / FPS
    d['aroundhead'] = d.aroundhead.fillna(0).astype(int)
    d['backhand'] = d.backhand.fillna(0).astype(int)

    # ---- 4. rally-level outcome and pre-rally score ---------------------------------------------------------------
    g = d.groupby('rally_id', sort=False)
    d['rally_len'] = g.ball_round.transform('size')
    d['rally_real_len'] = g.is_placeholder.transform(lambda s: (~s).sum())
    d['rally_has_placeholder'] = g.is_placeholder.transform('any')
    end = d[d.getpoint_player.notna()].drop_duplicates('rally_id', keep='last').set_index('rally_id')
    d['rally_winner'] = d.rally_id.map(end.getpoint_player)
    d['end_reason_zh'] = d.rally_id.map(end.lose_reason)
    d['end_stroke_player'] = d.rally_id.map(end.player)
    d['end_stroke_flawed'] = d.rally_id.map(end.flaw).fillna(0).astype(int)
    d['end_ball_round'] = d.rally_id.map(end.ball_round)
    after_end = d[d.end_ball_round.notna() & (d.ball_round > d.end_ball_round)]
    log(f'Rallies: {d.rally_id.nunique()}; with a recorded winner {end.shape[0]}; '
        f'without one {d.rally_id.nunique() - end.shape[0]} (annotation flaw, outcome unknown); '
        f'strokes recorded after the rally-ending stroke: {len(after_end)}.')

    # consistency of the ending: stroke owner vs winner vs reason
    owner_won = d.end_stroke_player == d.rally_winner
    cat = d.end_reason_zh.map(lambda z: 'direct' if z in DIRECT_ZH else 'error' if z in ERROR_ZH else None)
    d['end_category'] = cat
    consistent = ((cat == 'direct') & owner_won) | ((cat == 'error') & ~owner_won)
    d['outcome_consistent'] = np.where(d.rally_winner.notna(), consistent & (d.end_stroke_flawed == 0), np.nan)
    one = d.drop_duplicates('rally_id').set_index('rally_id')
    n_rw = one.rally_winner.notna().sum()
    n_inc = int(((one.rally_winner.notna()) & (one.outcome_consistent == 0)).sum())
    log(f'Of {n_rw} rallies with a winner, {n_inc} have an ending that contradicts the winner or is flagged flawed '
        f'(reason says one thing, getpoint_player another); S1/S2 are left empty for those.')

    # pre-rally score: roundscore_* is the score AFTER the rally
    first = one.copy()
    pa, pb = first.roundscore_A.copy(), first.roundscore_B.copy()
    wa, wb = (first.rally_winner == 'A'), (first.rally_winner == 'B')
    pre_a = pa - wa.astype(int)
    pre_b = pb - wb.astype(int)
    first['pre_a'], first['pre_b'] = pre_a, pre_b
    first = first.sort_values(['video', 'set', 'rally'])
    prev_a = first.groupby(['video', 'set']).roundscore_A.shift(1).fillna(0)
    prev_b = first.groupby(['video', 'set']).roundscore_B.shift(1).fillna(0)
    known = first.rally_winner.notna()
    agree = ((first.pre_a == prev_a) & (first.pre_b == prev_b))[known]
    log(f'Score check: roundscore_* is the post-rally score. Pre-rally score (post minus winner point) equals the previous '
        f'rally\'s post score in {agree.mean():.1%} of {int(known.sum())} rallies; the rest are rallies missing from the data.')
    d['score_A_before'] = d.rally_id.map(first.pre_a)
    d['score_B_before'] = d.rally_id.map(first.pre_b)
    # where the winner is unknown fall back to the previous rally's post score
    fb_a, fb_b = d.rally_id.map(prev_a), d.rally_id.map(prev_b)
    d['score_A_before'] = d.score_A_before.fillna(fb_a)
    d['score_B_before'] = d.score_B_before.fillna(fb_b)
    is_a = d.player == 'A'
    d['shooter_score_before'] = np.where(is_a, d.score_A_before, d.score_B_before)
    d['opponent_score_before'] = np.where(is_a, d.score_B_before, d.score_A_before)
    d['score_diff_before'] = d.shooter_score_before - d.opponent_score_before

    # ---- 5. coordinates --------------------------------------------------------------------------------------------
    pts = {'hit': ('hit_x', 'hit_y'), 'landing': ('landing_x', 'landing_y'),
           'player_location': ('player_location_x', 'player_location_y'),
           'opponent_location': ('opponent_location_x', 'opponent_location_y')}
    for name, (xc, yc) in pts.items():
        d[f'{name}_court_x'], d[f'{name}_court_y'] = court_transform(d, hom, xc, yc)
    ref_y = d.player_location_court_y.fillna(d.hit_court_y)
    d['rotated'] = np.where(ref_y.notna(), ref_y > NET_Y, np.nan)
    xl_s, xr_s = d.video.map(mi.court_x_left), d.video.map(mi.court_x_right)
    rot = d.rotated == 1
    short = {'hit': 'hit', 'landing': 'land', 'player_location': 'shooter', 'opponent_location': 'opp'}
    for name, s in short.items():
        x, y = d[f'{name}_court_x'], d[f'{name}_court_y']
        xc = np.where(rot, xl_s + xr_s - x, x)
        yc = np.where(rot, Y_SUM - y, y)
        d[f'{s}_x_m'] = np.where(x.notna() & d.rotated.notna(), (xc - xl_s) / UNITS_PER_M, np.nan)
        d[f'{s}_y_m'] = np.where(y.notna() & d.rotated.notna(), (yc - Y_BACK) / UNITS_PER_M, np.nan)
    # coordinate sanity
    lines = []
    for name, s in short.items():
        x, y = d[f'{name}_court_x'], d[f'{name}_court_y']
        have = x.notna()
        inside = (x >= xl_s - 1) & (x <= xr_s + 1) & (y >= Y_BACK - 1) & (y <= 810 + 1)
        far = (x < xl_s - 150) | (x > xr_s + 150) | (y < Y_BACK - 150) | (y > 810 + 150)
        lines.append(f'{name}: {int(have.sum())} points, {inside[have].mean():.1%} inside the court, '
                     f'{int(far[have].sum())} more than 3 m outside')
    log('Coordinates through the homography: ' + '; '.join(lines) + '.')
    d['coord_suspect'] = ((d.player_location_court_x < xl_s - 150) | (d.player_location_court_x > xr_s + 150) |
                          (d.opponent_location_court_x < xl_s - 150) | (d.opponent_location_court_x > xr_s + 150))
    # Rally-level geometry check: after rotation the shooter is on the top half, so the opponent must be on the bottom half.
    # If most strokes of a rally put the opponent on the shooter's half, the positions of the whole rally are unreliable
    # (an annotation or camera-view problem), so every stroke of that rally is flagged, not just the inconsistent ones.
    d['opp_same_half'] = (d.opp_y_m < NET_M) & d.shooter_y_m.notna()
    both_pos = d.shooter_y_m.notna() & d.opp_y_m.notna()
    share_bad = d[both_pos].groupby('rally_id').opp_same_half.mean()
    d['rally_bad_geometry'] = d.rally_id.map(share_bad).fillna(0) > 0.5
    d['geometry_ok'] = ~d.rally_bad_geometry & ~d.coord_suspect
    bad_r = share_bad[share_bad > 0.5]
    bad_matches = d[d.rally_id.isin(bad_r.index)].groupby('video').rally_id.nunique()
    log(f'Geometry check: {len(bad_r)} rallies ({int(d.rally_bad_geometry.sum())} strokes, '
        f'{int((d.rally_bad_geometry & (d.is_smash | d.is_wrist_smash)).sum())} smash-type) put the opponent on the shooter\'s half '
        f'for most strokes and are flagged geometry_ok = False' + (': ' + '; '.join(f'{v} ({n} rallies)' for v, n in bad_matches.items()) if len(bad_matches) else '') + '.')

    # ---- 6. timing ------------------------------------------------------------------------------------------------
    real = d[~d.is_placeholder]
    nxt_frame = real.groupby('rally_id').frame_sec.shift(-1)
    nxt_type = real.groupby('rally_id').type.shift(-1)
    d['next_stroke_type'] = nxt_type.reindex(d.index)
    d['t_next_s'] = (nxt_frame - real.frame_sec).reindex(d.index)
    bad_t = (d.t_next_s <= 0)
    d['t_next_valid'] = d.t_next_s.notna() & ~bad_t
    d.loc[bad_t, 't_next_s'] = np.nan
    log(f'Timing: frame_num/30 is the clock (the hh:mm:ss `time` column disagrees with it by more than 1.5 s on '
        f'{int(((pd.to_timedelta(d.time).dt.total_seconds() - d.frame_num / FPS).abs() > 1.5).sum())} strokes and is not used). '
        f'{int(bad_t.sum())} strokes have a non-positive gap to the next stroke and get t_next_s empty.')

    # ---- 7. smash labels ----------------------------------------------------------------------------------------------
    d['real'] = (~d.is_placeholder).astype(int)
    rev = d.iloc[::-1].groupby('rally_id').real.cumsum().iloc[::-1]
    d['later_real_strokes'] = rev - d.real
    d['smash_kind'] = np.select([d.is_smash, d.is_wrist_smash], ['smash', 'wrist smash'], default='')
    s1 = np.where(d.rally_winner.notna() & (d.outcome_consistent == 1), (d.rally_winner == d.player).astype(float), np.nan)
    d['S1_rally_won'] = s1
    last_real = d.later_real_strokes == 0
    shooter_won_end = d.rally_winner == d.player
    res = np.full(len(d), '', dtype=object)
    sm = d.is_smash | d.is_wrist_smash
    res[(sm & ~last_real).values] = 'returned'
    ending = (sm & last_real).values
    ok_end = ending & d.rally_winner.notna().values
    won_direct = ok_end & (d.end_category == 'direct').values & shooter_won_end.values & (d.outcome_consistent == 1).values
    sh_error = ok_end & (d.end_category == 'error').values & ~shooter_won_end.values & (d.outcome_consistent == 1).values
    res[ending] = 'unknown'
    res[ok_end] = 'ambiguous'
    res[won_direct] = 'won_directly'
    res[sh_error] = 'shooter_error'
    d['smash_result'] = res
    d['S2_won_directly'] = np.where(d.smash_result == 'won_directly', 1.0, np.where(d.smash_result == 'returned', 0.0, np.nan))
    d.loc[~sm, ['S1_rally_won', 'S2_won_directly']] = np.nan
    d['t_next_for_smash_s'] = np.where(sm & (d.smash_result == 'returned'), d.t_next_s, np.nan)
    need = ['hit_court_x', 'landing_court_x', 'player_location_court_x', 'opponent_location_court_x']
    d['has_reach_inputs'] = d[need].notna().all(axis=1) & d.geometry_ok

    # ---- 8. outputs --------------------------------------------------------------------------------------------------
    d['end_reason'] = d.end_reason_zh.map(REASON_EN)
    d['lose_reason'] = d.lose_reason.map(REASON_EN).where(d.lose_reason.isna() | d.lose_reason.isin(REASON_EN), d.lose_reason)
    cols = ['source', 'match_uid', 'match_id', 'video', 'tournament', 'round', 'date', 'gender', 'set', 'rally', 'rally_id', 'ball_round',
            'stroke_status', 'player', 'player_name', 'opponent_name', 'type', 'type_zh', 'is_placeholder', 'is_smash',
            'is_wrist_smash', 'smash_kind', 'aroundhead', 'backhand', 'hit_height', 'landing_height',
            'hit_area', 'landing_area', 'player_location_area', 'opponent_location_area',
            'frame_num', 'frame_sec', 't_next_s', 't_next_valid', 'next_stroke_type', 'later_real_strokes',
            'score_A_before', 'score_B_before', 'shooter_score_before', 'opponent_score_before', 'score_diff_before',
            'rally_len', 'rally_real_len', 'rally_has_placeholder',
            'rally_winner', 'match_winner_name', 'end_reason', 'end_category', 'outcome_consistent',
            'S1_rally_won', 'S2_won_directly', 'smash_result', 't_next_for_smash_s', 'has_reach_inputs',
            'rotated', 'coord_suspect', 'geometry_ok',
            'hit_court_x', 'hit_court_y', 'landing_court_x', 'landing_court_y', 'player_location_court_x',
            'player_location_court_y', 'opponent_location_court_x', 'opponent_location_court_y',
            'hit_x_m', 'hit_y_m', 'land_x_m', 'land_y_m', 'shooter_x_m', 'shooter_y_m', 'opp_x_m', 'opp_y_m']
    strokes = d[cols].copy()
    smashes = strokes[strokes.smash_kind != ''].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    strokes.round(4).to_csv(OUT / 'strokes_clean.csv', index=False, encoding='utf-8-sig')
    smashes.round(4).to_csv(OUT / 'smashes_clean.csv', index=False, encoding='utf-8-sig')
    m_out = m[['id', 'video', 'tournament', 'round', 'date', 'gender', 'set', 'duration', 'winner_name', 'loser_name',
               'court_x_left', 'court_x_right', 'corner_labels_swapped']].rename(columns={'id': 'match_id', 'set': 'n_sets'})
    m_out.insert(0, 'match_uid', uid_prefix + '-' + m_out.match_id.astype(str))
    m_out.insert(0, 'source', label)
    m_out.round(2).to_csv(OUT / 'matches_clean.csv', index=False, encoding='utf-8-sig')

    # player table (counts only; height / handedness / age are added separately from outside sources)
    rows = []
    for name in sorted(set(names), key=norm_name):
        sh = smashes[smashes.player_name == name]
        rows.append(dict(source=label, player_name=name, gender=gmap[norm_name(name)],
                         matches=int(((m.winner_name == name) | (m.loser_name == name)).sum()),
                         strokes=int((strokes.player_name == name).sum()),
                         smashes=int((sh.smash_kind == 'smash').sum()), wrist_smashes=int((sh.smash_kind == 'wrist smash').sum()),
                         smashes_faced=int(((smashes.opponent_name == name) & (smashes.smash_kind == 'smash')).sum())))
    players = pd.DataFrame(rows)
    players.to_csv(OUT / 'players.csv', index=False, encoding='utf-8-sig')

    if dataset == 'shuttleset22':
        m1 = pd.read_csv(SET1 / 'match.csv')
        name1 = {norm_name(n): n.strip() for n in pd.concat([m1.winner, m1.loser])}
        cw = players[['player_name', 'gender', 'matches', 'smashes', 'wrist_smashes', 'smashes_faced']].rename(
            columns={'player_name': 'player_name_shuttleset22'})
        cw['player_name_shuttleset'] = cw.player_name_shuttleset22.map(lambda n: name1.get(norm_name(n)))
        cw['seen_in_shuttleset'] = cw.player_name_shuttleset.notna()
        cw.to_csv(OUT / 'player_crosswalk.csv', index=False, encoding='utf-8-sig')
        flags.to_csv(OUT / 'matches_all_shuttleset22.csv', index=False, encoding='utf-8-sig')
        seen = {norm_name(n) for n in name1}
        mm = m_out.assign(w_seen=m_out.winner_name.map(lambda n: norm_name(n) in seen),
                          l_seen=m_out.loser_name.map(lambda n: norm_name(n) in seen))
        both, one_, none_ = int((mm.w_seen & mm.l_seen).sum()), int((mm.w_seen ^ mm.l_seen).sum()), int((~mm.w_seen & ~mm.l_seen).sum())
        log(f'Crosswalk to ShuttleSet players (names only, no data merged): {int(cw.seen_in_shuttleset.sum())} of {len(cw)} test-set players '
            f'also appear in ShuttleSet, {int((~cw.seen_in_shuttleset).sum())} do not. Test matches with both players known {both}, '
            f'one known {one_}, neither {none_}.')

    # ---- 9. report ---------------------------------------------------------------------------------------------------
    smk = smashes[smashes.smash_kind == 'smash']
    log(f'Output: {len(strokes)} strokes, {len(smashes)} smash-type strokes ({len(smk)} smash, '
        f'{int((smashes.smash_kind == "wrist smash").sum())} wrist smash).')
    for kind, sub in smashes.groupby('smash_kind'):
        vc = sub.smash_result.value_counts()
        log(f'{kind}: result counts {vc.to_dict()}; S1 available {int(sub.S1_rally_won.notna().sum())} '
            f'(winner {int(sub.S1_rally_won.sum())}); S2 available {int(sub.S2_won_directly.notna().sum())} '
            f'(won directly {int(sub.S2_won_directly.sum())}); reach inputs present {int(sub.has_reach_inputs.sum())}.')
        miss = {c: int(sub[c].isna().sum()) for c in ['hit_court_x', 'landing_court_x', 'player_location_court_x', 'opponent_location_court_x']}
        log(f'{kind}: missing coordinates {miss}; t_next_s available {int(sub.t_next_for_smash_s.notna().sum())}.')
    ok = smashes.dropna(subset=['shooter_y_m', 'opp_y_m'])
    amb = (ok.shooter_y_m - NET_M).abs() < 0.5
    same_side = (ok.opp_y_m < NET_M)
    log(f'Orientation: after rotation the shooter is on the top half for '
        f'{(ok.shooter_y_m < NET_M).mean():.1%} of {len(ok)} smash-type strokes; {int(amb.sum())} are within 0.5 m of the net line; '
        f'the opponent is on the same half as the shooter for {int(same_side.sum())}.')
    lan = smashes.dropna(subset=['land_y_m'])
    log(f'Smash landing on the shooter\'s own half (y < net): {int((lan.land_y_m < NET_M).sum())} of {len(lan)}.')
    per_match = smk.groupby('video').size()
    log(f'Smashes per match: min {per_match.min()}, median {int(per_match.median())}, max {per_match.max()}.')
    (OUT / 'cleaning_report.md').write_text(f'# {label} cleaning report (generated by cleaning/clean_shuttleset.py)\n\n'
                                            + '\n\n'.join(f'- {r}' for r in report) + '\n', encoding='utf-8')
    print('\n'.join(report))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    names = list(DATASETS) if which == 'all' else [which]
    assert all(n in DATASETS for n in names), f'choose one of {list(DATASETS)} or all'
    for n in names:
        print(f'\n===== {DATASETS[n]["label"]} =====')
        run(n)


if __name__ == '__main__':
    main()
