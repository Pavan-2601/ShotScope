"""EDA on the cleaned tables.  Run from anywhere:  python EDA/eda_all.py

TRAINING data (ShuttleSet, all 44 matches, data/clean/)  ->  EDA/all_matches/   full EDA, including outcome rates.
TEST data (ShuttleSet22, 50 matches, data/clean_shuttleset22/)  ->  EDA/test_set/   DESCRIPTIVE ONLY and outcome-blind:
    the loader drops every outcome column, so no decision (cutoffs, features, models) can be informed by how test smashes ended.
    Only counts, data quality and the distribution of INPUT features are looked at.

Prototype reach features used here (the final versions are built in Phase 5):
    d        distance (m) from the opponent's position to the line of flight (the infinite line through hit and landing point)
    t_hat    hit-to-opponent distance / v_med, a distance-based flight-time estimate (known without the next stroke)
    t_obs    observed time to the next stroke (only exists for returned smashes, so it cannot be a model feature)
    v_med    median observed shuttle speed of returned smashes in the TRAINING data (also applied to the test data)
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO / 'data' / 'clean'
TEST_DIR = REPO / 'data' / 'clean_shuttleset22'
OUT = REPO / 'EDA' / 'all_matches'
OUT_TEST = REPO / 'EDA' / 'test_set'

MIN_N = 10            # a rate is shown only with at least this many smashes behind it
THIN_N = 30           # between MIN_N and THIN_N the rate is shown but marked "thin evidence"
NET_M = (480 - 150) / 49.2
HOME = (3.05, NET_M + 3.35)   # centre of the opponent's half, the "home" position for out-of-position measures

# reference palette slots 1-3 (blue, orange, aqua) and the blue sequential ramp; ink tokens for all text
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
RAMP = ['#b7d3f6', '#6da7ec', '#2a78d6', '#184f95']
INK, MUTED, SURF, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#d9d8d3'
GENDER_COLOR = {'men': BLUE, 'women': ORANGE}

# columns that describe how a rally or smash turned out; the test-set loader removes them
OUTCOME_COLS = ['rally_winner', 'match_winner_name', 'end_reason', 'end_category', 'outcome_consistent', 'S1_rally_won',
                'S2_won_directly', 'smash_result', 'later_real_strokes', 'next_stroke_type', 't_next_s', 't_next_valid',
                't_next_for_smash_s', 'rally_len', 'rally_real_len', 'stroke_status', 'landing_height']

REPORT = []


def say(text):
    REPORT.append(text)
    print(text)


# ------------------------------------------------------------------------------------------------------------ helpers
def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def style(ax, grid='y'):
    ax.set_facecolor(SURF)
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, length=0, labelsize=8)
    if grid:
        getattr(ax, f'{grid}axis').grid(True, color=GRID, lw=1, zorder=0)
    ax.set_axisbelow(True)


def heading(fig, title, sub, y=0.985):
    fig.text(0.01, y, title, color=INK, fontsize=13, fontweight='bold', ha='left', va='top')
    fig.text(0.01, y - 0.035, sub, color=MUTED, fontsize=8.5, ha='left', va='top')


def legend(ax, handles, **kw):
    leg = ax.legend(handles=handles, frameon=False, fontsize=8, **kw)
    for t in leg.get_texts():
        t.set_color(INK)
    return leg


def dot(color, label, hollow=False, marker='o', size=8):
    return Line2D([], [], marker=marker, ls='', markersize=size, label=label, color=color,
                  markerfacecolor=SURF if hollow else color, markeredgecolor=color, markeredgewidth=1.8)


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, dpi=150, facecolor=SURF, bbox_inches='tight')
    plt.close(fig)


def label_rate(n):
    """How a rate with n smashes behind it is drawn: none / thin / solid."""
    return np.where(n < MIN_N, 'hidden', np.where(n < THIN_N, 'thin', 'solid'))


# ------------------------------------------------------------------------------------------------------------ data
def load(folder, outcome_blind):
    sm = pd.read_csv(folder / 'smashes_clean.csv')
    st = pd.read_csv(folder / 'strokes_clean.csv')
    if outcome_blind:
        sm = sm.drop(columns=[c for c in OUTCOME_COLS if c in sm.columns])
        st = st.drop(columns=[c for c in OUTCOME_COLS if c in st.columns])
    return st, sm


def add_reach(df, v_med=None):
    """Prototype reach features. Needs has_reach_inputs; rows without it get NaN."""
    ok = df.has_reach_inputs.to_numpy(bool)
    H = df[['hit_x_m', 'hit_y_m']].to_numpy(float)
    L = df[['land_x_m', 'land_y_m']].to_numpy(float)
    O = df[['opp_x_m', 'opp_y_m']].to_numpy(float)
    flight = np.hypot(*(L - H).T)
    h2o = np.hypot(*(O - H).T)
    num = np.abs((L[:, 0] - H[:, 0]) * (H[:, 1] - O[:, 1]) - (H[:, 0] - O[:, 0]) * (L[:, 1] - H[:, 1]))
    with np.errstate(divide='ignore', invalid='ignore'):
        d = num / flight
    out = df.copy()
    out['flight_m'] = np.where(ok, flight, np.nan)
    out['hit_to_opp_m'] = np.where(ok, h2o, np.nan)
    out['d_m'] = np.where(ok, d, np.nan)
    out['opp_home_m'] = np.hypot(df.opp_x_m - HOME[0], df.opp_y_m - HOME[1])
    if 't_next_for_smash_s' in df.columns:
        out['t_obs_s'] = out.t_next_for_smash_s.where(ok)
        out['v_obs_mps'] = out.flight_m / out.t_obs_s
    if v_med is not None:
        out['t_hat_s'] = out.hit_to_opp_m / v_med
    return out


def add_context(st):
    """Previous real stroke's type and the time since it. Both are known when the stroke is hit."""
    x = st[['rally_id', 'ball_round', 'is_placeholder', 'type', 'frame_sec']].sort_values(['rally_id', 'ball_round'])
    real = x[~x.is_placeholder].copy()
    g = real.groupby('rally_id')
    real['prev_type'] = g.type.shift(1)
    real['t_since_prev_s'] = real.frame_sec - g.frame_sec.shift(1)
    real.loc[real.t_since_prev_s <= 0, 't_since_prev_s'] = np.nan
    return real[['rally_id', 'ball_round', 'prev_type', 't_since_prev_s']]


def zone_region(zone, x, y):
    """Plain-language place of a landing zone from the median landing position (rotated frame, metres)."""
    if zone >= 10:
        return 'outside the singles court'
    depth = 'at the net' if y < 8 else 'mid court' if y < 11.5 else 'rear court'
    lane = 'left' if x < 2 else 'centre' if x < 4 else 'right'
    return f'{depth}, {lane}'


# ------------------------------------------------------------------------------------------------------------ 1. counts
def counts(sm, st, players):
    s = sm[sm.smash_kind == 'smash']
    lab = s.dropna(subset=['S2_won_directly'])
    # per player
    rows = []
    for name, g in sm.groupby('player_name'):
        rows.append(dict(player_name=name, gender=g.gender.iloc[0], smashes=int((g.smash_kind == 'smash').sum()),
                         wrist_smashes=int((g.smash_kind == 'wrist smash').sum()),
                         smashes_with_S2=int(lab[lab.player_name == name].shape[0]),
                         smashes_faced=int(((s.opponent_name == name)).sum())))
    pp = pd.DataFrame(rows).sort_values('smashes', ascending=False)
    pp.to_csv(OUT / 'counts_per_player.csv', index=False)
    n_hidden = int((pp.smashes < MIN_N).sum())
    say(f'Smashes per player (smash only): min {pp.smashes.min()}, median {int(pp.smashes.median())}, max {pp.smashes.max()}; '
        f'{n_hidden} players have fewer than {MIN_N} ({", ".join(pp[pp.smashes < MIN_N].player_name)}), '
        f'{int((pp.smashes < THIN_N).sum())} fewer than {THIN_N}.')
    g = sm.groupby(['gender', 'smash_kind']).size().unstack()
    say(f'Smash counts by gender: {g.to_dict()}.')

    fig, ax = plt.subplots(figsize=(7.6, 8.2), facecolor=SURF)
    style(ax, 'x')
    y = np.arange(len(pp))[::-1]
    for yi, r in zip(y, pp.itertuples()):
        c = GENDER_COLOR[r.gender]
        hollow = r.smashes < MIN_N
        ax.barh(yi, r.smashes, height=.62, color=SURF if hollow else c, edgecolor=c, linewidth=1.8, zorder=3)
        ax.text(r.smashes + 4, yi, f'{r.smashes}' + ('  (below the minimum of 10)' if hollow else ''), va='center', color=MUTED, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(pp.player_name, color=INK, fontsize=8)
    ax.set_xlim(0, pp.smashes.max() * 1.18)
    ax.set_xlabel('smashes (smash only, not wrist smash)', color=MUTED, fontsize=8)
    ax.axvline(MIN_N, color=MUTED, lw=1, ls=':', zorder=2)
    legend(ax, [dot(BLUE, 'men', marker='s'), dot(ORANGE, 'women', marker='s'), dot(MUTED, 'outline = fewer than 10', hollow=True, marker='s')],
           loc='lower right')
    heading(fig, 'Smashes per player', 'ShuttleSet, 44 matches. A player with fewer than 10 smashes gets no rate in the interface.')
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, OUT, 'counts_per_player.png')

    # per landing zone
    zone = lab.dropna(subset=['landing_area']).copy()
    zone['zone'] = zone.landing_area.astype(int)
    allz = s.dropna(subset=['landing_area']).assign(zone=lambda d: d.landing_area.astype(int))
    zt = allz.groupby('zone').agg(smashes=('zone', 'size'), land_x_m=('land_x_m', 'median'), land_y_m=('land_y_m', 'median'))
    zt['smashes_with_S2'] = zone.groupby('zone').size().reindex(zt.index).fillna(0).astype(int)
    zt = zt.sort_values('smashes', ascending=False)
    zt['region'] = [zone_region(z, r_.land_x_m, r_.land_y_m) for z, r_ in zt.iterrows()]
    zt.round(2).to_csv(OUT / 'counts_per_zone.csv')
    top3 = zt.smashes.head(3)
    say(f'Landing zones: {len(zt)} zones; zones {list(top3.index)} hold {top3.sum()} of {zt.smashes.sum()} smashes '
        f'({top3.sum() / zt.smashes.sum():.0%}); {int((zt.smashes_with_S2 < MIN_N).sum())} zones have fewer than {MIN_N} labelled smashes.')

    # player x zone
    pz = zone.groupby(['player_name', 'zone']).size().unstack(fill_value=0)
    long = pz.stack().rename('smashes_with_S2').reset_index()
    long[long.smashes_with_S2 > 0].to_csv(OUT / 'counts_player_zone.csv', index=False)
    cells = pz.values[pz.values > 0]
    say(f'Player x landing-zone cells: {len(cells)} non-empty; {int((cells >= MIN_N).sum())} reach {MIN_N} smashes '
        f'(covering {cells[cells >= MIN_N].sum() / cells.sum():.0%} of labelled smashes), {int((cells >= THIN_N).sum())} reach {THIN_N}; '
        f'median cell {int(np.median(cells))}.')
    order_p = pz.sum(axis=1).sort_values(ascending=False).index
    order_z = pz.sum(axis=0).sort_values(ascending=False).index
    pz = pz.loc[order_p, order_z]
    fig, ax = plt.subplots(figsize=(8.6, 8.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    bins = [(MIN_N, 20), (20, 30), (30, 60), (60, 10 ** 6)]
    for i, p in enumerate(pz.index):
        for j, z in enumerate(pz.columns):
            n = int(pz.loc[p, z])
            if n == 0:
                continue
            if n < MIN_N:
                ax.add_patch(plt.Rectangle((j - .45, i - .45), .9, .9, facecolor=SURF, edgecolor=MUTED, lw=1.3, zorder=2))
                ax.text(j, i, str(n), ha='center', va='center', fontsize=7, color=MUTED)
            else:
                k = [b[0] <= n < b[1] for b in bins].index(True)
                ax.add_patch(plt.Rectangle((j - .45, i - .45), .9, .9, facecolor=RAMP[k], edgecolor=SURF, lw=2, zorder=2))
                ax.text(j, i, str(n), ha='center', va='center', fontsize=7, color='#ffffff' if k >= 2 else INK)
    ax.set_xlim(-.6, len(pz.columns) - .4)
    ax.set_ylim(len(pz.index) - .4, -.6)
    ax.set_xticks(range(len(pz.columns)))
    ax.set_xticklabels([f'zone {z}' for z in pz.columns], rotation=45, ha='right', color=INK, fontsize=8)
    ax.set_yticks(range(len(pz.index)))
    ax.set_yticklabels(pz.index, color=INK, fontsize=8)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    handles = [dot(MUTED, 'fewer than 10: no rate shown', hollow=True, marker='s')] + \
              [dot(RAMP[k], t, marker='s') for k, t in enumerate(['10-19 (thin evidence)', '20-29 (thin evidence)', '30-59', '60 or more'])]
    legend(ax, handles, loc='upper center', bbox_to_anchor=(.5, -.12), ncol=3)
    heading(fig, 'Smashes behind each player x landing-zone cell', 'ShuttleSet, smashes with a known S2 label. Numbers are counts.')
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, OUT, 'counts_player_zone.png')
    return pp, zt


# ------------------------------------------------------------------------------------------------------------ 2. outcome rates by landing zone (B1 raw)
def zone_rates(sm):
    s = sm[sm.smash_kind == 'smash'].dropna(subset=['landing_area']).copy()
    s['zone'] = s.landing_area.astype(int)
    rows = []
    for z, g in s.groupby('zone'):
        g2 = g.dropna(subset=['S2_won_directly'])
        g1 = g.dropna(subset=['S1_rally_won'])
        lo2, hi2 = wilson(g2.S2_won_directly.sum(), len(g2))
        lo1, hi1 = wilson(g1.S1_rally_won.sum(), len(g1))
        rows.append(dict(zone=z, smashes=len(g), n_S2=len(g2), S2_rate=g2.S2_won_directly.mean(), S2_lo=lo2, S2_hi=hi2,
                         n_S1=len(g1), S1_rate=g1.S1_rally_won.mean(), S1_lo=lo1, S1_hi=hi1,
                         shooter_error_rate=(g.smash_result == 'shooter_error').mean(),
                         evidence=str(label_rate(np.array([len(g2)]))[0])))
    t = pd.DataFrame(rows).sort_values('smashes', ascending=False)
    zinfo = pd.read_csv(OUT / 'counts_per_zone.csv').set_index('zone').region
    t['region'] = t.zone.map(zinfo)
    t.round(3).to_csv(OUT / 'outcome_rates_by_zone.csv', index=False)
    out_ = s[s.landing_area >= 10]
    say(f'Smashes that land outside the singles court (zones 10-16): {len(out_)} of {len(s)} ({len(out_) / len(s):.1%}); '
        f'{(out_.smash_result == "shooter_error").mean():.0%} are recorded as shooter errors (shooter won the rally in {out_.S1_rally_won.mean():.0%} of them).')
    net_ = s[s.landing_area.isin(t[t.region.str.startswith('at the net')].zone)]
    say(f'Smashes that land at the net line (zones {sorted(net_.landing_area.astype(int).unique())}): {len(net_)}, '
        f'{(net_.smash_result == "shooter_error").mean():.0%} are shooter errors.')
    allr = s.dropna(subset=['S2_won_directly'])
    say(f'Overall (smash only): S2 direct-win rate {allr.S2_won_directly.mean():.1%} of {len(allr)}; '
        f'S1 rally-won rate {s.S1_rally_won.mean():.1%} of {int(s.S1_rally_won.notna().sum())}; '
        f'shooter error rate {(s.smash_result == "shooter_error").mean():.1%}.')
    big = t[t.n_S2 >= MIN_N]
    say('Rates by landing zone (zones with at least 10 labelled smashes): ' +
        '; '.join(f'zone {int(r.zone)}: S2 {r.S2_rate:.0%} (n={r.n_S2}), S1 {r.S1_rate:.0%}' for r in big.itertuples()) + '.')

    shown = t[t.smashes >= MIN_N].reset_index(drop=True)
    hidden = t[t.smashes < MIN_N]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6), facecolor=SURF, sharey=True)
    ylab = [f'zone {int(r.zone)}, {r.region}  ({r.smashes} smashes)' for r in shown.itertuples()]
    y = np.arange(len(shown))[::-1]
    for ax, key, title in ((axes[0], 'S2', 'S2: won the point directly (shooter errors left out)'),
                           (axes[1], 'S1', 'S1: shooter won the rally')):
        style(ax, 'x')
        ax.set_title(title, loc='left', color=INK, fontsize=9.5, fontweight='bold')
        for yi, r in zip(y, shown.itertuples()):
            n = getattr(r, f'n_{key}')
            if n < MIN_N:
                ax.text(.01, yi, f'only {n} with a label: no rate', va='center', color=MUTED, fontsize=7.5)
                continue
            rate, lo, hi = getattr(r, f'{key}_rate'), getattr(r, f'{key}_lo'), getattr(r, f'{key}_hi')
            ax.plot([lo, hi], [yi, yi], color=BLUE, lw=2, solid_capstyle='round', zorder=3, alpha=.9)
            ax.plot(rate, yi, 'o', markersize=9, markerfacecolor=SURF if n < THIN_N else BLUE, markeredgecolor=BLUE, markeredgewidth=2, zorder=4)
            ax.text(min(hi + .03, .97), yi + .27, f'n={n}', color=MUTED, fontsize=7, va='bottom')
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ylab, color=INK, fontsize=8)
    legend(axes[1], [dot(BLUE, '30 or more smashes'), dot(BLUE, '10-29 smashes: thin evidence', hollow=True)], loc='lower right')
    fig.text(0.01, 0.005, f'Not drawn: {len(hidden)} zones with fewer than {MIN_N} smashes in total ({int(hidden.smashes.sum())} smashes).', color=MUTED, fontsize=8)
    heading(fig, 'How smashes end, by landing zone (the "landing zone only" baseline B1, raw)',
            'ShuttleSet, smash only. Dot = rate, line = 95% interval. A rate needs at least 10 smashes behind it (counted separately for S1 and S2).')
    fig.tight_layout(rect=(0, 0.02, 1, .92))
    save(fig, OUT, 'outcome_rates_by_zone.png')
    return t


# ------------------------------------------------------------------------------------------------------------ 3. distributions
def distributions(r):
    feats = [('d_m', "d: opponent's distance to the line of flight (m)"), ('t_hat_s', 't_hat: estimated flight time (s)'),
             ('t_obs_s', 't_obs: observed time to next stroke (s), whole frames'), ('flight_m', 'flight distance, hit to landing (m)'),
             ('v_obs_mps', 'observed shuttle speed (m/s), returned smashes only'), ('hit_to_opp_m', 'hit point to opponent (m)')]
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), facecolor=SURF)
    for ax, (col, title) in zip(axes.ravel(), feats):
        style(ax, 'y')
        ax.set_title(title, loc='left', color=INK, fontsize=8.5)
        lo, hi = r[col].quantile([.005, .995])
        bins = np.arange(lo - 1 / 60, hi + 1 / 60, 1 / 30) if col == 't_obs_s' else np.linspace(lo, hi, 36)
        for kind, color in (('smash', BLUE), ('wrist smash', AQUA)):
            v = r.loc[r.smash_kind == kind, col].dropna()
            rows.append(dict(feature=col, smash_kind=kind, n=len(v), mean=v.mean(), sd=v.std(), min=v.min(), q05=v.quantile(.05),
                             q25=v.quantile(.25), median=v.median(), q75=v.quantile(.75), q95=v.quantile(.95), max=v.max()))
            ax.hist(v, bins=bins, range=(bins[0], bins[-1]), density=True, histtype='step', color=color, lw=2, zorder=3)
        ax.set_yticks([])
    legend(axes[0, 0], [Line2D([], [], color=BLUE, lw=2, label='smash'), Line2D([], [], color=AQUA, lw=2, label='wrist smash')], loc='upper right')
    heading(fig, 'Distributions of the prototype reach features', 'ShuttleSet. Shapes are normalised so smash and wrist smash can be compared; the extreme 1% of values are not drawn. Observed times are whole video frames (1/30 s).')
    fig.tight_layout(rect=(0, 0, 1, .92))
    save(fig, OUT, 'distributions_reach_features.png')
    t = pd.DataFrame(rows)
    t.round(3).to_csv(OUT / 'distributions_reach_features.csv', index=False)
    sm_ = t[t.smash_kind == 'smash'].set_index('feature')
    say('Reach features (smash): ' + '; '.join(f'{f} median {sm_.loc[f, "median"]:.2f} (IQR {sm_.loc[f, "q25"]:.2f} to {sm_.loc[f, "q75"]:.2f})'
                                                for f in ['d_m', 't_hat_s', 't_obs_s', 'flight_m', 'v_obs_mps']) + '.')
    w = t[t.smash_kind == 'wrist smash'].set_index('feature')
    say(f'Wrist smashes are slower than smashes: median observed speed {w.loc["v_obs_mps", "median"]:.1f} vs {sm_.loc["v_obs_mps", "median"]:.1f} m/s.')
    return t


# ------------------------------------------------------------------------------------------------------------ 4. t_obs vs t_hat
def t_check(r, v_med):
    x = r.dropna(subset=['t_obs_s', 't_hat_s']).copy()
    x['diff_s'] = x.t_obs_s - x.t_hat_s
    kind_med = x.groupby('smash_kind').v_obs_mps.median()
    x['t_hat_kind_s'] = x.hit_to_opp_m / x.smash_kind.map(kind_med)
    res = {}
    for name, col in (('pooled speed', 't_hat_s'), ('speed by smash type', 't_hat_kind_s')):
        e = x.t_obs_s - x[col]
        res[name] = dict(corr=x.t_obs_s.corr(x[col]), mae=e.abs().mean(), median_error=e.median(), within_0_10s=(e.abs() < .10).mean())
    say(f'Flight time, observed vs estimated ({len(x)} returned smash-type strokes): v_med = {v_med:.1f} m/s. ' +
        '; '.join(f'{k}: correlation {v["corr"]:.2f}, mean absolute error {v["mae"]:.3f} s, {v["within_0_10s"]:.0%} within 0.10 s' for k, v in res.items()) + '.')
    say(f'Spread: t_hat has standard deviation {x.t_hat_s.std():.3f} s (coefficient of variation {x.t_hat_s.std() / x.t_hat_s.mean():.0%}), '
        f't_obs {x.t_obs_s.std():.3f} s ({x.t_obs_s.std() / x.t_obs_s.mean():.0%}); the hit-to-opponent distance varies little because almost every smash travels about {x.flight_m.median():.0f} m. '
        f'One video frame is {1 / 30:.3f} s, so part of the spread in t_obs is timing noise.')
    slow = x[x.t_obs_s < 0.2]
    say(f'Observed flight times under 0.2 s: {len(slow)} of {len(x)}; over 1.0 s: {int((x.t_obs_s > 1.0).sum())}.')
    out = x.reindex(x.diff_s.abs().sort_values(ascending=False).index).head(25)
    out[['video', 'set', 'rally', 'ball_round', 'smash_kind', 'shooter_name' if 'shooter_name' in out else 'player_name', 'flight_m', 'hit_to_opp_m',
         't_obs_s', 't_hat_s', 'diff_s']].round(3).to_csv(OUT / 't_obs_vs_t_hat_outliers.csv', index=False)
    pd.DataFrame(res).T.round(3).to_csv(OUT / 't_obs_vs_t_hat_summary.csv')

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), facecolor=SURF)
    ax = axes[0]
    style(ax, 'y')
    for kind, color, m in (('smash', BLUE, 'o'), ('wrist smash', AQUA, '^')):
        s_ = x[x.smash_kind == kind]
        ax.scatter(s_.t_hat_s, s_.t_obs_s, s=14, color=color, alpha=.45, edgecolor='none', marker=m, zorder=3)
    lim = [0.1, 1.2]
    ax.plot(lim, lim, color=INK, lw=1.2, zorder=4)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('t_hat: estimated from distance (s)', color=MUTED, fontsize=8)
    ax.set_ylabel('t_obs: observed (s)', color=MUTED, fontsize=8)
    ax.set_title('Observed vs estimated flight time', loc='left', color=INK, fontsize=9.5, fontweight='bold')
    legend(ax, [dot(BLUE, 'smash'), dot(AQUA, 'wrist smash', marker='^'), Line2D([], [], color=INK, lw=1.2, label='equal')], loc='upper left')
    ax = axes[1]
    style(ax, 'y')
    lo, hi = x.diff_s.quantile([.005, .995])
    ax.hist(x.diff_s, bins=40, range=(lo, hi), color=BLUE, edgecolor=SURF, linewidth=1, zorder=3)
    ax.axvline(0, color=INK, lw=1.2, zorder=4)
    ax.set_xlabel('t_obs minus t_hat (s)', color=MUTED, fontsize=8)
    ax.set_yticks([])
    ax.set_title('Estimation error', loc='left', color=INK, fontsize=9.5, fontweight='bold')
    heading(fig, 'Can the distance-based estimate stand in for the observed flight time?', 'ShuttleSet, returned smashes and wrist smashes (the only ones with an observed time).')
    fig.tight_layout(rect=(0, 0, 1, .9))
    save(fig, OUT, 't_obs_vs_t_hat.png')
    return res


# ------------------------------------------------------------------------------------------------------------ 5. heterogeneity across matches
def match_heterogeneity(st, sm, r):
    s = r[r.smash_kind == 'smash']
    rows = []
    for v, g in st.groupby('video'):
        real = g[~g.is_placeholder]
        gs = s[s.video == v]
        both = r[r.video == v]
        n_w = int((both.smash_kind == 'wrist smash').sum())
        g2 = gs.dropna(subset=['S2_won_directly'])
        g1 = gs.dropna(subset=['S1_rally_won'])
        pos = real.dropna(subset=['shooter_x_m', 'opp_x_m'])
        inside = (pos.shooter_x_m.between(-.3, 6.4) & pos.shooter_y_m.between(-.3, 13.7) & pos.opp_x_m.between(-.3, 6.4) & pos.opp_y_m.between(-.3, 13.7)).mean()
        rows.append(dict(video=v, gender=g.gender.iloc[0], tournament=g.tournament.iloc[0], strokes=len(real), smashes=len(gs),
                         smash_per_100_strokes=100 * len(gs) / len(real), wrist_per_100_strokes=100 * n_w / len(real),
                         smash_type_per_100_strokes=100 * len(both) / len(real), wrist_share_of_smash_type=n_w / max(len(both), 1), n_S2=len(g2), S2_rate=g2.S2_won_directly.mean() if len(g2) else np.nan,
                         n_S1=len(g1), S1_rate=g1.S1_rally_won.mean() if len(g1) else np.nan, inside_court_share=inside,
                         median_d_m=gs.d_m.median(), median_t_hat_s=gs.t_hat_s.median(), median_flight_m=gs.flight_m.median(),
                         missing_hit_share=gs.hit_court_x.isna().mean() if len(gs) else np.nan))
    t = pd.DataFrame(rows)
    # robust outliers within gender (modified z-score)
    flags = {}
    for col in ['smash_per_100_strokes', 'smash_type_per_100_strokes', 'wrist_share_of_smash_type', 'S2_rate', 'S1_rate', 'inside_court_share', 'median_d_m', 'median_t_hat_s', 'median_flight_m']:
        z = pd.Series(np.nan, index=t.index)
        for gnd, idx in t.groupby('gender').groups.items():
            x = t.loc[idx, col]
            ok = (t.loc[idx, 'n_S2'] >= MIN_N) if col in ('S2_rate', 'S1_rate') else x.notna()
            xs = x[ok]
            mad = (xs - xs.median()).abs().median()
            if mad > 0:
                z.loc[xs.index] = 0.6745 * (xs - xs.median()) / mad
        t[f'z_{col}'] = z
        flags[col] = t.loc[z.abs() > 3.5, 'video'].tolist()
    t['flagged_on'] = t.apply(lambda row: ', '.join(c for c in flags if row.video in flags[c]), axis=1)
    t.round(3).to_csv(OUT / 'match_heterogeneity.csv', index=False)
    say('Odd matches (modified z > 3.5 within gender): ' + ('; '.join(f'{c}: {len(v)}' for c, v in flags.items() if v) or 'none') + '.')
    for c, v in flags.items():
        for vid in v:
            say(f'  flagged on {c}: {vid}')

    # funnel check for S2: is the spread between matches larger than chance?
    f = t[t.n_S2 >= MIN_N]
    k_all, n_all = (f.S2_rate * f.n_S2).sum(), f.n_S2.sum()
    p0 = k_all / n_all
    zz = (f.S2_rate - p0) / np.sqrt(p0 * (1 - p0) / f.n_S2)
    chi2 = float((zz ** 2).sum())
    pval = 1 - stats.chi2.cdf(chi2, len(f) - 1)
    say(f'Spread of the S2 direct-win rate across the {len(f)} matches with at least {MIN_N} labelled smashes: pooled rate {p0:.1%}; '
        f'{int((zz.abs() > 1.96).sum())} matches lie outside the 95% funnel (about {0.05 * len(f):.1f} expected by chance); '
        f'chi-square {chi2:.0f} on {len(f) - 1} degrees of freedom (dispersion {chi2 / (len(f) - 1):.2f}, p = {pval:.2g}).')
    ws = t.wrist_share_of_smash_type
    say(f'Share of smash-type strokes labelled wrist smash, per match: median {ws.median():.0%}, middle half {ws.quantile(.25):.0%} to {ws.quantile(.75):.0%}, '
        f'range {ws.min():.0%} to {ws.max():.0%}. Smash-type strokes per 100 strokes (both kinds together) vary less: median {t.smash_type_per_100_strokes.median():.1f}, '
        f'middle half {t.smash_type_per_100_strokes.quantile(.25):.1f} to {t.smash_type_per_100_strokes.quantile(.75):.1f}.')
    odd = t[(t.wrist_share_of_smash_type >= .75) | (t.wrist_share_of_smash_type <= .10)]
    for o in odd.itertuples():
        say(f'  extreme wrist-smash share: {o.video}: {o.wrist_share_of_smash_type:.0%} of its {int(round(o.smash_type_per_100_strokes * o.strokes / 100))} smash-type strokes '
            f'({o.smash_per_100_strokes:.1f} smashes and {o.wrist_per_100_strokes:.1f} wrist smashes per 100 strokes)')
    allk = r.dropna(subset=['S2_won_directly'])
    byk = allk.groupby('smash_kind').S2_won_directly.agg(['mean', 'size'])
    m_ = allk.groupby('video').agg(share=('smash_kind', lambda x: (x == 'wrist smash').mean()), s2=('S2_won_directly', 'mean'), n=('S2_won_directly', 'size'))
    m_ = m_[m_.n >= MIN_N]
    say(f'S2 direct-win rate: smash {byk.loc["smash", "mean"]:.1%} (n={int(byk.loc["smash", "size"])}) vs wrist smash {byk.loc["wrist smash", "mean"]:.1%} (n={int(byk.loc["wrist smash", "size"])}). '
        f'Across the {len(m_)} matches with at least {MIN_N} labelled smash-type strokes, the match-level rank correlation between the wrist-smash share and the S2 rate of all its smash-type strokes is '
        f'{stats.spearmanr(m_.share, m_.s2)[0]:+.2f}.')
    wide = allk.groupby(['video', 'smash_kind']).S2_won_directly.agg(['mean', 'size']).unstack()
    both_ok = wide[(wide[('size', 'smash')] >= 5) & (wide[('size', 'wrist smash')] >= 5)]
    diff = both_ok[('mean', 'smash')] - both_ok[('mean', 'wrist smash')]
    say(f'Within the same match (matches with at least 5 of each kind, n = {len(both_ok)}): smash direct-win rate minus wrist-smash rate has median {diff.median():+.1%} '
        f'and is positive in {int((diff > 0).sum())} of {len(diff)} matches.')
    gs_ = t.groupby('gender').agg(matches=('video', 'size'), smash_per_100=('smash_per_100_strokes', 'median'), S2=('S2_rate', 'median'))
    say(f'By gender (median over matches): {gs_.round(3).to_dict("index")}.')

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), facecolor=SURF)
    ax = axes[0]
    style(ax, 'y')
    nn = np.arange(MIN_N, max(60, f.n_S2.max() + 5))
    for zlim, ls in ((1.96, '--'), (3.09, ':')):
        ax.plot(nn, p0 + zlim * np.sqrt(p0 * (1 - p0) / nn), color=MUTED, lw=1.2, ls=ls, zorder=2)
        ax.plot(nn, np.clip(p0 - zlim * np.sqrt(p0 * (1 - p0) / nn), 0, 1), color=MUTED, lw=1.2, ls=ls, zorder=2)
    ax.axhline(p0, color=INK, lw=1.2, zorder=2)
    for gnd, color in GENDER_COLOR.items():
        g_ = f[f.gender == gnd]
        ax.scatter(g_.n_S2, g_.S2_rate, s=46, color=color, edgecolor=SURF, linewidth=1.5, zorder=4)
    ax.set_xlabel('smashes with a known S2 label in the match', color=MUTED, fontsize=8)
    ax.set_ylabel('share that won the point directly', color=MUTED, fontsize=8)
    ax.set_yticks([0, .1, .2, .3, .4])
    ax.set_yticklabels(['0%', '10%', '20%', '30%', '40%'])
    ax.set_title('Direct-win rate per match vs the range chance allows', loc='left', color=INK, fontsize=8.5)
    legend(ax, [dot(BLUE, 'men'), dot(ORANGE, 'women'), Line2D([], [], color=MUTED, lw=1.2, ls='--', label='95% limit'),
                Line2D([], [], color=MUTED, lw=1.2, ls=':', label='99.8% limit')], loc='upper right')
    ax = axes[1]
    style(ax, 'y')
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    for gnd, color in GENDER_COLOR.items():
        g_ = t[t.gender == gnd]
        ax.scatter(g_.smash_per_100_strokes, g_.wrist_per_100_strokes, s=46, color=color, edgecolor=SURF, linewidth=1.5, zorder=4)
    n_hi, n_lo = int((odd.wrist_share_of_smash_type >= .75).sum()), int((odd.wrist_share_of_smash_type <= .10).sum())
    ax.text(0.35, 9.3, f'{n_hi} matches file almost every\nsmash-type stroke as a wrist smash', color=INK, fontsize=7.5, va='top')
    ax.text(14.3, 3.9, f'{n_lo} matches file almost\nnone as a wrist smash', color=INK, fontsize=7.5, va='top', ha='right')
    ax.set_xlabel('smashes per 100 strokes', color=MUTED, fontsize=8)
    ax.set_ylabel('wrist smashes per 100 strokes', color=MUTED, fontsize=8)
    ax.set_title('Smash vs wrist smash: is the split the same in every match?', loc='left', color=INK, fontsize=8.5)
    legend(ax, [dot(BLUE, 'men'), dot(ORANGE, 'women')], loc='center right')
    heading(fig, 'Do the 44 matches differ more than chance?', 'ShuttleSet. Left: smash only. Right: both kinds, one dot per match.')
    fig.tight_layout(rect=(0, 0, 1, .92))
    save(fig, OUT, 'match_heterogeneity.png')
    return t


# ------------------------------------------------------------------------------------------------------------ 6. selection effect
def selection_effect(st_r, r):
    s = st_r[(~st_r.is_placeholder) & st_r.geometry_ok & st_r.opp_home_m.notna() & (st_r.stroke_status != 'serve')].copy()
    order = ['smash', 'wrist smash', 'clear', 'drop', 'lob', 'push', 'drive', 'net shot', 'return net', 'rush']
    rows = []
    for t_ in order:
        v = s.loc[s.type == t_, 'opp_home_m']
        rows.append(dict(type=t_, n=len(v), median=v.median(), q25=v.quantile(.25), q75=v.quantile(.75)))
    ty = pd.DataFrame(rows).sort_values('median', ascending=False)
    ty.round(2).to_csv(OUT / 'selection_opponent_distance_by_type.csv', index=False)

    back = s[(s.shooter_y_m < 4.5) & (s.hit_height == 1)].copy()
    back['is_smash_type'] = back.smash_kind != ''
    bins = [0, 1, 2, 3, 20]
    labels = ['under 1 m', '1-2 m', '2-3 m', '3 m or more']
    back['bin'] = pd.cut(back.opp_home_m, bins, labels=labels, right=False)
    pb = back.groupby('bin', observed=True).agg(strokes=('is_smash_type', 'size'), smash_type=('is_smash_type', 'sum')).reset_index()
    pb['share'] = pb.smash_type / pb.strokes
    pb['lo'], pb['hi'] = wilson(pb.smash_type, pb.strokes)
    pb.round(3).to_csv(OUT / 'selection_smash_propensity.csv', index=False)
    say(f'Selection, opponent displacement: among {len(back)} strokes hit from the back of the court at or above net height, the share that are smash or wrist smash is '
        + ', '.join(f'{r_.share:.0%} when the opponent is {r_.bin} from the centre of their half (n={r_.strokes})' for r_ in pb.itertuples())
        + '. Smashes are NOT more likely when the opponent is further from home; most opponents are within 2 m of home at the moment of the smash '
        f'(median {s.loc[s.smash_kind != "", "opp_home_m"].median():.1f} m).')

    sm_ = r[r.smash_kind == 'smash']
    prev = sm_.prev_type.value_counts()
    pt = pd.DataFrame({'preceding_stroke': prev.index, 'n': prev.values, 'share': (prev / prev.sum()).values})
    pt.round(3).to_csv(OUT / 'selection_preceding_stroke.csv', index=False)
    say('What smashes answer (the opponent\'s preceding stroke, smash only): ' +
        ', '.join(f'{x.preceding_stroke} {x.share:.0%}' for x in pt.head(4).itertuples()) + f' (n = {int(prev.sum())}).')

    lab = sm_.dropna(subset=['S2_won_directly'])
    rows = []
    specs = (('opp_home_m', 'opponent distance from home', [0, 1, 2, 3, 20], labels),
             ('d_m', 'd: distance to the line of flight', [0, .5, 1, 1.5, 2.5, 30], ['under 0.5 m', '0.5-1 m', '1-1.5 m', '1.5-2.5 m', '2.5 m or more']))
    for col, name, edges, lab_ in specs:
        x = lab.dropna(subset=[col])
        b_ = pd.cut(x[col], edges, labels=lab_, right=False)
        g = x.groupby(b_, observed=True).agg(n=('S2_won_directly', 'size'), k=('S2_won_directly', 'sum')).reset_index()
        g.columns = ['bin', 'n', 'k']
        g['variable'] = name
        rows.append(g)
    x = lab.dropna(subset=['t_since_prev_s']).copy()
    qb = pd.qcut(x.t_since_prev_s, 4, duplicates='drop')
    g = x.groupby(qb, observed=True).agg(n=('S2_won_directly', 'size'), k=('S2_won_directly', 'sum')).reset_index()
    g.columns = ['bin', 'n', 'k']
    g['bin'] = [f'{i.left:.2f}-{i.right:.2f} s' for i in g.bin]
    g['variable'] = "time since the opponent's previous stroke"
    rows.append(g)
    ro = pd.concat(rows)
    ro['rate'] = ro.k / ro.n
    ro['lo'], ro['hi'] = wilson(ro.k, ro.n)
    ro.round(3).to_csv(OUT / 'selection_S2_by_opponent_position.csv', index=False)
    for name, g in ro.groupby('variable', sort=False):
        say(f'S2 direct-win rate by {name} (smash only): ' + '; '.join(f'{b.bin} {b.rate:.0%} (n={b.n})' for b in g.itertuples()) + '.')
    xd = lab.dropna(subset=['d_m', 't_since_prev_s'])
    say(f'Time since the opponent\'s previous stroke (smash only): median {sm_.t_since_prev_s.median():.2f} s (IQR {sm_.t_since_prev_s.quantile(.25):.2f} to '
        f'{sm_.t_since_prev_s.quantile(.75):.2f}); rank correlation with d {stats.spearmanr(xd.d_m, xd.t_since_prev_s)[0]:+.2f}, '
        f'with S2 {stats.spearmanr(x.t_since_prev_s, x.S2_won_directly)[0]:+.2f}; rank correlation of d with S2 '
        f'{stats.spearmanr(lab.dropna(subset=["d_m"]).d_m, lab.dropna(subset=["d_m"]).S2_won_directly)[0]:+.2f}.')
    lz = lab.dropna(subset=['d_m', 'landing_area']).copy()
    lz['zone_group'] = np.where(lz.landing_area.isin([5, 6, 8]), 'zones 5/6/8', 'other zones')
    say('d versus landing zone: ' + '; '.join(f'{k}: median d {v.d_m.median():.2f} m, S2 {v.S2_won_directly.mean():.0%} (n={len(v)})' for k, v in lz.groupby('zone_group')) +
        '; within zones 5/6/8 the rank correlation of d with S2 is ' +
        f'{stats.spearmanr(lz[lz.zone_group == "zones 5/6/8"].d_m, lz[lz.zone_group == "zones 5/6/8"].S2_won_directly)[0]:+.2f}.')

    fig, axes = plt.subplots(2, 3, figsize=(14.4, 8.8), facecolor=SURF)
    ax = axes[0, 0]
    style(ax, 'x')
    y = np.arange(len(ty))[::-1]
    for yi, r_ in zip(y, ty.itertuples()):
        c = BLUE if r_.type in ('smash', 'wrist smash') else MUTED
        ax.plot([r_.q25, r_.q75], [yi, yi], color=c, lw=2, solid_capstyle='round', zorder=3)
        ax.plot(r_.median, yi, 'o', markersize=9, color=c, markeredgecolor=SURF, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{t_} (n={n})' for t_, n in zip(ty.type, ty.n)], color=INK, fontsize=8)
    ax.set_xlabel("opponent's distance from the centre of their half (m)", color=MUTED, fontsize=8)
    ax.set_title('Where the opponent is, by shot type', loc='left', color=INK, fontsize=8.5, fontweight='bold')
    legend(ax, [dot(BLUE, 'smash types'), dot(MUTED, 'other shots')], loc='lower right')
    ax = axes[0, 1]
    style(ax, 'y')
    xs = np.arange(len(pb))
    ax.vlines(xs, pb.lo, pb.hi, color=BLUE, lw=2, zorder=3)
    ax.plot(xs, pb.share, 'o', markersize=9, color=BLUE, markeredgecolor=SURF, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{b}\n(n={n})' for b, n in zip(pb.bin, pb.strokes)], fontsize=7.5, color=MUTED)
    ax.set_ylim(0, .6)
    ax.set_yticks([0, .2, .4, .6])
    ax.set_yticklabels(['0%', '20%', '40%', '60%'])
    ax.set_ylabel('share of such strokes that are smashes', color=MUTED, fontsize=8)
    ax.set_xlabel('opponent distance from home', color=MUTED, fontsize=8)
    ax.set_title('Do players smash when the opponent is displaced?', loc='left', color=INK, fontsize=8.5, fontweight='bold')
    ax = axes[0, 2]
    style(ax, 'x')
    top = pt.head(7)
    yy = np.arange(len(top))[::-1]
    ax.barh(yy, top.share, height=.62, color=BLUE, zorder=3)
    for yi, r_ in zip(yy, top.itertuples()):
        ax.text(r_.share + .01, yi, f'{r_.share:.0%}', va='center', color=MUTED, fontsize=8)
    ax.set_yticks(yy)
    ax.set_yticklabels(top.preceding_stroke, color=INK, fontsize=8)
    ax.set_xlim(0, top.share.max() * 1.2)
    ax.set_xticks([])
    ax.set_title("What a smash answers (the opponent's previous stroke)", loc='left', color=INK, fontsize=8.5, fontweight='bold')
    for ax, var, color, ttl, xl in ((axes[1, 0], 'd: distance to the line of flight', ORANGE, 'Does reach relate to winning directly?', 'd: opponent distance to the line of flight'),
                                    (axes[1, 1], "time since the opponent's previous stroke", ORANGE, 'Does recovery time relate to winning directly?', "time since the opponent's previous stroke"),
                                    (axes[1, 2], 'opponent distance from home', ORANGE, 'Does displacement relate to winning directly?', 'opponent distance from home')):
        style(ax, 'y')
        g = ro[ro.variable == var]
        xs = np.arange(len(g))
        ax.vlines(xs, g.lo, g.hi, color=color, lw=2, zorder=3)
        for xi, r_ in zip(xs, g.itertuples()):
            ax.plot(xi, r_.rate, 'o', markersize=9, markerfacecolor=SURF if r_.n < THIN_N else color, markeredgecolor=color, markeredgewidth=2, zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels([f'{b}\n(n={n})' for b, n in zip(g.bin, g.n)], fontsize=7, color=MUTED)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
        ax.set_ylabel('smashes that won the point directly (S2)', color=MUTED, fontsize=8)
        ax.set_xlabel(xl, color=MUTED, fontsize=8)
        ax.set_title(ttl, loc='left', color=INK, fontsize=8.5, fontweight='bold')
    legend(axes[1, 2], [dot(ORANGE, '30 or more smashes'), dot(ORANGE, 'fewer than 30: thin evidence', hollow=True)], loc='upper left')
    heading(fig, 'Selection: when do players smash, and what predicts a direct win?', 'ShuttleSet, smash only for the bottom row. Dot = rate, line = 95% interval.')
    fig.tight_layout(rect=(0, 0, 1, .93))
    save(fig, OUT, 'selection_effect.png')


# ------------------------------------------------------------------------------------------------------------ 7. test set (outcome-blind)
def test_set_eda(train_r, v_med, zones_train):
    st, sm = load(TEST_DIR, outcome_blind=True)
    assert not (set(OUTCOME_COLS) & set(sm.columns)), 'outcome columns reached the test-set EDA'
    cw = pd.read_csv(TEST_DIR / 'player_crosswalk.csv')
    seen = dict(zip(cw.player_name_shuttleset22, cw.seen_in_shuttleset))
    r = add_reach(sm, v_med)
    OUT_TEST.mkdir(parents=True, exist_ok=True)
    s = sm[sm.smash_kind == 'smash']
    pp = sm.groupby('player_name').agg(gender=('gender', 'first'), smashes=('smash_kind', lambda x: int((x == 'smash').sum())),
                                       wrist_smashes=('smash_kind', lambda x: int((x == 'wrist smash').sum()))).reset_index()
    pp['seen_in_shuttleset'] = pp.player_name.map(seen)
    pp['smashes_faced'] = pp.player_name.map(s.groupby('opponent_name').size()).fillna(0).astype(int)
    pp = pp.sort_values('smashes', ascending=False)
    pp.to_csv(OUT_TEST / 'test_counts_per_player.csv', index=False)
    shooter_unseen = int(pp[~pp.seen_in_shuttleset].smashes.sum())
    say(f'TEST (descriptive): {len(s)} smashes, {int((sm.smash_kind == "wrist smash").sum())} wrist smashes in {st.video.nunique()} matches; '
        f'per player min {pp.smashes.min()}, median {int(pp.smashes.median())}, max {pp.smashes.max()}; '
        f'players with fewer than {MIN_N} smashes: {int((pp.smashes < MIN_N).sum())}. Unseen players hit {shooter_unseen} of {len(s)} smashes '
        f'({shooter_unseen / len(s):.0%}) and are the opponent in {int(pp[~pp.seen_in_shuttleset].smashes_faced.sum())}.')
    zc = s.dropna(subset=['landing_area']).assign(zone=lambda d: d.landing_area.astype(int)).groupby('zone').size()
    say(f'TEST landing zones: zones {list(zc.sort_values(ascending=False).head(3).index)} hold {zc.sort_values(ascending=False).head(3).sum()} of {zc.sum()} smashes.')
    gl = sm[sm.smash_kind == 'smash'].gender.value_counts(normalize=True)
    say(f'TEST gender mix of smashes: women {gl.get("women", 0):.0%}, men {gl.get("men", 0):.0%}.')

    # data quality per match
    rows = []
    for v, g in st.groupby('video'):
        real = g[~g.is_placeholder]
        pos = real.dropna(subset=['shooter_x_m', 'opp_x_m'])
        inside = (pos.shooter_x_m.between(-.3, 6.4) & pos.shooter_y_m.between(-.3, 13.7) & pos.opp_x_m.between(-.3, 6.4) & pos.opp_y_m.between(-.3, 13.7)).mean()
        sk = sm[(sm.video == v)]
        wn, ln = g.player_name.iloc[0], g.opponent_name.iloc[0]
        rows.append(dict(video=v, gender=g.gender.iloc[0], strokes=len(real), smashes=int((sk.smash_kind == 'smash').sum()),
                         inside_court_share=inside, geometry_ok_share=g.geometry_ok.mean(),
                         smashes_with_reach_inputs=int(sk.has_reach_inputs.sum()), smashes_total=len(sk),
                         players_known=int(seen.get(wn, False)) + int(seen.get(ln, False))))
    q = pd.DataFrame(rows)
    q.round(3).to_csv(OUT_TEST / 'test_match_quality.csv', index=False)
    say(f'TEST data quality: positions inside the court in every match >= {q.inside_court_share.min():.1%}; '
        f'{int((q.geometry_ok_share < 1).sum())} match has flagged rallies; smash-type strokes with usable reach inputs '
        f'{int(q.smashes_with_reach_inputs.sum())} of {int(q.smashes_total.sum())}; matches with both / one / neither player known in ShuttleSet: '
        f'{int((q.players_known == 2).sum())} / {int((q.players_known == 1).sum())} / {int((q.players_known == 0).sum())}.')

    # shift check on INPUT features only (never outcomes)
    feats = ['d_m', 't_hat_s', 'flight_m', 'hit_to_opp_m', 'opp_home_m']
    tr = train_r[train_r.smash_kind == 'smash']
    te = r[r.smash_kind == 'smash']
    rows = []
    for f in feats:
        a, b = tr[f].dropna(), te[f].dropna()
        sd = math.sqrt((a.var() + b.var()) / 2)
        rows.append(dict(feature=f, train_n=len(a), test_n=len(b), train_median=a.median(), test_median=b.median(),
                         train_iqr=a.quantile(.75) - a.quantile(.25), test_iqr=b.quantile(.75) - b.quantile(.25),
                         standardised_difference=(b.mean() - a.mean()) / sd, ks_statistic=stats.ks_2samp(a, b).statistic))
    sh = pd.DataFrame(rows)
    sh.round(3).to_csv(OUT_TEST / 'shift_check_inputs.csv', index=False)
    say('SHIFT CHECK (input features only, smash only, training vs test): ' +
        '; '.join(f'{x.feature}: median {x.train_median:.2f} vs {x.test_median:.2f}, standardised difference {x.standardised_difference:+.2f}' for x in sh.itertuples()) + '.')
    zt = zones_train.smashes / zones_train.smashes.sum()
    zs = zc / zc.sum()
    mix = pd.DataFrame({'train_share': zt, 'test_share': zs}).fillna(0).sort_values('train_share', ascending=False)
    mix.round(3).to_csv(OUT_TEST / 'shift_check_landing_zone_mix.csv')
    tvd = 0.5 * (mix.train_share - mix.test_share).abs().sum()
    say(f'SHIFT CHECK landing-zone mix: total variation distance {tvd:.2f} (0 = identical).')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor=SURF)
    ax = axes[0]
    style(ax, 'x')
    ppp = pp.sort_values('smashes')
    y = np.arange(len(ppp))
    for yi, r_ in zip(y, ppp.itertuples()):
        ax.barh(yi, r_.smashes, height=.65, color=ORANGE if r_.seen_in_shuttleset else SURF, edgecolor=ORANGE, linewidth=1.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(ppp.player_name, fontsize=6.5, color=INK)
    ax.set_xlabel('smashes in the test matches', color=MUTED, fontsize=8)
    ax.set_title('Test matches: smashes per player', loc='left', color=INK, fontsize=8.5, fontweight='bold')
    legend(ax, [dot(ORANGE, 'player also in ShuttleSet', marker='s'), dot(ORANGE, 'player not in ShuttleSet', hollow=True, marker='s')], loc='lower right')
    ax = axes[1]
    style(ax, 'y')
    lo, hi = pd.concat([tr.d_m, te.d_m]).quantile([.005, .995])
    bins = np.linspace(lo, hi, 36)
    ax.hist(tr.d_m.dropna(), bins=bins, range=(lo, hi), density=True, histtype='step', color=BLUE, lw=2, zorder=3)
    ax.hist(te.d_m.dropna(), bins=bins, range=(lo, hi), density=True, histtype='step', color=ORANGE, lw=2, zorder=3)
    ax.set_yticks([])
    ax.set_xlabel('d: opponent distance to the line of flight (m)', color=MUTED, fontsize=8)
    ax.set_title('Input shift: d', loc='left', color=INK, fontsize=8.5, fontweight='bold')
    legend(ax, [Line2D([], [], color=BLUE, lw=2, label='training (ShuttleSet)'), Line2D([], [], color=ORANGE, lw=2, label='test (ShuttleSet22)')], loc='upper right')
    ax = axes[2]
    style(ax, 'y')
    xs = np.arange(len(mix.head(8)))
    mm_ = mix.head(8)
    ax.bar(xs - .2, mm_.train_share, width=.38, color=BLUE, zorder=3)
    ax.bar(xs + .2, mm_.test_share, width=.38, color=ORANGE, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f'zone {z}' for z in mm_.index], rotation=45, ha='right', fontsize=7.5, color=MUTED)
    ax.set_yticks([0, .1, .2, .3, .4])
    ax.set_yticklabels(['0%', '10%', '20%', '30%', '40%'])
    ax.set_title('Input shift: landing-zone mix', loc='left', color=INK, fontsize=8.5, fontweight='bold')
    legend(ax, [dot(BLUE, 'training (ShuttleSet)', marker='s'), dot(ORANGE, 'test (ShuttleSet22)', marker='s')], loc='upper right')
    heading(fig, 'The test set, described without looking at outcomes', 'ShuttleSet22, 50 matches. Only counts, data quality and input features are shown; no rates or outcomes.')
    fig.tight_layout(rect=(0, 0, 1, .9))
    save(fig, OUT_TEST, 'test_set_overview.png')


# ------------------------------------------------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st, sm = load(TRAIN_DIR, outcome_blind=False)
    say(f'TRAINING (ShuttleSet): {st.video.nunique()} matches, {len(st)} strokes, {int((sm.smash_kind == "smash").sum())} smashes, '
        f'{int((sm.smash_kind == "wrist smash").sum())} wrist smashes.')
    r0 = add_reach(sm)
    v_med = r0.v_obs_mps.median()
    r = add_reach(sm, v_med)
    st_r = add_reach(st.assign(smash_kind=st.smash_kind.fillna('')), v_med)
    ctx = add_context(st)
    r = r.merge(ctx, on=['rally_id', 'ball_round'], how='left')
    say(f'Median observed shuttle speed of returned smash-type strokes (training): v_med = {v_med:.1f} m/s.')
    counts(sm, st, None)
    pp_zone = pd.read_csv(OUT / 'counts_per_zone.csv', index_col=0)
    zone_rates(sm)
    distributions(r)
    t_check(r, v_med)
    match_heterogeneity(st_r, sm, r)
    selection_effect(st_r, r)
    test_set_eda(r, v_med, pp_zone)
    (OUT / 'eda_report.md').write_text('# EDA report (generated by EDA/eda_all.py)\n\n' + '\n\n'.join(f'- {x}' for x in REPORT) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
