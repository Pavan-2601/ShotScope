"""EDA on the merged datasets (Axelsen v Ng, Thailand Open 2021 final). Player A = Axelsen.
Run from the ShuttleSet folder: python3 EDA/eda.py  (needs matplotlib). Images are written to EDA/.
Every analysis is a function taking one dataset; DATASETS lists the datasets it is run on.
Add each new analysis as its own numbered section before the __main__ block."""
import csv, os, math, statistics, collections
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- shared setup
OUT = 'EDA'
DATASETS = [  # (merged csv, image-name suffix, subtitle)
    ('MergedSet/Axelsen_vs_Ng_Thailand_Open_2021_Finals_set1_merged.csv', '',
     'Axelsen (A) vs Ng, Thailand Open 2021 final, set 1'),
    ('MergedSet/Axelsen_vs_Ng_Thailand_Open_2021_Finals_sets1-2_merged.csv', '_sets1-2',
     'Axelsen (A) vs Ng, Thailand Open 2021 final, sets 1-2'),
]
CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a']  # first three slots of the reference palette: blue, orange, aqua
SET_COLOR = {'1': CATEGORICAL[0], '2': CATEGORICAL[2]}  # one colour per set; orange is reserved for winning shots
WIN_COLOR = CATEGORICAL[1]
TYPE_MARKER = ['o', 's', 'D', '^']  # shot type -> marker when colour is used for the set
INK, MUTED, SURF, GRID, CONTEXT = '#0b0b0b', '#52514e', '#fcfcfb', '#d9d8d3', '#c9c8c2'


class Dataset:
    def __init__(self, src, suffix, subtitle):
        self.suffix, self.subtitle = suffix, subtitle
        self.rows = list(csv.DictReader(open(src, encoding='utf-8-sig')))
        self.sets = sorted({r['set'] for r in self.rows})
        self.A_rows = [r for r in self.rows if r['player'] == 'A']
        self.A_win = [r for r in self.A_rows if r['getpoint_player'] == 'A' and r['landing_court_x']]
        self.multi = len(self.sets) > 1
        self.win_types = [t for t, _ in collections.Counter(r['type'] for r in self.A_win).most_common()]

    def tag(self, r):
        return f"S{r['set']} R{r['rally']}" if self.multi else f"R{r['rally']}"

    def style(self, r):
        """(colour, marker) of a winning shot: type -> colour for one set; set -> colour, type -> marker for several."""
        i = self.win_types.index(r['type'])
        return (SET_COLOR[r['set']], TYPE_MARKER[i % 4]) if self.multi else (CATEGORICAL[i], 'o')

    def win_legend(self):
        n = collections.Counter(r['type'] for r in self.A_win)
        if not self.multi:
            return [dot(CATEGORICAL[i], f'{t} ({n[t]})') for i, t in enumerate(self.win_types)]
        ns = collections.Counter(r['set'] for r in self.A_win)
        return ([dot(SET_COLOR[s], f'set {s} ({ns[s]})') for s in self.sets] +
                [dot(MUTED, f'{t} ({n[t]})', marker=TYPE_MARKER[i % 4]) for i, t in enumerate(self.win_types)])


def court_lines(ax, full=True):
    """Court in court coordinates (50 units per metre), net at y = 480. full=False draws the y<480 half only."""
    segs = [((27.4, 150), (327.6, 150)), ((27.4, 150), (27.4, 480)), ((327.6, 150), (327.6, 480)),
            ((50, 150), (50, 480)), ((305, 150), (305, 480)), ((177.5, 150), (177.5, 382.6)),
            ((27.4, 382.6), (327.6, 382.6))]
    if full:
        segs += [((27.4, 810), (327.6, 810)), ((27.4, 480), (27.4, 810)), ((327.6, 480), (327.6, 810)),
                 ((50, 480), (50, 810)), ((305, 480), (305, 810)), ((177.5, 577.4), (177.5, 810)),
                 ((27.4, 577.4), (327.6, 577.4))]
    for a, b in segs:
        ax.plot(*zip(a, b), color=GRID, lw=1.2, zorder=1)
    ax.plot([15, 340], [480, 480], color=MUTED, lw=2.5, zorder=1)


def finish(ax, title, subtitle, ylim, legend_handles=None, ncol=4):
    ax.set_xlim(5, 350); ax.set_ylim(*ylim)  # inverted y: far baseline at the top, net towards the bottom
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, loc='left', color=INK, fontsize=13, fontweight='bold', pad=26 + 11 * subtitle.count('\n'))
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, color=MUTED, fontsize=8, va='bottom')
    if legend_handles:
        leg = ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(.5, -.005),
                        ncol=ncol, frameon=False, fontsize=9)
        for t in leg.get_texts(): t.set_color(INK)


def dot(color, label, hollow=False, marker='o'):
    return Line2D([], [], marker=marker, ls='', markersize=9, label=label, color=color,
                  markerfacecolor=SURF if hollow else color, markeredgecolor=color, markeredgewidth=2)


def flip_to_top(x, y):
    """Players switch ends between sets: rotate 180 deg so a point on the y > 480 half lands on the y < 480 half."""
    return (354.99 - x, 960 - y) if y > 480 else (x, y)


def label(ax, x, y, text, placed):
    """Point label to the right of the marker; moved above it when another label is already nearby."""
    near = any(abs(x - px) < 60 and abs(y - py) < 14 for px, py in placed)
    ax.annotate(text, (x, y), xytext=(8, 8 if near else -3), textcoords='offset points', color=MUTED, fontsize=8)
    placed.append((x, y))


# ---------------------------------------------------------------- 1. winning shots, landing spots (half court)
def winning_landings_half_court(ds):
    fig, ax = plt.subplots(figsize=(7, 6.4), facecolor=SURF); ax.set_facecolor(SURF)
    court_lines(ax, full=False)
    ax.text(177.5, 489, 'NET', ha='center', va='top', color=MUTED, fontsize=9)
    placed = []
    for r in ds.A_win:
        x, y = flip_to_top(float(r['landing_court_x']), float(r['landing_court_y']))
        c, m = ds.style(r)
        ax.scatter(x, y, s=110, color=c, marker=m, edgecolor=SURF, lw=1.5, zorder=3)
        label(ax, x, y, ds.tag(r), placed)
    finish(ax, "Where Player A's winning shots land",
           f'{len(ds.A_win)} rally-winning shots, {ds.subtitle}. S = set, R = rally number.' if ds.multi
           else f'{len(ds.A_win)} rally-winning shots, {ds.subtitle}. R = rally number.',
           (492, 130), ds.win_legend(), ncol=3 if not ds.multi else 4)
    fig.tight_layout(); fig.savefig(f'{OUT}/winning_shots_A{ds.suffix}.png', dpi=160, facecolor=SURF)


# ---------------------------------------------------------------- 2. winning shots, hit-from and landing (full court)
def winning_hit_and_landing_full_court(ds):
    fig, ax = plt.subplots(figsize=(6.4, 9.2), facecolor=SURF); ax.set_facecolor(SURF)
    court_lines(ax)
    ax.text(335, 480, 'NET', ha='right', va='bottom', color=MUTED, fontsize=9)
    for r in ds.A_win:
        c, m = ds.style(r)
        hx, hy = float(r['player_location_court_x']), float(r['player_location_court_y'])
        lx, ly = float(r['landing_court_x']), float(r['landing_court_y'])
        ax.annotate('', (lx, ly), (hx, hy), arrowprops=dict(arrowstyle='-', color=c, lw=1.4, alpha=.55), zorder=2)
        ax.scatter(hx, hy, s=90, facecolor=SURF, edgecolor=c, marker=m, lw=2, zorder=3)
        ax.scatter(lx, ly, s=90, color=c, marker=m, edgecolor=SURF, lw=1.5, zorder=4)
        ax.annotate(ds.tag(r), (lx, ly), xytext=(8, -3), textcoords='offset points', color=MUTED, fontsize=8)
    finish(ax, "Player A's winning shots: hit from and landed",
           f'{len(ds.A_win)} rally-winning shots, {ds.subtitle}. S = set, R = rally number.' if ds.multi
           else f'{len(ds.A_win)} rally-winning shots, {ds.subtitle}. R = rally number.',
           (825, 135), ds.win_legend() + [dot(MUTED, 'hit from', hollow=True), dot(MUTED, 'landed')], ncol=4 if not ds.multi else 3)
    fig.tight_layout(); fig.savefig(f'{OUT}/winning_shots_A_fullcourt{ds.suffix}.png', dpi=160, facecolor=SURF)


# ---------------------------------------------------------------- 3. hit locations of all A shots, one panel per shot type
# Rally-winning shots are orange. With several sets, set 1 is blue and set 2 aqua, and A's hit positions are rotated
# onto one half-court (A swaps ends between sets); net is the bold line at the bottom of each panel.
def hit_locations_by_type(ds):
    shots = [r for r in ds.A_rows if r['player_location_court_x']]
    win_ids = {id(r) for r in ds.A_win}  # same row objects as in A_rows
    pos = {id(r): flip_to_top(float(r['player_location_court_x']), float(r['player_location_court_y'])) for r in shots}
    n = collections.Counter(r['type'] for r in shots)
    types = [t for t, _ in n.most_common()]
    cols = 4; nrows = math.ceil(len(types) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(11, 3.1 * nrows), facecolor=SURF)
    fig.suptitle("Where Player A hit each shot type from", x=.01, ha='left', color=INK, fontsize=14, fontweight='bold')
    fig.text(.01, .955, f'{len(shots)} shots by A, {ds.subtitle}. One panel per shot type; grey = all A shots for context.',
             color=MUTED, fontsize=9)
    for ax, t in zip(axes.flat, types):
        ax.set_facecolor(SURF); court_lines(ax, full=False)
        ax.scatter(*zip(*pos.values()), s=10, color=CONTEXT, zorder=2)
        sel = [r for r in shots if r['type'] == t]
        for s in ds.sets:
            for won, color, z in ((False, SET_COLOR[s], 3), (True, WIN_COLOR, 4)):
                g = [r for r in sel if r['set'] == s and (id(r) in win_ids) == won]
                ax.scatter([pos[id(r)][0] for r in g], [pos[id(r)][1] for r in g], s=34, color=color,
                           marker=TYPE_MARKER[ds.sets.index(s)] if ds.multi else 'o',
                           edgecolor=SURF, lw=.8, alpha=.95, zorder=z)
        ax.set_xlim(5, 350); ax.set_ylim(492, 135); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(f'{t} ({n[t]})', loc='left', color=INK, fontsize=10, fontweight='bold')
    for ax in list(axes.flat)[len(types):]: ax.axis('off')
    nw = sum(id(r) in win_ids for r in shots)
    handles = []
    for i, s in enumerate(ds.sets):
        k = sum(r['set'] == s and id(r) not in win_ids for r in shots)
        handles.append(dot(SET_COLOR[s], f'set {s}, rally continued ({k})' if ds.multi else f'shot type, rally continued ({k})',
                           marker=TYPE_MARKER[i] if ds.multi else 'o'))
    handles += [dot(WIN_COLOR, f'rally-winning shot ({nw})'), dot(CONTEXT, 'all A shots')]
    fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(.99, .02), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(f'{OUT}/hit_locations_A_by_type{ds.suffix}.png', dpi=140, facecolor=SURF)


# ---------------------------------------------------------------- 4. how A won its points: own winners vs opponent errors
# A rally's last shot is the row with getpoint_player filled. A's winner = last shot hit by A and the point is A's;
# opponent error = last shot hit by B and the point is A's (out of bounds, net, did not cross the net, ...).
def winners_vs_opponent_errors(ds):
    ended = [r for r in ds.rows if r['getpoint_player']]
    groups = [(f'Set {s}', [r for r in ended if r['set'] == s]) for s in ds.sets]
    if ds.multi: groups.append(('All sets', ended))
    counts = [(sum(r['player'] == 'A' and r['getpoint_player'] == 'A' for r in g),
               sum(r['player'] == 'B' and r['getpoint_player'] == 'A' for r in g)) for _, g in groups]
    fig, ax = plt.subplots(figsize=(1.9 * len(groups) + 3, 5.4), facecolor=SURF); ax.set_facecolor(SURF)
    w = .36
    for i, (win, err) in enumerate(counts):
        for dx, v, c in ((-w / 2 - .02, win, WIN_COLOR), (w / 2 + .02, err, CATEGORICAL[0])):
            ax.bar(i + dx, v, w, color=c, zorder=3)
            ax.text(i + dx, v + .4, f'{v}\n{v / (win + err):.0%}', ha='center', va='bottom', color=INK, fontsize=9)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f'{n}\n{win + err} points' for (n, _), (win, err) in zip(groups, counts)], color=INK, fontsize=10)
    ax.set_ylim(0, max(max(c) for c in counts) * 1.2)
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(axis='y', colors=MUTED, length=0); ax.tick_params(axis='x', length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_ylabel('Points won by A', color=MUTED)
    ax.set_title('How Player A won points: own winners vs opponent errors', loc='left', color=INK, fontsize=13,
                 fontweight='bold', pad=26)
    ax.text(0, 1.012, f'{ds.subtitle}. Share = % of A\'s points in that group.', transform=ax.transAxes,
            color=MUTED, fontsize=8, va='bottom')
    ax.legend(handles=[dot(WIN_COLOR, 'winner played by A'), dot(CATEGORICAL[0], 'point from opponent error')],
              loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(f'{OUT}/winners_vs_opponent_errors{ds.suffix}.png', dpi=160, facecolor=SURF)
    return counts


# ---------------------------------------------------------------- 5. A's winners vs A's shot just before an opponent error
# Left: A's winners (last shot of the rally hit by A, point to A). Right: the shot A hit immediately before B's last shot,
# when B's last shot lost the point (ball_round - 1 of an opponent-error rally). Shots without coordinates are skipped.
def winner_vs_pre_error_shots(ds):
    by_key = {(r['set'], r['rally'], float(r['ball_round'])): r for r in ds.rows}
    winners = [r for r in ds.A_rows if r['getpoint_player'] == 'A']
    pre_error = [by_key[(r['set'], r['rally'], float(r['ball_round']) - 1)] for r in ds.rows
                 if r['player'] == 'B' and r['getpoint_player'] == 'A']
    groups = [('Winners played by A', winners), ('Shot before the opponent\'s error', pre_error)]
    n = collections.Counter(r['type'] for _, g in groups for r in g)
    top = [t for t, _ in n.most_common(3)]  # 3 colours: the most the palette keeps apart in a scatter
    color = lambda t: CATEGORICAL[top.index(t)] if t in top else '#9a9a94'
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.6), facecolor=SURF)
    fig.suptitle("Player A's winners vs the shot before the opponent's error", x=.01, ha='left', color=INK,
                 fontsize=14, fontweight='bold')
    fig.text(.01, .945, f'{ds.subtitle}. Positions are true court positions; shots without coordinates are left out.',
             color=MUTED, fontsize=9)
    for ax, (name, g) in zip(axes, groups):
        ax.set_facecolor(SURF); court_lines(ax)
        ax.text(335, 480, 'NET', ha='right', va='bottom', color=MUTED, fontsize=8)
        g = [r for r in g if r['landing_court_x'] and r['player_location_court_x']]
        for r in g:
            c = color(r['type'])
            hx, hy = float(r['player_location_court_x']), float(r['player_location_court_y'])
            lx, ly = float(r['landing_court_x']), float(r['landing_court_y'])
            ax.annotate('', (lx, ly), (hx, hy), arrowprops=dict(arrowstyle='-', color=c, lw=1.3, alpha=.5), zorder=2)
            ax.scatter(hx, hy, s=60, facecolor=SURF, edgecolor=c, lw=1.8, zorder=3)
            ax.scatter(lx, ly, s=60, color=c, edgecolor=SURF, lw=1.2, zorder=4)
        ax.set_xlim(5, 350); ax.set_ylim(825, 135); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(f'{name} ({len(g)})', loc='left', color=INK, fontsize=11, fontweight='bold')
    other = sum(v for t, v in n.items() if t not in top)
    handles = [dot(color(t), t) for t in top] + ([dot('#9a9a94', 'other types')] if other else []) + \
              [dot(MUTED, 'hit from', hollow=True), dot(MUTED, 'landed')]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .93)); fig.savefig(f'{OUT}/winner_vs_pre_error_shots{ds.suffix}.png', dpi=150, facecolor=SURF)
    return [(name, len(g), len([r for r in g if r['landing_court_x'] and r['player_location_court_x']])) for name, g in groups]


# ---------------------------------------------------------------- 6. all A shots by shot type, winners and pre-error shots marked
# Same layout as section 3 (hit-from position, A's ends rotated onto one half). Orange = A's winner, blue = the shot
# A hit just before the opponent's error (same colours as section 4), dark grey = the type's other shots.
def hit_locations_by_type_marked(ds):
    by_key = {(r['set'], r['rally'], float(r['ball_round'])): r for r in ds.rows}
    pre_ids = {id(by_key[(r['set'], r['rally'], float(r['ball_round']) - 1)]) for r in ds.rows
               if r['player'] == 'B' and r['getpoint_player'] == 'A'}
    win_ids = {id(r) for r in ds.A_rows if r['getpoint_player'] == 'A'}
    shots = [r for r in ds.A_rows if r['player_location_court_x']]
    pos = {id(r): flip_to_top(float(r['player_location_court_x']), float(r['player_location_court_y'])) for r in shots}
    n = collections.Counter(r['type'] for r in shots)
    types = [t for t, _ in n.most_common()]
    cols = 4; nrows = math.ceil(len(types) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(11, 3.1 * nrows), facecolor=SURF)
    fig.suptitle("Where Player A hit each shot type from: winners and shots before an opponent error", x=.01, ha='left',
                 color=INK, fontsize=14, fontweight='bold')
    fig.text(.01, .955, f'{len(shots)} shots by A, {ds.subtitle}. One panel per shot type; light grey = all A shots for context.',
             color=MUTED, fontsize=9)
    layers = (('other', '#6b6a65', 3), ('pre', CATEGORICAL[0], 4), ('win', WIN_COLOR, 5))
    kind = lambda r: 'win' if id(r) in win_ids else 'pre' if id(r) in pre_ids else 'other'
    for ax, t in zip(axes.flat, types):
        ax.set_facecolor(SURF); court_lines(ax, full=False)
        ax.scatter(*zip(*pos.values()), s=10, color=CONTEXT, zorder=2)
        sel = [r for r in shots if r['type'] == t]
        for k, color, z in layers:
            g = [r for r in sel if kind(r) == k]
            ax.scatter([pos[id(r)][0] for r in g], [pos[id(r)][1] for r in g], s=34, color=color,
                       edgecolor=SURF, lw=.8, alpha=.95, zorder=z)
        ax.set_xlim(5, 350); ax.set_ylim(492, 135); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(f'{t} ({n[t]})', loc='left', color=INK, fontsize=10, fontweight='bold')
    for ax in list(axes.flat)[len(types):]: ax.axis('off')
    cnt = collections.Counter(kind(r) for r in shots)
    fig.legend(handles=[dot('#6b6a65', f"other shots of the type ({cnt['other']})"),
                        dot(CATEGORICAL[0], f"shot before opponent error ({cnt['pre']})"),
                        dot(WIN_COLOR, f"A's winner ({cnt['win']})"), dot(CONTEXT, 'all A shots')],
               loc='lower right', bbox_to_anchor=(.99, .02), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(f'{OUT}/hit_locations_A_by_type_marked{ds.suffix}.png', dpi=140, facecolor=SURF)
    return cnt


# ---------------------------------------------------------------- 7. where A hit from on the points A lost
# Points A lost = rallies whose point went to B. Left: A's own error (last shot hit by A). Right: the shot A hit just
# before B's winning last shot (ball_round - 1). Hit-from positions only (true court positions); shots without
# coordinates are skipped.
def lost_point_shots(ds):
    by_key = {(r['set'], r['rally'], float(r['ball_round'])): r for r in ds.rows}
    errors = [r for r in ds.A_rows if r['getpoint_player'] == 'B']
    pre = [by_key[(r['set'], r['rally'], float(r['ball_round']) - 1)] for r in ds.rows
           if r['player'] == 'B' and r['getpoint_player'] == 'B']
    assert all(r['player'] == 'A' for r in pre)
    groups = [("A's own error (last shot of the rally)", errors), ("Shot before the opponent's winner", pre)]
    n_all = [len(g) for _, g in groups]
    groups = [(name, [r for r in g if r['player_location_court_x']]) for name, g in groups]
    n = collections.Counter(r['type'] for _, g in groups for r in g)  # counts of the plotted shots only
    top = [t for t, _ in n.most_common(3)]
    color = lambda t: CATEGORICAL[top.index(t)] if t in top else '#9a9a94'
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.2), facecolor=SURF)
    fig.suptitle("Where Player A hit from on the points A lost", x=.01, ha='left', color=INK, fontsize=14, fontweight='bold')
    fig.text(.01, .945, f'{ds.subtitle}. Dot = where A stood when hitting; shots without coordinates are left out.',
             color=MUTED, fontsize=9)
    for ax, (name, g) in zip(axes, groups):
        ax.set_facecolor(SURF); court_lines(ax)
        ax.text(335, 480, 'NET', ha='right', va='bottom', color=MUTED, fontsize=8)
        ax.scatter([float(r['player_location_court_x']) for r in g], [float(r['player_location_court_y']) for r in g],
                   s=70, c=[color(r['type']) for r in g], edgecolor=SURF, lw=1.2, zorder=3)
        ax.set_xlim(5, 350); ax.set_ylim(825, 135); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(f'{name} ({len(g)})', loc='left', color=INK, fontsize=11, fontweight='bold')
    other = sum(v for t, v in n.items() if t not in top)
    fig.legend(handles=[dot(color(t), f'{t} ({n[t]})') for t in top] + ([dot('#9a9a94', f'other types ({other})')] if other else []),
               loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .93)); fig.savefig(f'{OUT}/lost_point_shots{ds.suffix}.png', dpi=150, facecolor=SURF)
    return [(name, len(g), k) for (name, g), k in zip(groups, n_all)], n


# ---------------------------------------------------------------- 8. all A shots by shot type, lost-point shots marked
# Same layout as sections 3 and 6 (hit-from position, A's ends rotated onto one half). Magenta diamond = A's own error
# (last shot of a rally B won), violet triangle = the shot A hit just before B's winner, dark grey = the type's other shots.
# Different colours from section 6 on purpose: orange/blue there mean A's winner / shot before B's error.
ERR_COLOR, PRE_LOST_COLOR = '#e87ba4', '#6250d6'


def hit_locations_by_type_lost(ds):
    by_key = {(r['set'], r['rally'], float(r['ball_round'])): r for r in ds.rows}
    pre_ids = {id(by_key[(r['set'], r['rally'], float(r['ball_round']) - 1)]) for r in ds.rows
               if r['player'] == 'B' and r['getpoint_player'] == 'B'}
    err_ids = {id(r) for r in ds.A_rows if r['getpoint_player'] == 'B'}
    shots = [r for r in ds.A_rows if r['player_location_court_x']]
    pos = {id(r): flip_to_top(float(r['player_location_court_x']), float(r['player_location_court_y'])) for r in shots}
    n = collections.Counter(r['type'] for r in shots)
    types = [t for t, _ in n.most_common()]
    cols = 4; nrows = math.ceil(len(types) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(11, 3.1 * nrows), facecolor=SURF)
    fig.suptitle("Where Player A hit each shot type from: A's errors and shots before an opponent winner", x=.01,
                 ha='left', color=INK, fontsize=14, fontweight='bold')
    fig.text(.01, .955, f'{len(shots)} shots by A, {ds.subtitle}. One panel per shot type; light grey = all A shots for context.',
             color=MUTED, fontsize=9)
    layers = (('other', '#6b6a65', 'o', 3), ('pre', PRE_LOST_COLOR, '^', 4), ('err', ERR_COLOR, 'D', 5))
    kind = lambda r: 'err' if id(r) in err_ids else 'pre' if id(r) in pre_ids else 'other'
    for ax, t in zip(axes.flat, types):
        ax.set_facecolor(SURF); court_lines(ax, full=False)
        ax.scatter(*zip(*pos.values()), s=10, color=CONTEXT, zorder=2)
        sel = [r for r in shots if r['type'] == t]
        for k, color, marker, z in layers:
            g = [r for r in sel if kind(r) == k]
            ax.scatter([pos[id(r)][0] for r in g], [pos[id(r)][1] for r in g], s=34 if k == 'other' else 46, color=color,
                       marker=marker, edgecolor=SURF, lw=.8, alpha=.95, zorder=z)
        ax.set_xlim(5, 350); ax.set_ylim(492, 135); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(f'{t} ({n[t]})', loc='left', color=INK, fontsize=10, fontweight='bold')
    for ax in list(axes.flat)[len(types):]: ax.axis('off')
    cnt = collections.Counter(kind(r) for r in shots)
    fig.legend(handles=[dot('#6b6a65', f"other shots of the type ({cnt['other']})"),
                        dot(PRE_LOST_COLOR, f"shot before opponent's winner ({cnt['pre']})", marker='^'),
                        dot(ERR_COLOR, f"A's own error ({cnt['err']})", marker='D'), dot(CONTEXT, 'all A shots')],
               loc='lower right', bbox_to_anchor=(.99, .02), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(f'{OUT}/hit_locations_A_by_type_lost{ds.suffix}.png', dpi=140, facecolor=SURF)
    return cnt


# ---------------------------------------------------------------- 9. reach features d and t for A's smashes (smash + wrist smash)
# Court coordinates are converted to metres (49.2 court units per metre).
#   d      = distance (m) from the opponent's position at the hit to the line of flight (the infinite line through the
#            hit point and the landing point). d_alt starts the line at A's annotated position instead of the hit point.
#   v_obs  = shuttle speed (m/s) = hit-to-landing distance / time to the opponent's next stroke (frame_num diff / 30 fps).
#            Only smashes that were returned have a next stroke; v_med is the median of v_obs.
#   t_obs  = observed time to the next stroke (s); t_hat = (hit point to opponent distance) / v_med, known at hit time.
# Outcome: winner = A's last shot and A's point; own error = A's last shot, B's point; otherwise returned, split by who
# won the rally.
M = 49.2
OUTCOMES = [('winner', WIN_COLOR), ('returned, A won rally', CATEGORICAL[0]), ('returned, A lost rally', '#6b6a65'),
            ('own error', '#e87ba4')]


def smash_reach(ds):
    by_key = {(r['set'], r['rally'], float(r['ball_round'])): r for r in ds.rows}
    rally_winner = {(r['set'], r['rally']): r['getpoint_player'] for r in ds.rows if r['getpoint_player']}
    out = []
    for r in ds.A_rows:
        if r['type'] not in ('smash', 'wrist smash'): continue
        P = lambda a, b: (float(r[a]) / 1, float(r[b]) / 1)
        hit, land = P('hit_court_x', 'hit_court_y'), P('landing_court_x', 'landing_court_y')
        opp, me = P('opponent_location_court_x', 'opponent_location_court_y'), P('player_location_court_x', 'player_location_court_y')
        dist = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]) / M
        def to_line(q, a, b):  # distance from q to the infinite line through a and b
            return abs((b[0] - a[0]) * (a[1] - q[1]) - (a[0] - q[0]) * (b[1] - a[1])) / math.hypot(b[0] - a[0], b[1] - a[1]) / M
        nxt = by_key.get((r['set'], r['rally'], float(r['ball_round']) + 1))
        t_obs = (float(nxt['frame_num']) - float(r['frame_num'])) / 30 if nxt else None
        won = rally_winner[(r['set'], r['rally'])]
        outcome = ('winner' if won == 'A' else 'own error') if not nxt else \
                  ('returned, A won rally' if won == 'A' else 'returned, A lost rally')
        out.append(dict(set=r['set'], rally=r['rally'], ball_round=int(float(r['ball_round'])), type=r['type'], outcome=outcome,
                        flight_m=dist(hit, land), hit_to_opp_m=dist(hit, opp), opp_to_landing_m=dist(opp, land),
                        d_m=to_line(opp, hit, land), d_alt_m=to_line(opp, me, land), t_obs_s=t_obs,
                        v_obs_mps=dist(hit, land) / t_obs if t_obs else None, hit=hit, land=land, opp=opp))
    v_med = statistics.median(o['v_obs_mps'] for o in out if o['v_obs_mps'])
    for o in out: o['t_hat_s'] = o['hit_to_opp_m'] / v_med
    cols = ['set', 'rally', 'ball_round', 'type', 'outcome', 'flight_m', 'hit_to_opp_m', 'opp_to_landing_m', 'd_m', 'd_alt_m',
            'v_obs_mps', 't_obs_s', 't_hat_s']
    with open(f'{OUT}/smash_reach_A{ds.suffix}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(cols)
        for o in out: w.writerow([f'{o[c]:.2f}' if isinstance(o[c], float) else ('' if o[c] is None else o[c]) for c in cols])

    def rot(o):  # rotate 180 deg when A hits from the bottom half, so A is always at the top of the court
        f = (lambda q: (354.99 - q[0], 960 - q[1])) if o['hit'][1] > 480 else (lambda q: q)
        return f(o['hit']), f(o['land']), f(o['opp'])
    color = dict(OUTCOMES)
    handles = [dot(c, f"{n} ({sum(o['outcome'] == n for o in out)})") for n, c in OUTCOMES]

    # (a) every smash on one court: flight line, A's hit point (filled), opponent's position (hollow square)
    fig, ax = plt.subplots(figsize=(8, 9.4), facecolor=SURF); ax.set_facecolor(SURF); court_lines(ax)
    for o in out:
        h, l, q = rot(o); c = color[o['outcome']]
        ax.plot([h[0], l[0]], [h[1], l[1]], color=c, lw=1.4, alpha=.6, zorder=2)
        ax.scatter(*h, s=34, color=c, edgecolor=SURF, lw=.8, zorder=4)
        ax.scatter(*q, s=60, marker='s', facecolor=SURF, edgecolor=c, lw=1.8, zorder=3)
    ax.text(335, 480, 'NET', ha='right', va='bottom', color=MUTED, fontsize=8)
    finish(ax, "A's smashes: hit point, flight line, opponent position",
           f'{len(out)} smashes and wrist smashes, {ds.subtitle}. Dot = hit point, square = opponent at the hit.',
           (825, 135), handles, ncol=2)
    fig.tight_layout(); fig.savefig(f'{OUT}/smash_reach_court{ds.suffix}.png', dpi=150, facecolor=SURF)

    # (b) d against t_hat
    fig, ax = plt.subplots(figsize=(7.5, 5.8), facecolor=SURF); ax.set_facecolor(SURF)
    for o in out:
        ax.scatter(o['t_hat_s'], o['d_m'], s=70, color=color[o['outcome']], edgecolor=SURF, lw=1,
                   marker='o' if o['type'] == 'smash' else '^', zorder=3)
    ax.yaxis.grid(True, color=GRID, lw=1, zorder=0); ax.set_axisbelow(True)
    ax.set_xlabel('t_hat: expected shuttle time to the opponent (s)', color=MUTED)
    ax.set_ylabel("d: opponent's distance to the line of flight (m)", color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title('Reach features of A\'s smashes: d against t', loc='left', color=INK, fontsize=13, fontweight='bold', pad=26)
    ax.text(0, 1.012, f'{len(out)} smashes, {ds.subtitle}. Circle = smash, triangle = wrist smash.', transform=ax.transAxes,
            color=MUTED, fontsize=8, va='bottom')
    leg = ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.14), ncol=2, frameon=False, fontsize=9)
    for t in leg.get_texts(): t.set_color(INK)
    fig.tight_layout(); fig.savefig(f'{OUT}/smash_reach_d_vs_t{ds.suffix}.png', dpi=150, facecolor=SURF)

    # (c) one mini court per smash: the perpendicular from the opponent to the line of flight is d
    cols_n = 6; rows_n = math.ceil(len(out) / cols_n)
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(13, 3.6 * rows_n), facecolor=SURF)
    fig.suptitle("Each of A's smashes: the dotted line is d, the opponent's distance to the line of flight", x=.01, ha='left',
                 color=INK, fontsize=14, fontweight='bold')
    for ax, o in zip(axes.flat, sorted(out, key=lambda o: o['d_m'])):
        ax.set_facecolor(SURF); court_lines(ax)
        h, l, q = rot(o); c = color[o['outcome']]
        (hx, hy), (lx, ly), (qx, qy) = h, l, q
        k = ((qx - hx) * (lx - hx) + (qy - hy) * (ly - hy)) / ((lx - hx) ** 2 + (ly - hy) ** 2)
        foot = (hx + k * (lx - hx), hy + k * (ly - hy))  # closest point on the (infinite) line
        ax.plot([hx - 3 * (lx - hx), hx + 3 * (lx - hx)], [hy - 3 * (ly - hy), hy + 3 * (ly - hy)], color=c, lw=.8, alpha=.35, zorder=2)
        ax.plot([hx, lx], [hy, ly], color=c, lw=1.6, zorder=3)
        ax.plot([qx, foot[0]], [qy, foot[1]], color=INK, lw=1, ls=':', zorder=3)
        ax.scatter(hx, hy, s=26, color=c, edgecolor=SURF, lw=.6, zorder=5)
        ax.scatter(lx, ly, s=26, marker='x', color=c, lw=1.4, zorder=5)
        ax.scatter(qx, qy, s=40, marker='s', facecolor=SURF, edgecolor=INK, lw=1.4, zorder=4)
        ax.set_xlim(5, 350); ax.set_ylim(825, 135); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        tobs = f"{o['t_obs_s']:.1f}" if o['t_obs_s'] is not None else '-'
        ax.set_title(f"S{o['set']} R{o['rally']} {'WS ' if o['type'] == 'wrist smash' else ''}\nd {o['d_m']:.1f} m, t {o['t_hat_s']:.2f} s (obs {tobs})",
                     loc='left', color=INK, fontsize=8, fontweight='bold')
    for ax in list(axes.flat)[len(out):]: ax.axis('off')
    fig.legend(handles=handles + [dot(MUTED, 'hit point'), Line2D([], [], marker='x', ls='', color=MUTED, label='landing'),
                                  Line2D([], [], marker='s', ls='', markerfacecolor=SURF, markeredgecolor=INK, label='opponent')],
               loc='lower center', ncol=7, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .96)); fig.savefig(f'{OUT}/smash_reach_each{ds.suffix}.png', dpi=130, facecolor=SURF)
    return out, v_med


# ---------------------------------------------------------------- 10. observed stroke-to-stroke time, and how far the opponent really moved
# (a) Time from a stroke to the next stroke of the rally (frame_num difference / 30 fps), for both players, per shot type.
#     Types with fewer than MIN_N strokes are drawn hollow: too few to trust (feedback point 6).
# (b) For A's returned smashes: the opponent's position at A's hit -> the same player's position at their next hit. Compare
#     the distance actually moved with d (distance to the line of flight) and with the distance to the landing point.
MIN_N = 10


def observed_time_and_opponent_movement(ds):
    by_key = {(r['set'], r['rally'], float(r['ball_round'])): r for r in ds.rows}
    xy = lambda r, a, b: (float(r[a]), float(r[b]))
    dist = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]) / M
    gaps = collections.defaultdict(list)  # shot type -> [(seconds to next stroke, hit-to-landing metres)]
    for r in ds.rows:
        nxt = by_key.get((r['set'], r['rally'], float(r['ball_round']) + 1))
        if not nxt or not r['hit_court_x'] or not r['landing_court_x']: continue
        dt = (float(nxt['frame_num']) - float(r['frame_num'])) / 30
        if dt > 0: gaps[r['type']].append((dt, dist(xy(r, 'hit_court_x', 'hit_court_y'), xy(r, 'landing_court_x', 'landing_court_y'))))
    q = lambda v, i: statistics.quantiles(v, n=4)[i] if len(v) > 3 else statistics.median(v)
    table = []
    for t, g in gaps.items():
        v = [a for a, _ in g]
        table.append((t, len(v), statistics.median(v), q(v, 0), q(v, 2), min(v), max(v), statistics.median(b for _, b in g)))
    table.sort(key=lambda row: row[2])
    with open(f'{OUT}/stroke_time_by_type{ds.suffix}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['type', 'n', 'median_s', 'q1_s', 'q3_s', 'min_s', 'max_s', 'median_distance_m'])
        w.writerows([[t, n_] + [f'{x:.2f}' for x in rest] for t, n_, *rest in table])
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(table) + 2.2), facecolor=SURF); ax.set_facecolor(SURF)
    for i, (t, n_, med, q1, q3, lo, hi, _) in enumerate(table):
        ok = n_ >= MIN_N
        ax.plot([lo, hi], [i, i], color=GRID, lw=1.5, zorder=2)
        ax.plot([q1, q3], [i, i], color=CATEGORICAL[0], lw=5, alpha=.9 if ok else .35, solid_capstyle='butt', zorder=3)
        ax.scatter(med, i, s=60, facecolor=CATEGORICAL[0] if ok else SURF, edgecolor=CATEGORICAL[0], lw=1.8, zorder=4)
    ax.set_yticks(range(len(table))); ax.set_yticklabels([f'{t} (n={n_})' for t, n_, *_ in table], color=INK, fontsize=9)
    ax.xaxis.grid(True, color=GRID, lw=1, zorder=0); ax.set_axisbelow(True); ax.tick_params(colors=MUTED, length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xlabel('time from the stroke to the next stroke (s)', color=MUTED)
    fig.suptitle('How long until the next stroke, by shot type', x=.01, ha='left', color=INK, fontsize=13, fontweight='bold')
    fig.text(.01, .93, f'{ds.subtitle}, both players. Dot = median, bar = middle half, line = min to max;\nhollow = fewer than {MIN_N} strokes.',
             color=MUTED, fontsize=8, va='top')
    fig.tight_layout(rect=(0, 0, 1, .9)); fig.savefig(f'{OUT}/stroke_time_by_type{ds.suffix}.png', dpi=150, facecolor=SURF)

    smashes = []
    for r in ds.A_rows:
        if r['type'] not in ('smash', 'wrist smash'): continue
        nxt = by_key.get((r['set'], r['rally'], float(r['ball_round']) + 1))
        if not nxt or not nxt['player_location_court_x']: continue
        hit, land = xy(r, 'hit_court_x', 'hit_court_y'), xy(r, 'landing_court_x', 'landing_court_y')
        before, after = xy(r, 'opponent_location_court_x', 'opponent_location_court_y'), xy(nxt, 'player_location_court_x', 'player_location_court_y')
        d_line = abs((land[0] - hit[0]) * (hit[1] - before[1]) - (hit[0] - before[0]) * (land[1] - hit[1])) / math.hypot(land[0] - hit[0], land[1] - hit[1]) / M
        dt = (float(nxt['frame_num']) - float(r['frame_num'])) / 30
        outcome = 'returned, A won rally' if next(x for x in ds.rows if x['set'] == r['set'] and x['rally'] == r['rally'] and x['getpoint_player'])['getpoint_player'] == 'A' else 'returned, A lost rally'
        smashes.append(dict(r=r, hit=hit, land=land, before=before, after=after, moved=dist(before, after), d=d_line,
                            to_land=dist(before, land), dt=dt, outcome=outcome, next_vs_land=dist(after, land)))
    color = {n: c for n, c in OUTCOMES}
    # court view: opponent's position at the hit (hollow square) -> their position at the next hit (filled), inside A's flight
    fig, ax = plt.subplots(figsize=(8, 9.4), facecolor=SURF); ax.set_facecolor(SURF); court_lines(ax)
    ax.text(335, 480, 'NET', ha='right', va='bottom', color=MUTED, fontsize=8)
    for o in smashes:
        flip = (lambda p: (354.99 - p[0], 960 - p[1])) if o['hit'][1] > 480 else (lambda p: p)
        h, l, b, a = flip(o['hit']), flip(o['land']), flip(o['before']), flip(o['after']); c = color[o['outcome']]
        ax.plot([h[0], l[0]], [h[1], l[1]], color=c, lw=1, alpha=.35, zorder=2)
        ax.annotate('', a, b, arrowprops=dict(arrowstyle='-|>', color=c, lw=1.6, shrinkA=0, shrinkB=0), zorder=4)
        ax.scatter(*b, s=44, marker='s', facecolor=SURF, edgecolor=c, lw=1.6, zorder=5)
    finish(ax, "How far the opponent moved after A's smash",
           f"{len(smashes)} returned smashes, {ds.subtitle}.\nHollow square = opponent at A's hit; arrow = to their next hit; thin line = flight.",
           (825, 135), [dot(c, n) for n, c in OUTCOMES if n.startswith('returned')], ncol=2)
    fig.tight_layout(); fig.savefig(f'{OUT}/smash_opponent_movement_court{ds.suffix}.png', dpi=150, facecolor=SURF)
    # moved distance against d and against the distance to the landing point
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), facecolor=SURF)
    for ax, key, label in ((axes[0], 'd', 'd: opponent to the line of flight (m)'), (axes[1], 'to_land', 'opponent to the landing point (m)')):
        ax.set_facecolor(SURF)
        for o in smashes:
            ax.scatter(o[key], o['moved'], s=60, color=color[o['outcome']], edgecolor=SURF, lw=1, zorder=3,
                       marker='o' if o['r']['type'] == 'smash' else '^')
        top = max(max(o[key], o['moved']) for o in smashes) * 1.08
        ax.plot([0, top], [0, top], color=MUTED, lw=1, ls='--', zorder=2)
        ax.set_xlim(0, top); ax.set_ylim(0, top)
        ax.yaxis.grid(True, color=GRID, lw=1, zorder=0); ax.set_axisbelow(True); ax.tick_params(colors=MUTED, length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xlabel(label, color=MUTED); ax.set_ylabel('distance the opponent actually moved (m)', color=MUTED)
    fig.suptitle("Did the opponent move as far as the smash required?", x=.01, ha='left', color=INK, fontsize=13, fontweight='bold')
    fig.text(.01, .925, f'{len(smashes)} returned smashes, {ds.subtitle}. Dashed line: moved distance = x. Circle = smash, triangle = wrist smash.',
             color=MUTED, fontsize=8)
    fig.legend(handles=[dot(c, n) for n, c in OUTCOMES if n.startswith('returned')], loc='lower center', ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .92)); fig.savefig(f'{OUT}/smash_opponent_movement_vs_needed{ds.suffix}.png', dpi=150, facecolor=SURF)
    return table, smashes


# ---------------------------------------------------------------- run all sections on every dataset
if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for spec in DATASETS:
        ds = Dataset(*spec)
        winning_landings_half_court(ds)
        winning_hit_and_landing_full_court(ds)
        hit_locations_by_type(ds)
        if ds.multi: print(winner_vs_pre_error_shots(ds)), print(hit_locations_by_type_marked(ds)), print(lost_point_shots(ds)), print(hit_locations_by_type_lost(ds))
        if ds.multi:
            smash_out, v_med = smash_reach(ds); print('smash reach: n =', len(smash_out), 'median shuttle speed m/s =', round(v_med, 1))
            stroke_table, moved = observed_time_and_opponent_movement(ds)
            for row in stroke_table: print('  ', row[0], row[1], 'median s', round(row[2], 2), 'IQR', round(row[3], 2), round(row[4], 2))
            mv = [o['moved'] for o in moved]; print('  moved m median/IQR', round(statistics.median(mv), 2), [round(x, 2) for x in statistics.quantiles(mv, n=4)], 'n', len(mv), 'corr moved~d', round(sum((a-statistics.mean(mv))*(b-statistics.mean(o['d'] for o in moved)) for a, b in zip(mv, [o['d'] for o in moved]))/math.sqrt(sum((a-statistics.mean(mv))**2 for a in mv)*sum((o['d']-statistics.mean(o['d'] for o in moved))**2 for o in moved)), 2), 'median next_vs_land', round(statistics.median(o['next_vs_land'] for o in moved), 2))
        print(ds.suffix or 'set1', 'winners vs opponent errors:', winners_vs_opponent_errors(ds))
    print('saved:', sorted(f for f in os.listdir(OUT) if f.endswith('.png')))
