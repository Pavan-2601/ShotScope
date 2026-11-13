"""Build the player-attribute tables (height, handedness, birth date) for each dataset, kept separate:

  data/external/player_attributes.csv               the 27 ShuttleSet players (training data)
  data/external/player_attributes_shuttleset22.csv  the 35 players in the ShuttleSet22 test matches

Run after clean_shuttleset.py:  python cleaning/build_player_attributes.py

Values were collected on 2026-11-05 from two public sources per player and are typed in below with their provenance:
  * Wikipedia infobox (read directly).
  * The official BWF profile (bwfbadminton.com/player/<id>), read through a web-search summary: the site returns HTTP 403
    to direct requests. BWF dates are written DD/MM/YYYY. BWF lists a height for only some players, and a height of 0.00
    on a BWF page is a placeholder, treated here as missing. For Kento Momota no BWF page was found; his second source is
    his Olympics.com athlete profile.
Where the two sources disagree, both values are kept and flagged. Nothing here comes from match results, so none of it
can leak the outcome. The 19 players who appear in both datasets carry identical values in both tables.
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / 'data' / 'external'

# key (lower-case player name as in players.csv):
#   (wikipedia page, wiki dob, wiki height, wiki hand, other source, other dob, other height, other hand, other id/url)
R, L, N = 'right', 'left', None
BWF = 'BWF'
D = {
    'an se young': ('An_Se-young', '2002-02-05', 170, R, BWF, '2002-02-05', N, R, 87442),
    'anders antonsen': ('Anders_Antonsen', '1997-04-27', 185, R, BWF, '1997-04-27', N, R, 91554),
    'anthony sinisuka ginting': ('Anthony_Sinisuka_Ginting', '1996-10-20', 171, R, BWF, '1996-05-11', 171, R, 95661),
    'busanan ongbamrungphan': ('Busanan_Ongbamrungphan', '1996-03-22', 168, R, BWF, '1996-03-22', 168, R, 58271),
    'carolina marin': ('Carolina_Mar%C3%ADn', '1993-06-15', 172, L, BWF, '1993-06-15', N, L, 18228),
    'chen long': ('Chen_Long_(badminton)', '1989-01-18', 187, R, BWF, '1989-01-18', N, R, 75787),
    'chou tien chen': ('Chou_Tien-chen', '1990-01-08', 180, R, BWF, '1990-01-08', N, R, 34810),
    'evgeniya kosetskaya': ('Evgeniya_Kosetskaya', '1994-12-16', 176, R, BWF, '1994-11-16', 176, R, 92400),
    'hans-kristian solberg vittinghus': ('Hans-Kristian_Vittinghus', '1986-01-16', 180, R, BWF, '1986-01-16', N, R, 53753),
    'jonatan christie': ('Jonatan_Christie', '1997-09-15', 179, R, BWF, '1997-09-15', N, R, 73442),
    'kento momota': ('Kento_Momota', '1994-09-01', 175, L, 'Olympics.com', '1994-09-01', 175, L,
                     'https://www.olympics.com/en/athletes/kento-momota'),
    'kidambi srikanth': ('Srikanth_Kidambi', '1993-02-07', 178, R, BWF, '1993-02-07', N, R, 93627),
    'lee cheuk yiu': ('Lee_Cheuk_Yiu', '1996-08-28', 178, R, BWF, '1996-08-28', N, R, 94780),
    'lee zii jia': ('Lee_Zii_Jia', '1998-03-29', 186, N, BWF, '1998-03-29', N, R, 81561),
    'liew daren': ('Liew_Daren', '1987-08-06', 178, R, BWF, '1987-08-06', N, R, 52864),
    'mia blichfeldt': ('Mia_Blichfeldt', '1997-08-19', 172, R, BWF, '1997-08-19', N, R, 91224),
    'michelle li': ('Michelle_Li_(badminton)', '1991-11-03', 173, R, BWF, '1991-11-03', N, R, 84523),
    'neslihan yigit': ('Neslihan_Yi%C4%9Fit', '1994-02-26', 180, R, BWF, '1994-02-26', N, R, 87885),
    'ng ka long angus': ('Ng_Ka_Long', '1994-06-24', 181, R, BWF, '1994-06-24', N, R, 45171),
    'pornpawee chochuwong': ('Pornpawee_Chochuwong', '1998-01-22', 170, R, BWF, '1998-01-22', 170, R, 84062),
    'pusarla v. sindhu': ('P._V._Sindhu', '1995-07-05', 179, R, BWF, '1995-07-05', N, R, 73173),
    'rasmus gemke': ('Rasmus_Gemke', '1997-01-11', N, R, BWF, '1997-01-11', 185, R, 62855),
    'ratchanok intanon': ('Ratchanok_Intanon', '1995-02-05', 169, R, BWF, '1995-02-05', N, R, 35642),
    'sameer verma': ('Sameer_Verma', '1994-10-22', 170, R, BWF, '1994-10-22', N, R, 77003),
    'shi yuqi': ('Shi_Yuqi', '1996-02-28', 184, R, BWF, '1996-02-28', N, R, 57945),
    'supanida katethong': ('Supanida_Katethong', '1997-10-26', 163, L, BWF, '1997-10-26', 163, L, 91422),
    'viktor axelsen': ('Viktor_Axelsen', '1994-01-04', 194, R, BWF, '1994-01-04', 194, R, 25831),
}

# The 16 players who appear in the ShuttleSet22 test matches but not in ShuttleSet (keys: norm_name of the ShuttleSet22 spelling).
# BWF gives no usable height for any of them (missing, or a 0.00 placeholder), so every height here is Wikipedia only.
# Aakarshi Kashyap: BWF birth date blank. Chico Aura Dwi Wardoyo and Zhao Jun Peng: BWF shows height 0.00 and no handedness.
# Wang Zhi Yi: Wikipedia infobox not read (birth date taken from the article text), no height or handedness found anywhere.
D22_NEW = {
    'aakarshi kashyap': ('Aakarshi_Kashyap', '2001-08-24', 159, R, BWF, N, N, N, 80402),
    'akane yamaguchi': ('Akane_Yamaguchi', '1997-06-06', 156, R, BWF, '1997-06-06', N, R, 96312),
    'chen yufei': ('Chen_Yufei', '1998-03-01', 171, R, BWF, '1998-03-01', N, R, 78778),
    'chico aura dwi wardoyo': ('Chico_Aura_Dwi_Wardoyo', '1998-06-15', 183, R, BWF, '1998-06-15', N, N, 71408),
    'gregoria mariska tunjung': ('Gregoria_Mariska_Tunjung', '1999-08-11', 164, R, BWF, '1999-08-11', N, R, 83802),
    'he bingjiao': ('He_Bingjiao', '1997-03-21', 169, L, BWF, '1997-03-21', N, L, 87434),
    'kenta nishimoto': ('Kenta_Nishimoto', '1994-08-30', 180, R, BWF, '1994-08-30', N, R, 85394),
    'kodai naraoka': ('Kodai_Naraoka', '2001-06-30', 173, R, BWF, '2001-06-30', N, R, 62063),
    'kunlavut vitidsarn': ('Kunlavut_Vitidsarn', '2001-05-11', 177, R, BWF, '2001-05-11', N, R, 64032),
    'li shi feng': ('Li_Shifeng', '2000-01-09', 185, R, BWF, '2000-01-09', N, R, 95476),
    'loh kean yew': ('Loh_Kean_Yew', '1997-06-26', 175, R, BWF, '1997-06-26', N, R, 76115),
    'lu guang zu': ('Lu_Guangzu', '1996-10-19', 178, R, BWF, '1996-10-19', N, R, 77427),
    'lakshya sen': ('Lakshya_Sen', '2001-08-16', 180, R, BWF, '2001-08-16', N, R, 68870),
    'prannoy h s': ('H._S._Prannoy', '1992-07-17', 179, R, BWF, '1992-07-17', N, R, 83950),
    'wang zhi yi': ('Wang_Zhiyi', '2000-04-29', N, N, BWF, '2000-04-29', N, N, 61854),
    'zhao jun peng': ('Zhao_Junpeng', '1996-02-02', 185, L, BWF, '1996-02-02', N, N, 95930),
}


def norm(s):
    return ' '.join(str(s).strip().lower().split())


def make_row(name, gender, src, label):
    wiki, wdob, wh, whand, osrc, odob, oh, ohand, oid = src
    if odob is None:
        dob_status = 'single source (Wikipedia)'
    else:
        dob_status = 'agree' if wdob == odob else 'CONFLICT (birth year agrees)'
    hand_vals = {h for h in (whand, ohand) if h}
    if not hand_vals:
        hand, h_stat = None, 'missing'
    elif len(hand_vals) > 1:
        hand, h_stat = None, 'DISAGREE'
    elif whand and ohand:
        hand, h_stat = next(iter(hand_vals)), 'two sources agree'
    else:
        hand, h_stat = next(iter(hand_vals)), f'single source ({"Wikipedia" if whand else osrc})'
    if wh and oh:
        height_status = 'two sources agree' if wh == oh else 'two sources DISAGREE'
    elif wh:
        height_status = 'single source (Wikipedia)'
    elif oh:
        height_status = f'single source ({osrc})'
    else:
        height_status = 'missing'
    other_url = oid if isinstance(oid, str) else f'https://bwfbadminton.com/player/{oid}'
    return dict(
        source=label, player_name=name, gender=gender, birth_year=int(wdob[:4]), dob_wikipedia=wdob, dob_other=odob,
        dob_status=dob_status, height_cm=(oh if oh else wh), height_cm_wikipedia=wh, height_cm_other=oh,
        height_status=height_status, handedness=hand, handedness_wikipedia=whand, handedness_other=ohand,
        handedness_status=h_stat, other_source=osrc, other_url=other_url,
        wikipedia_url=f'https://en.wikipedia.org/wiki/{wiki}')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # ---- ShuttleSet (training)
    pl = pd.read_csv(REPO / 'data' / 'clean' / 'players.csv')
    names = {norm(n): n for n in pl.player_name}
    assert set(names) == set(D), f'missing {set(names) - set(D)}, unexpected {set(D) - set(names)}'
    gender = pl.set_index('player_name').gender
    rows = [make_row(names[k], gender[names[k]], D[k], 'ShuttleSet') for k in D]
    out = pd.DataFrame(rows).sort_values('player_name', key=lambda s: s.map(norm))
    out.to_csv(OUT_DIR / 'player_attributes.csv', index=False, encoding='utf-8-sig')
    print(f'ShuttleSet: wrote {len(out)} players')
    summarise(out)

    # ---- ShuttleSet22 (test): shared players reuse the ShuttleSet values, new players come from D22_NEW
    cw_path = REPO / 'data' / 'clean_shuttleset22' / 'player_crosswalk.csv'
    if not cw_path.exists():
        print('ShuttleSet22 crosswalk not found; run clean_shuttleset.py first. Skipping the test-set table.')
        return
    cw = pd.read_csv(cw_path)
    base = out.set_index('player_name')
    rows22 = []
    for r in cw.itertuples():
        if r.seen_in_shuttleset:
            row = base.loc[r.player_name_shuttleset].to_dict()
            row.update(source='ShuttleSet22', player_name=r.player_name_shuttleset22, gender=r.gender)
        else:
            row = make_row(r.player_name_shuttleset22, r.gender, D22_NEW[norm_key(r.player_name_shuttleset22)], 'ShuttleSet22')
        row['seen_in_shuttleset'] = bool(r.seen_in_shuttleset)
        rows22.append(row)
    unused = set(D22_NEW) - {norm_key(n) for n in cw[~cw.seen_in_shuttleset].player_name_shuttleset22}
    assert not unused, f'entries not used: {unused}'
    out22 = pd.DataFrame(rows22)
    cols = ['source', 'player_name', 'gender', 'seen_in_shuttleset'] + [c for c in out.columns if c not in ('source', 'player_name', 'gender')]
    out22 = out22[cols].sort_values('player_name', key=lambda s: s.map(norm_key))
    out22.to_csv(OUT_DIR / 'player_attributes_shuttleset22.csv', index=False, encoding='utf-8-sig')
    print(f'\nShuttleSet22: wrote {len(out22)} players ({int(out22.seen_in_shuttleset.sum())} shared with ShuttleSet, '
          f'{int((~out22.seen_in_shuttleset).sum())} new)')
    summarise(out22[~out22.seen_in_shuttleset], 'new players only')


def norm_key(name):
    """Same key as norm_name in clean_shuttleset.py: ASCII, lower case, hyphens and dots as spaces."""
    import unicodedata
    t = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode()
    return ' '.join(t.lower().replace('-', ' ').replace('.', ' ').split())


def summarise(df, note=''):
    print(f'  {note}'.rstrip())
    print('  height:', df.height_status.value_counts().to_dict())
    print('  dob   :', df.dob_status.value_counts().to_dict())
    print('  hand  :', df.handedness_status.value_counts().to_dict())
    print('  left-handed:', df[df.handedness == 'left'].player_name.tolist())


if __name__ == '__main__':
    main()
