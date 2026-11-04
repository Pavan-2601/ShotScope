import csv, json, sys
root, video, out = sys.argv[1:4]
sets = [int(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else [1]  # optional: e.g. 1,2
import os, re
T = {}
for line in open(os.path.join(root, '..', 'README.md'), encoding='utf-8'):
    m = re.match(r'\|\s*([A-Za-z\- ]+?)\s*\|\s*([^\s|]+)\s*\|', line)
    if m and m.group(1) != 'English':
        T[m.group(2)] = m.group(1)
# variants found in the data but missing from the README table
T.setdefault('過度切球', 'passive drop')
T.setdefault('未知球種', 'unknown')
# not in the README; translated by hand (covers every value found in the dataset)
R = {'對手落地致勝':'opponent landed (winning shot)', '出界':'out of bounds', '掛網':'hit the net',
     '未過網':'did not cross the net', '落點判斷失誤':"misjudged the shuttle's landing point",
     '對手落地判斷失誤':"opponent misjudged the landing point", '犯規':'fault',
     '落地致勝':'landed (winning shot)', '對手出界':'opponent out of bounds', '對手掛網':'opponent hit the net',
     '對手未過網':'opponent did not cross the net', '對手落點判斷失誤':"opponent misjudged the shuttle's landing point",
     '落地判斷失誤':'misjudged the landing point', '對手犯規':'opponent fault'}
match = next(r for r in csv.DictReader(open(f'{root}/match.csv')) if r['video']==video)
hom = next(r for r in csv.DictReader(open(f'{root}/homography.csv')) if r['video']==video)
H = json.loads(hom['homography_matrix'])
def tf(x, y):
    if x in ('', None) or y in ('', None): return ('', '')
    x, y = float(x), float(y)
    w = H[2][0]*x + H[2][1]*y + H[2][2]
    return (round((H[0][0]*x+H[0][1]*y+H[0][2])/w, 2), round((H[1][0]*x+H[1][1]*y+H[1][2])/w, 2))
rows = []
for n in sets:
    for r in csv.DictReader(open(f'{root}/{video}/set{n}.csv', encoding='utf-8')):
        r['_set'] = n
        rows.append(r)
pairs = [('hit','hit_x','hit_y'),('landing','landing_x','landing_y'),
         ('player_location','player_location_x','player_location_y'),
         ('opponent_location','opponent_location_x','opponent_location_y')]
res = []
for r in rows:
    d = {'match_id':match['id'],'video':video,'tournament':match['tournament'],'round':match['round'],
         'date':f"{match['year']}-{int(match['month']):02d}-{int(match['day']):02d}",'set':r['_set']}
    d.update({k: v for k, v in r.items() if k != '_set'})
    d['type'] = T.get(r['type'], r['type'])
    for c in ('lose_reason', 'win_reason'):
        d[c] = R.get(r[c], r[c])
    d['player_name'] = match['winner'] if r['player']=='A' else match['loser']
    gp = r['getpoint_player']
    d['getpoint_player_name'] = match['winner'] if gp=='A' else match['loser'] if gp=='B' else ''
    for n, xc, yc in pairs:
        d[n+'_court_x'], d[n+'_court_y'] = tf(r[xc], r[yc])
    for k in ['homography_matrix','upleft_x','upright_x','downleft_x','downright_x','upleft_y','upright_y','downleft_y','downright_y']:
        d['h_'+k if k!='homography_matrix' else k] = hom[k]
    res.append(d)
with open(out,'w',newline='',encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
print(len(res), 'rows;', len(res[0]), 'cols'); print(sorted({r['type'] for r in rows}-set(T)))
