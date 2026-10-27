#!/usr/bin/env python3
"""Walk-off-aware Monte Carlo comparison using the saved six-class PA model.

Requires bullpen_predict.py in the same directory. No game-day outcomes are used.
Simplified running: 1B advances runners one base, 2B_3B is a double,
BB_HBP forces only, HR scores everyone, outs do not advance runners.
No double plays, sacrifices, steals, errors or in-inning fatigue model.
"""
import argparse
import json
import pickle
import csv
import warnings
import urllib.request
from datetime import date as dtdate
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from types import SimpleNamespace

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import bullpen_predict as bp

# The trusted history pickle references functions defined in __main__.
History = bp.History
zero_counts = bp.zero_counts


def player(value):
    try:
        ident, hand = value.split(':')
        ident = int(ident)
        if ident <= 0 or hand.upper() not in ('L', 'R'):
            raise ValueError()
        return ident, hand.upper()
    except ValueError:
        raise argparse.ArgumentTypeError('Expected MLBAM_ID:L or MLBAM_ID:R')


def same_day(value):
    try:
        ident, pitches = value.split(':')
        ident, pitches = int(ident), int(pitches)
        if ident <= 0 or pitches < 0:
            raise ValueError()
        return ident, pitches
    except ValueError:
        raise argparse.ArgumentTypeError('Expected PITCHER_ID:PITCHES')


def update(outcome, bases, outs):
    a, b, c = bases
    if outcome in ('K', 'OTHER_OUT'):
        return bases, outs + 1, 0
    if outcome == 'HR':
        return (0, 0, 0), outs, 1 + a + b + c
    if outcome == '1B':
        return (1, a, b), outs, c
    if outcome == '2B_3B':
        return (0, 1, a), outs, b + c
    if outcome == 'BB_HBP':
        return (1, b or a, c or (a and b)), outs, int(bool(a and b and c))
    raise ValueError(f'Unknown outcome: {outcome}')


def wilson(successes, n):
    if n == 0:
        return (float('nan'), float('nan'))
    z = 1.95996398454
    p = successes / n
    den = 1 + z*z/n
    ctr = (p + z*z/(2*n))/den
    margin = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return ctr-margin, ctr+margin


def export_dashboard(rows, args, cutoff):
    """Export all three run-prevention metrics and context-appropriate primary risk."""
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    names={}
    for item in args.pitcher_names:
        ident,sep,name=item.partition(':')
        if sep and ident.isdigit():names[int(ident)]=name.strip()
    walkoff=args.walkoff
    metric='Walk-off probability' if walkoff else 'Expected runs allowed'
    records=[]
    for r in rows:
        records.append(dict(pitcher_id=r['pid'],pitcher_name=names.get(r['pid'],str(r['pid'])),throws=r['hand'],
            expected_runs=round(r['er'],5),p_allow_1plus=round(r['p1'],6),p_allow_2plus=round(r['p2'],6),
            walkoff_probability=round(r['walkoff'],6) if walkoff else '',
            primary_metric=metric,primary_value=round(r['walkoff'] if walkoff else r['er'],6),
            primary_ci95_lower=round(r['ci'][0],6),primary_ci95_upper=round(r['ci'][1],6),
            actual_choice=r['pid']==args.actual_pitcher,recommended=r is rows[0],
            capped_simulations=r['capped'],same_day_pitches=r['pitches'] if r['pitches'] is not None else '',
            date=args.date,inning=args.inning,model_cutoff=str(cutoff.date()),simulations=args.sims,
            candidate_status='PROVISIONAL - not verified available'))
#    csv_path=out/'bullpen_results.csv'
#    with csv_path.open('w',newline='',encoding='utf-8') as f:
#        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    n=len(rows); fig,ax=plt.subplots(figsize=(13,max(6.8,4.7+n*.57)),dpi=165)
    fig.patch.set_facecolor('white');ax.set_facecolor('white')
    display=list(reversed(rows));ys=np.arange(n)
    vals=[r['walkoff']*100 if walkoff else r['er'] for r in display]
    labels=[];colors=[]
    for r in display:
        lab=f"{names.get(r['pid'],r['pid'])} ({r['hand']})"
        if r['pid']==args.actual_pitcher:lab+=' [ACTUAL]'
        if r is rows[0]:lab+=' [BEST MODEL]'
        labels.append(lab)
        colors.append('#168675' if r is rows[0] else ('#D39A41' if r['pid']==args.actual_pitcher else '#7A90A6'))
    ax.barh(ys,vals,color=colors,height=.62)
    if walkoff:
        lo=np.array([r['ci'][0]*100 for r in display]);hi=np.array([r['ci'][1]*100 for r in display])
        ax.errorbar(vals,ys,xerr=[np.maximum(0,np.array(vals)-lo),np.maximum(0,hi-np.array(vals))],
                    fmt='none',ecolor='#263548',elinewidth=1.4,capsize=3)
    maxv=max(vals) if vals else 1
    ax.set_xlim(0,max(0.2,maxv*1.65))
    for i,r in enumerate(display):
        value=f'{r["walkoff"]:.2%}' if walkoff else f'{r["er"]:.3f} runs'
        extra=f'P(>=1): {r["p1"]:.1%}  |  P(>=2): {r["p2"]:.1%}'
        ax.text(vals[i]+maxv*.04,i,value,va='center',fontsize=12,fontweight='bold')
        ax.text(vals[i]+maxv*.04,i-.27,extra,va='center',fontsize=8,color='#506176')
    ax.set_yticks(ys,labels,fontsize=11)
    ax.set_xlabel('Estimated walk-off probability (%)' if walkoff else 'Expected runs allowed (runs / half-inning)',fontsize=12)
    ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('MLB Bullpen Decision Support',x=.07,y=.985,ha='left',fontsize=21,fontweight='bold')
    fig.text(.07,.938,('WALK-OFF PREVENTION' if walkoff else 'RUN PREVENTION')+'  |  Lower is better',fontsize=13,color='#168675',weight='bold')
    fig.text(.07,.904,f'{args.date} | Inning {args.inning} | Outs {args.outs} | Bases {args.first}{args.second}{args.third} | Pitching lead {args.lead:+d}',fontsize=11,color='#46556A')
    fig.text(.07,.873,f'Model through {cutoff.date()} | {args.sims:,} simulations per pitcher | {len(args.batters)} batters',fontsize=10,color='#46556A')
    fig.subplots_adjust(left=.31,right=.87,top=.81,bottom=.20)
    fig.text(.07,.065,'P(>=1) = probability of allowing at least 1 run; P(>=2) = at least 2 runs.\n'
             'Candidates are PROVISIONAL; starter/rest/availability restrictions may be incomplete.\n'
             'Simplified baserunning, no in-inning fatigue. Monte Carlo uncertainty excludes model error.',
             fontsize=9,color='#46556A',linespacing=1.5)
    png=out/'bullpen_dashboard.png';fig.savefig(png,facecolor='white',dpi=165);plt.close(fig)
#    print(f'\nSaved CSV: {csv_path.resolve()}\nSaved dashboard: {png.resolve()}')


def mlb_json(url):
    request = urllib.request.Request(url, headers={'User-Agent':'AcademicBullpenResearch/1.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def likely_rotation_starter(pid, season=2025):
    """Conservative MLB season-stats screen. Return True, False or None (unknown)."""
    try:
        result=mlb_json(f'https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=season&group=pitching&season={season}')
        for section in result.get('stats',[]):
            for split in section.get('splits',[]):
                st=split.get('stat',{}); gs=int(st.get('gamesStarted',0));gp=int(st.get('gamesPitched',0))
                return gs>=5 and gs>=max(3,int(.50*gp))
        return None
    except Exception as ex:
        print(f'WARNING: Could not classify pitcher {pid}: {ex}')
        return None


def auto_game(args, ap):
    """Resolve an inning-start decision from public MLB game records.

    This intentionally excludes used pitchers and known starters. Availability
    of unused relievers cannot be proven from public box scores alone.
    """
    d = dtdate.fromisoformat(args.date)
    if not (dtdate(2025,9,4) < d <= dtdate(2025,12,31)):
        ap.error('Automatic game lookup currently supports 2025-09-05 through 2025-12-31')
    if not args.team:
        ap.error('--team is required for automatic game lookup')
    root = 'https://statsapi.mlb.com/api/v1'
    schedule = mlb_json(f'{root}/schedule?sportId=1&date={args.date}&hydrate=team')
    games = [g for day in schedule.get('dates',[]) for g in day.get('games',[])]
    def abbr(game,side):
        return game['teams'][side]['team'].get('abbreviation','').upper()
    games = [g for g in games if args.team.upper() in (abbr(g,'away'),abbr(g,'home'))
             and (not args.opponent or args.opponent.upper() in (abbr(g,'away'),abbr(g,'home')))
             and (not args.game_pk or args.game_pk==g['gamePk'])]
    if len(games)!=1:
        ap.error('Game ambiguous/not found: specify --opponent or --game-pk; matches: '+str([(g['gamePk'],abbr(g,'away'),abbr(g,'home')) for g in games]))
    game=games[0]; pk=game['gamePk']
    pside='away' if abbr(game,'away')==args.team.upper() else 'home'
    bside='home' if pside=='away' else 'away'
    half='bottom' if bside=='home' else 'top'
    if args.half and args.half!=half:
        ap.error(f'{args.team} is pitching in the {half} half, not {args.half}')
    feed=mlb_json(f'{root}.1/game/{pk}/feed/live')
    plays=feed['liveData']['plays']['allPlays']
    inning_plays=[(idx,pl) for idx,pl in enumerate(plays)
                  if pl.get('about',{}).get('inning')==args.inning and pl.get('about',{}).get('halfInning')==half]
    if not inning_plays:
        ap.error('No completed play-by-play for selected inning/half; choose a played inning')
    ix,first=inning_plays[0]
    if ix:
        prev=plays[ix-1].get('result',{})
        away,home=prev.get('awayScore'),prev.get('homeScore')
    else:
        away,home=0,0
    if away is None or home is None:
        ap.error('Score before inning unavailable; refusing to guess')
    args.lead=int(away-home if pside=='away' else home-away)
    args.outs=args.first=args.second=args.third=0
    if args.inning >= 10:
        # MLB extra-inning automatic runner on second at start of half inning.
        # The exact runner identity is not used by the current model.
        args.second=1
    args.walkoff=(half=='bottom' and args.inning>=9 and args.lead>=0)
    box=feed['liveData']['boxscore']['teams']
    bteam=box[bside]; pteam=box[pside]
    roster_ids=set(bteam.get('battingOrder',[])) | set(pteam.get('pitchers',[])) | set(pteam.get('players',{}).get('ID'+str(x),{}).get('person',{}).get('id',x) for x in pteam.get('pitchers',[]))
    team_id=game['teams'][pside]['team']['id']
    roster=[]
    try:
        roster=mlb_json(f'{root}/teams/{team_id}/roster?rosterType=active&date={args.date}').get('roster',[])
        roster_ids.update(r['person']['id'] for r in roster)
    except Exception as ex:
        ap.error('Could not verify historical roster: '+str(ex))
    if not roster:
        ap.error('Historical active roster empty; cannot build bullpen')
    people={}
    ids=sorted(roster_ids)
    # Fetch in batches to avoid oversized API requests.
    for i in range(0,len(ids),75):
        for person in mlb_json(f'{root}/people?personIds='+','.join(map(str,ids[i:i+75]))).get('people',[]):
            people[person['id']]=person
    def bio(pid):return people.get(pid,{})
    def handed(pid,key):return bio(pid).get(key,{}).get('code')
    # Determine nine slots from boxscore battingOrder; the order field is
    # typically 100, 200, ... 900, while substitutions share a slot.
    slots={}
    for pid in bteam.get('battingOrder',[]):
        player_box=bteam.get('players',{}).get('ID'+str(pid),{})
        try: slot=int(player_box.get('battingOrder',0))//100
        except (ValueError,TypeError):continue
        if 1<=slot<=9:slots.setdefault(slot,[]).append(pid)
    if len(slots)!=9:
        ap.error('Cannot reconstruct nine batting slots from box score')
    # Players who appeared as batters before this decision are known;
    # for unseen slots choose first listed (usually starter).
    seen={}
    for pl in plays[:ix]:
        batter=pl.get('matchup',{}).get('batter',{}).get('id')
        for slot,members in slots.items():
            if batter in members:seen[slot]=batter
    current={slot:seen.get(slot,members[0]) for slot,members in slots.items()}
    next_batter=first.get('matchup',{}).get('batter',{}).get('id')
    start_slot=next((slot for slot,members in slots.items() if next_batter in members),None)
    if start_slot is None:ap.error('Upcoming batter not in reconstructed batting order')
    current[start_slot]=next_batter

    order = []

    for step in range(9):
        pid = current[(start_slot - 1 + step) % 9 + 1]
        hand = handed(pid, 'batSide')

        if hand not in ('L', 'R', 'S'):
            detail = mlb_json(f'{root}/people/{pid}')
            players = detail.get('people', [])
            if players:
                hand = players[0].get('batSide', {}).get('code')

        if hand not in ('L', 'R', 'S'):
            ap.error(f'Cannot determine batting hand for {pid}')

        order.append((pid, hand))

    args.batters=order
    used_before={pl.get('matchup',{}).get('pitcher',{}).get('id') for pl in plays[:ix]}
    # Current inning-start pitcher, if any, is the actual first-PA pitcher.
    actual=first.get('matchup',{}).get('pitcher',{}).get('id')
    args.actual_pitcher=actual
    # MLB historical roster position labels often say P, not RP. The prior
    # RP-only filter silently discarded nearly all of the bullpen.
    # Use every identified pitcher on the historical active roster, excluding
    # anyone who appeared BEFORE the decision, and also keep the actual pitcher.
    # These are *provisional* candidates; starter/availability checks are needed.
    candidates=[]
    roster_pitcher_ids=set()
    for r in roster:
        pid=r['person']['id']
        primary=bio(pid).get('primaryPosition',{}).get('abbreviation','')
        pos=r.get('position',{}).get('abbreviation','')
        ptype=bio(pid).get('primaryPosition',{}).get('type','')
        if primary in ('P','RP','SP','TWP') or pos in ('P','RP','SP','TWP') or ptype in ('Pitcher','Two-Way Player'):
            roster_pitcher_ids.add(pid)
    # Boxscore pitching participants are reliable historical evidence of roster
    # membership, even when the date-filtered active-roster endpoint is partial.
    roster_pitcher_ids.update(pteam.get('pitchers',[]))
    for pid in sorted(roster_pitcher_ids):
        if pid in used_before and pid != actual:continue
        if pid in args.exclude_pitchers and pid != actual:continue
        if pid != actual and not args.include_starters:
            starter=likely_rotation_starter(pid)
            if starter is True:
                print('Excluded likely rotation starter:',bio(pid).get('fullName',pid))
                continue
            if starter is None:
                print('WARNING: Could not verify starter/reliever role for',bio(pid).get('fullName',pid))
        throws=handed(pid,'pitchHand')
        if throws in ('L','R'):candidates.append((pid,throws))
    if actual not in [pid for pid,_ in candidates]:
        throws=handed(actual,'pitchHand')
        if throws in ('L','R'):candidates.append((actual,throws))
    if len(candidates)<2:
        ap.error('Only one candidate identified. Refusing to label the actual pitcher BEST MODEL. Historical roster/bullpen data must be verified.')
    args.pitchers=candidates
    args.pitcher_names=[f"{pid}:{bio(pid).get('fullName',pid)}" for pid,_ in candidates]
    year, month, day = args.date.split("-")

    args.output_dir = str(
        Path(args.output_dir)
        / f"{year}-{month}-{day}"
        / f"{args.opponent.upper()}-{args.team.upper()}"
        / f"inning_{args.inning}"
    )

    print(f'\nAUTO GAME: {abbr(game,"away")} @ {abbr(game,"home")} | MLB game {pk}')
    print(f'Pitching team: {args.team.upper()} | {half} {args.inning} | score {away}-{home} | lead {args.lead:+d}')
    print('Upcoming order:',', '.join(bio(pid).get('fullName',str(pid)) for pid,_ in order))
    print('Actual pitcher:',bio(actual).get('fullName',str(actual)))
    print('Candidate pool (PROVISIONAL):',', '.join(bio(pid).get('fullName',str(pid)) for pid,_ in candidates))
    print('WARNING: Pool includes potentially unavailable starters; roster eligibility, rest and workload require independent verification.')
    print('WARNING: Substitutions in other batting slots may require review; this is an inning-start reconstruction.')
    print('WARNING: Historical model features remain frozen at 2025-09-04 for all game dates.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', default='results/bullpen_quick')
    ap.add_argument('--batters', nargs='+', type=player, help='Batters in upcoming order, ID:hand')
    ap.add_argument('--pitchers', nargs='+', type=player, help='Candidate pitchers ID:throws')
    ap.add_argument('--date', required=True)
    ap.add_argument('--team', help='Pitching team abbreviation; enables automatic game lookup')
    ap.add_argument('--opponent', help='Opponent abbreviation')
    ap.add_argument('--game-pk', type=int, help='Exact MLB game ID, useful for doubleheaders')
    ap.add_argument('--half', choices=['top','bottom'], help='Optional half-inning validation')
    ap.add_argument('--inning', type=int, default=9)
    ap.add_argument('--outs', type=int, choices=(0,1,2), default=0)
    ap.add_argument('--first', type=int, choices=(0,1), default=0)
    ap.add_argument('--second', type=int, choices=(0,1), default=0)
    ap.add_argument('--third', type=int, choices=(0,1), default=0)
    ap.add_argument('--lead', type=int, default=0, help='Pitching team lead (negative if trailing)')
    ap.add_argument('--walkoff', action='store_true', help='Stop as soon as home batting team takes lead in bottom 9+')
    ap.add_argument('--sims', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-pa', type=int, default=40)
    ap.add_argument('--output-dir', default='results/bullpen_demo', help='Folder for PNG and CSV')
    ap.add_argument('--actual-pitcher', type=int, help='Pitcher actually chosen by manager (MLBAM ID)')
    ap.add_argument('--pitcher-names', nargs='*', default=[], help='Optional quoted ID:Name entries')
    ap.add_argument('--today-pitches', nargs='*', type=same_day, default=[], help='Known pitches already thrown today ID:count (reported, not used as model feature)')
    ap.add_argument('--include-starters', action='store_true', help='Do not exclude likely rotation starters')
    ap.add_argument('--exclude-pitchers', nargs='*', type=int, default=[], help='Manually exclude known unavailable pitchers by MLBAM ID')
    ap.add_argument('--max-today-pitches', type=int, help='Exclude candidates exceeding this pre-existing same-day pitch count')
    args = ap.parse_args()
    if args.team and not args.batters and not args.pitchers:
        auto_game(args, ap)
    elif not args.batters or not args.pitchers:
        ap.error('Use --team for automatic lookup, or supply both --batters and --pitchers')
    if args.sims < 1 or args.max_pa < 1: ap.error('--sims and --max-pa must be positive')
    if args.walkoff and args.inning < 9: ap.error('--walkoff requires inning >= 9')
    if args.walkoff and args.lead < 0: ap.error('Home batting team already leads; not a valid start for a walk-off scenario')
    if len(args.batters) != 9:
        print(f'WARNING: {len(args.batters)} batters supplied, not a full nine-player order; the provided list will cycle.')
    today = dict(args.today_pitches)
    folder = Path(args.model_dir)
    meta = json.loads((folder/'model_metadata.json').read_text())
    cutoff = pd.Timestamp(meta['final_fit_through']).normalize()
    date = pd.Timestamp(args.date).normalize()
    if date <= cutoff or date.year != cutoff.year:
        ap.error(f'Prediction date must be after {cutoff.date()} and in the same season')
    if (date-cutoff).days > 30:
        print('WARNING: Historical state is over 30 days old.')
    with (folder/meta['history_path']).open('rb') as f:
        state = pickle.load(f)  # Only load trusted, locally trained pickle files.
    workload = pd.read_parquet(folder/meta['workload_history_path'])
    workload['game_date'] = pd.to_datetime(workload['game_date']).dt.normalize()
    model = XGBClassifier()
    model.load_model(str(folder/meta['model_path']))
    classes = meta['classes']
    if set(classes) != {'1B','2B_3B','BB_HBP','HR','K','OTHER_OUT'}:
        raise ValueError('Unexpected model outcome classes')
    if model.n_features_in_ != len(meta['feature_names']):
        raise ValueError('Feature count mismatch')
    print(f'\nDate: {args.date} | inning {args.inning} | outs {args.outs} | bases {args.first}{args.second}{args.third} | pitching lead {args.lead:+d}')
    print(f'Model cutoff: {cutoff.date()} | {args.sims:,} simulations per pitcher | walk-off stop: {args.walkoff}')
    print('Same-day pitch counts are informational or used for exclusion, NOT included in XGBoost predictions.')
    print('No in-inning fatigue update; each candidate pitches the remainder of the half-inning.')
    rows=[]
    for pitcher, throws in args.pitchers:
        prior_pitches=dict(args.today_pitches).get(pitcher)
        if args.max_today_pitches is not None and prior_pitches is not None and prior_pitches>args.max_today_pitches:
            print(f'Excluded {pitcher}: {prior_pitches} pitches already today');continue
        cache={}
        def probabilities(batter_index,outs,bases,lead):
            key=(batter_index,outs,bases,lead)
            if key not in cache:
                batter,stand=args.batters[batter_index]

                if stand == 'S':
                    effective_stand = 'R' if throws == 'L' else 'L'
                else:
                    effective_stand = stand

                req=SimpleNamespace(batter=batter,pitcher=pitcher,stand=stand,throws=throws,
    date=args.date,inning=args.inning,outs=outs,first=bases[0],second=bases[1],third=bases[2],lead=lead)
                features,_=bp.make_features(req,meta,state,workload)
                probs=np.asarray(model.predict_proba(features)[0],dtype=float)
                probs=np.maximum(probs,0);cache[key]=probs/probs.sum()
            return cache[key]
        rng=np.random.default_rng(args.seed)  # consistent seed across candidates
        runs=np.empty(args.sims,dtype=np.int32);walkoffs=0;capped=0
        for i in range(args.sims):
            bases=(args.first,args.second,args.third)
            outs=args.outs;scored=0;batter_idx=0;pa=0
            while outs<3 and pa<args.max_pa:
                probs=probabilities(batter_idx,outs,bases,args.lead-scored)
                outcome=classes[int(rng.choice(len(classes),p=probs))]
                bases,outs,added=update(outcome,bases,outs)
                scored+=added;pa+=1;batter_idx=(batter_idx+1)%len(args.batters)
                if args.walkoff and scored>args.lead:
                    walkoffs+=1;break
            if outs<3 and pa>=args.max_pa and not(args.walkoff and scored>args.lead):capped+=1
            runs[i]=scored
        p1=int(np.count_nonzero(runs>=1));p2=int(np.count_nonzero(runs>=2))
        primary_count=walkoffs if args.walkoff else p1
        ci=wilson(primary_count,args.sims) if args.walkoff else (float('nan'),float('nan'))
        rows.append(dict(pid=pitcher,hand=throws,er=float(runs.mean()),p1=p1/args.sims,
                         p2=p2/args.sims,walkoff=walkoffs/args.sims if args.walkoff else float('nan'),
                         ci=ci,capped=capped,pitches=prior_pitches))
    if not rows:ap.error('No eligible pitchers after filters')
    if args.team and len(rows)<2:ap.error('Fewer than two candidates; cannot rank alternatives')
    rows.sort(key=lambda r:r['walkoff'] if args.walkoff else r['er'])
    print('\nMetric:', 'P(WALK-OFF), lower is better' if args.walkoff else 'EXPECTED RUNS ALLOWED, lower is better')
    print(f'{"Pitcher":<17} {"Exp Runs":>10} {"P(>=1)":>10} {"P(>=2)":>10} {"P(WO)":>10} {"Capped":>8}')
    for r in rows:
        wo=f'{r["walkoff"]:.2%}' if args.walkoff else 'n/a'
        print(f'{r["pid"]} ({r["hand"]})'.ljust(17)+f'{r["er"]:10.3f} {r["p1"]:10.2%} {r["p2"]:10.2%} {wo:>10} {r["capped"]:8d}')
    print('NOTE: These are model-based simulated probabilities, not measured real-world rates.')
    export_dashboard(rows,args,cutoff)

if __name__ == '__main__':
    main()
