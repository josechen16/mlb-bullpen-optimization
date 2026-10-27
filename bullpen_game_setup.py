#!/usr/bin/env python3
"""Prepare a game-specific bullpen simulation command from public MLB Stats API.

This is a decision-review helper, NOT a verified historical availability engine.
Review printed lineup and candidate eligibility before interpreting results.
Uses MLB's final boxscore lineup as an approximation (can contain future subs).
"""
import argparse
import json
import shlex
import urllib.request
from datetime import date
from pathlib import Path

BASE="https://statsapi.mlb.com/api/v1"
def fetch(url):
    req=urllib.request.Request(url,headers={"User-Agent":"BullpenResearch/0.1 (academic)"})
    with urllib.request.urlopen(req,timeout=25) as r:
        return json.load(r)

def person(pid, people):
    obj=people.get("ID"+str(pid),{})
    p=obj.get("person",obj)
    return p.get("fullName",str(pid)), obj

def hand(obj, kind, default=None):
    p=obj.get("person",obj)
    value=p.get(kind,{}).get("code")
    return value if value in ("L","R") else default

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date",required=True,help="YYYY-MM-DD")
    ap.add_argument("--team",required=True,help="Pitching team abbreviation (e.g. LAD)")
    ap.add_argument("--opponent",help="Optional opponent abbreviation")
    ap.add_argument("--game-pk",type=int,help="Use exact MLB game ID if doubleheader")
    ap.add_argument("--inning",type=int,default=9)
    ap.add_argument("--half",choices=["top","bottom"],default="bottom")
    ap.add_argument("--outs",type=int,choices=[0,1,2],default=0)
    ap.add_argument("--first",type=int,choices=[0,1],default=0)
    ap.add_argument("--second",type=int,choices=[0,1],default=0)
    ap.add_argument("--third",type=int,choices=[0,1],default=0)
    ap.add_argument("--lead",type=int,default=0,help="Pitching team's lead at decision")
    ap.add_argument("--sims",type=int,default=10000)
    ap.add_argument("--model-dir",default="results/bullpen_quick")
    ap.add_argument("--out",default="results/bullpen_demo")
    args=ap.parse_args()
    if not (date(2025,9,4)<date.fromisoformat(args.date)<=date(2025,12,31)):
        ap.error("Your current model supports 2025 dates after 2025-09-04 only")
    schedule=fetch(f"{BASE}/schedule?sportId=1&date={args.date}&hydrate=team")
    games=[g for d in schedule.get("dates",[]) for g in d.get("games",[])]
    def abbr(g,side):
        return g["teams"][side]["team"].get("abbreviation","").upper()
    games=[g for g in games if args.team.upper() in (abbr(g,"away"),abbr(g,"home"))
           and (not args.opponent or args.opponent.upper() in (abbr(g,"away"),abbr(g,"home")))]
    if args.game_pk: games=[g for g in games if g["gamePk"]==args.game_pk]
    if len(games)!=1:
        print("Matching games:",[(g["gamePk"],abbr(g,"away"),abbr(g,"home")) for g in games])
        ap.error("Specify --game-pk if there is more than one matching game")
    game=games[0]
    pk=game["gamePk"]
    feed=fetch(f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live")
    box=feed["liveData"]["boxscore"]["teams"]
    allpeople=feed["gameData"].get("players",{})
    # Some historical MLB feeds omit gameData.players. Fetch authoritative
    # player bio metadata in one batch from the MLB Stats API.
    all_ids=set(batting_id for side in ("home","away")
                for batting_id in (box[side].get("battingOrder",[]) + box[side].get("pitchers",[])))
    missing=[pid for pid in all_ids if "ID"+str(pid) not in allpeople]
    if missing:
        try:
            response=fetch(f"{BASE}/people?personIds="+",".join(map(str,sorted(missing))))
            for entry in response.get("people",[]):
                allpeople["ID"+str(entry["id"])]=entry
        except Exception as e:
            print(f"WARNING: MLB player bio lookup failed: {e}")

    pitching_side="away" if abbr(game,"away")==args.team.upper() else "home"
    batting_side="home" if pitching_side=="away" else "away"
    batting=box[batting_side]
    pitching=box[pitching_side]
    print(f"\nGame {pk}: {abbr(game,'away')} @ {abbr(game,'home')} on {args.date}")
    print(f"Pitching team: {args.team.upper()} | batting team: {abbr(game,batting_side)}")
    print("\nFINAL BOXSCORE batting order (WARNING: substitutions may have occurred AFTER decision):")
    lineup=[]
    for pid in batting.get("battingOrder",[]):
        obj=batting["players"].get("ID"+str(pid),{})
        name,info=person(pid,allpeople)
        if name==str(pid): name=obj.get("person",{}).get("fullName",name)
        stand=hand(info,"batSide") or hand(obj,"batSide")
        if stand not in ("L","R"):
            print("Missing batting hand:",name,pid)
            continue
        lineup.append((pid,stand,name))
    # A full game boxscore battingOrder may have 10+ players due to subs.
    # Refuse to silently present that as an actual 9-player decision-time order.
    print("\n".join(f"  {i+1:2d}. {name} ({stand}) MLBAM {pid}" for i,(pid,stand,name) in enumerate(lineup)))
    if len(lineup)!=9:
        print(f"WARNING: {len(lineup)} boxscore batters; NOT a validated decision-time nine-player order.")
        print("Choose nine batters in the correct upcoming order manually before running simulation.")
    print("\nPITCHERS in final game boxscore (NOT verified available at decision time):")
    candidates=[]
    for pid in pitching.get("pitchers",[]):
        name,obj=person(pid,allpeople)
        if name==str(pid):
            name=pitching["players"].get("ID"+str(pid),{}).get("person",{}).get("fullName",name)
        throws=hand(obj,"pitchHand") or hand(pitching["players"].get("ID"+str(pid),{}),"pitchHand")
        print(f"  {name}: {pid}:{throws or '?'}")
        if throws in ("L","R"): candidates.append((pid,throws,name))
    print("\nIMPORTANT: This list includes pitchers used earlier or later; it excludes unused relievers.")
    print("Check actual roster, injuries, prior usage, decision-time bullpen status, and fatigue.")
    if len(lineup)!=9 or not candidates:
        print("\nNo automatic command generated because the inputs are incomplete or potentially misleading.")
        return
    # Create a TEMPLATE command. Do not automatically run: a human must verify eligibility.
    tokens=["python3","bullpen_simulation_v3.py","--model-dir",args.model_dir,
            "--batters",*(f"{pid}:{h}" for pid,h,_ in lineup),
            "--pitchers",*(f"{pid}:{h}" for pid,h,_ in candidates),
            "--pitcher-names",*(f"{pid}:{name}" for pid,_,name in candidates),
            "--date",args.date,"--inning",str(args.inning),"--outs",str(args.outs),
            "--first",str(args.first),"--second",str(args.second),"--third",str(args.third),
            "--lead",str(args.lead),"--sims",str(args.sims),
            "--output-dir",str(Path(args.out)/str(pk))]
    if args.half=="bottom" and args.inning>=9 and args.lead>=0: tokens.append("--walkoff")
    print("\nUNVERIFIED TEMPLATE ONLY (edit batting order, pitchers, and actual pitcher first):\n")
    print(" ".join(shlex.quote(x) for x in tokens))
    print("\nThe current model's historical features stop at 2025-09-04, even for later game dates.")
if __name__=="__main__":
    main()
