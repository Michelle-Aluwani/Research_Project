#!/usr/bin/env python3
"""Collect path_tracking_diagnostics.csv files and export detailed per-run JSON.

Place in the research root (alongside terrain_*), then run:
    python collect_diagnostics.py

Optional:
    python collect_diagnostics.py --root "C:/Research" --output "C:/Research/diagnostics"

The program does not modify the input files. It only uses the diagnostic CSV;
fall/success cannot be inferred solely from the length of the recording.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

FILENAME = 'path_tracking_diagnostics.csv'
NUMERIC_FIELDS = ['time_s','robot_x','robot_y','waypoint','look_x','look_y','target_s_m','route_error_m','local_route_error_m','desired_travel_yaw_deg','actual_travel_yaw_deg','heading_error_deg','target_change_deg','steering','yaw_rate_deg_s','pitch_deg','roll_deg','progress_m','recovering','body_yaw_deg']
STATS_FIELDS = ['route_error_m','local_route_error_m','heading_error_deg','pitch_deg','roll_deg','yaw_rate_deg_s','steering','target_change_deg']

def safe_float(s):
    try:
        value=float(str(s).strip())
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None

def stats(vals):
    vals=[v for v in vals if v is not None]
    if not vals: return None
    v=sorted(vals); n=len(v)
    def q(p):
        index=(n-1)*p; a=int(index); b=min(n-1,a+1)
        return v[a]+(v[b]-v[a])*(index-a)
    return {'count':n,'mean':statistics.fmean(v),'median':q(.5),'min':v[0],'max':v[-1],
            'std_population':statistics.pstdev(v) if n>1 else 0.0,'p90':q(.9),'p95':q(.95),'p99':q(.99),
            'rmse':math.sqrt(statistics.fmean(x*x for x in v))}

def ident(rel):
    parts=rel.parts[:-1]
    terrain=next((p for p in parts if re.fullmatch(r'terrain_\d+',p,re.I)),None)
    seed=next((p for p in reversed(parts) if re.fullmatch(r'seed[_-]?\d+',p,re.I)),None)
    budget=next((float(m.group(1)) for p in parts if (m:=re.fullmatch(r'D([0-9]+(?:\.[0-9]+)?)',p,re.I))),None)
    alpha=next((float(m.group(1)) for p in parts if (m:=re.fullmatch(r'alpha[_-]?([0-9]+(?:\.[0-9]+)?)',p,re.I))),None)
    scenario=next((p for p in parts if re.fullmatch(r'(mixed|mud|water|hazard|long)_\d+',p,re.I)),None)
    joined='/'.join(parts).lower()
    planner=('risk_weighted_rrt_star' if 'risk_weighted_rrt_star' in joined else ('rrt_star' if 'rrt_star' in joined else 'cmdp' if budget is not None else None))
    return {'terrain':terrain,'scenario':scenario,'planner':planner,'risk_budget_D':budget,'alpha':alpha,'seed':seed,'parent_directory':str(rel.parent),'source_relative_path':str(rel)}

def calculate(path, root, include_rows=False):
    rel=path.relative_to(root)
    meta=ident(rel)
    rows=[]; invalid_rows=0
    with path.open('r',newline='',encoding='utf-8-sig') as f:
        rdr=csv.DictReader(f)
        columns=rdr.fieldnames or []
        for r in rdr:
            if not r or all(not (x or '').strip() for x in r.values() if isinstance(x,str)):
                continue
            item={k:safe_float(r.get(k)) for k in NUMERIC_FIELDS}
            item['mode']=r.get('mode')
            item['time_s']=safe_float(r.get('time_s'))
            if item['time_s'] is None: invalid_rows+=1
            rows.append(item)
    ordered=sorted((r for r in rows if r['time_s'] is not None),key=lambda r:r['time_s'])
    intervals=[]; travel=0.; first_xy=None; last_xy=None; negative_or_duplicate_times=0
    for r in ordered:
        if r['robot_x'] is not None and r['robot_y'] is not None:
            if first_xy is None: first_xy=[r['robot_x'],r['robot_y']]
            last_xy=[r['robot_x'],r['robot_y']]
    for a,b in zip(ordered,ordered[1:]):
        dt=b['time_s']-a['time_s']
        if dt<=0: negative_or_duplicate_times+=1; continue
        d=None
        if None not in (a['robot_x'],a['robot_y'],b['robot_x'],b['robot_y']):
            d=math.hypot(b['robot_x']-a['robot_x'],b['robot_y']-a['robot_y']);travel+=d
        intervals.append((dt,d))
    t0=ordered[0]['time_s'] if ordered else None
    t1=ordered[-1]['time_s'] if ordered else None
    report={
        'schema_version':1, 'source':{'csv_file':str(path),'relative_path':str(rel),'columns':columns},
        'identity':meta,
        'coverage':{'row_count':len(rows),'valid_timestamp_rows':len(ordered),'invalid_timestamp_rows':invalid_rows,
                    'first_time_s':t0,'last_time_s':t1,'recorded_duration_s':None if t0 is None else t1-t0,
                    'sampling_interval_s':stats([dt for dt,_ in intervals]),'duplicate_time_pairs':negative_or_duplicate_times,
                    'modes':dict(Counter(r['mode'] or '<missing>' for r in rows))},
        'trajectory':{'start_xy_m':first_xy,'end_xy_m':last_xy,'sampled_distance_2d_m':travel if len(ordered)>1 else None,
                      'net_displacement_2d_m':math.dist(first_xy,last_xy) if first_xy and last_xy else None,
                      'note':'2D length sampled from diagnostic timestamps; may underestimate full-resolution PyBullet trajectory distance.'},
        'tracking':{},'orientation':{},'control':{},'progress':{},'data_quality':{},
        'classification':{'fell':None,'mission_success':None,
                          'note':'A short recording is NOT proof of a fall. Confirm from PyBullet result JSON and final goal distance.'}
    }
    for field in STATS_FIELDS:
        values=[r[field] for r in ordered]
        signed=stats(values)
        abs_stats=stats([abs(v) for v in values if v is not None])
        dest='tracking' if 'error_m' in field else 'orientation' if field in ('heading_error_deg','pitch_deg','roll_deg') else 'control'
        report[dest][field]={'signed':signed,'absolute':abs_stats}
    rec=[r['recovering'] for r in ordered if r['recovering'] is not None]
    wp=[int(r['waypoint']) for r in ordered if r['waypoint'] is not None]
    progress=[r['progress_m'] for r in ordered if r['progress_m'] is not None]
    along=[r['target_s_m'] for r in ordered if r['target_s_m'] is not None]
    report['progress']={'first_waypoint':wp[0] if wp else None,'last_waypoint':wp[-1] if wp else None,
                        'unique_waypoint_count':len(set(wp)),'max_waypoint_index':max(wp) if wp else None,
                        'first_progress_m':progress[0] if progress else None,'last_progress_m':progress[-1] if progress else None,
                        'max_progress_m':max(progress) if progress else None,
                        'first_target_s_m':along[0] if along else None,'last_target_s_m':along[-1] if along else None,
                        'recovery_samples':sum(v!=0 for v in rec),'recovery_sample_fraction':(sum(v!=0 for v in rec)/len(rec)) if rec else None}
    report['data_quality']={'missing_numeric_values_by_column':{k:sum(r[k] is None for r in rows) for k in NUMERIC_FIELDS},
                             'warning':'The logged route_error_m is controller-reported deviation; independent geometric point-to-polyline error requires cmdp_path_world.csv.'}
    if include_rows:report['samples']=rows
    return report

def main():
    p=argparse.ArgumentParser(description='Collect path tracking diagnostics into per-run JSON files.')
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--include-rows',action='store_true',help='Include every original sampled row in JSON (larger files)')
    args=p.parse_args()
    root=args.root.resolve(); out=(args.output or (root/'diagnostics')).resolve()
    out.mkdir(parents=True,exist_ok=True)
    paths=sorted(x for x in root.rglob('*') if x.is_file() and x.name.lower()==FILENAME and out not in x.parents)
    summary=[]; errors=[]
    for i,path in enumerate(paths,1):
        try:
            result=calculate(path,root,args.include_rows)
            # Entire relative parent path encoded in name, preventing same scenario/seed collisions.
            name='__'.join(re.sub(r'[^A-Za-z0-9_.-]','_',segment) for segment in path.relative_to(root).parent.parts)
            name=(name or 'root')+'_diagnostics.json'
            dest=out/name
            if dest.exists():
                raise RuntimeError('Output filename conflict: '+str(dest))
            with dest.open('w',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)
            summary.append({'input':str(path.relative_to(root)),'output':str(dest.relative_to(root)) if dest.is_relative_to(root) else str(dest),
                            'rows':result['coverage']['row_count'],'duration_s':result['coverage']['recorded_duration_s'],
                            'mean_route_error_m':((result['tracking']['route_error_m']['signed'] or {}).get('mean')),
                            'max_route_error_m':((result['tracking']['route_error_m']['signed'] or {}).get('max'))})
        except Exception as e:
            errors.append({'input':str(path),'error':repr(e)})
        if i%25==0: print(f'Processed {i}/{len(paths)} files')
    with (out/'processing_report.json').open('w',encoding='utf-8') as f:
        json.dump({'root':str(root),'found':len(paths),'success':len(summary),'errors':errors,'files':summary},f,indent=2)
    with (out/'diagnostics_summary.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['input','output','rows','duration_s','mean_route_error_m','max_route_error_m'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)
    print(f'Found {len(paths)} files; converted {len(summary)}; errors {len(errors)}')
    print('Outputs:',out)
    if errors: print('See processing_report.json for errors.')

if __name__=='__main__': main()
