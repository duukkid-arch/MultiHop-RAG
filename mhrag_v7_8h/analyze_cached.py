#!/usr/bin/env python3
"""V7 CPU-only: DEV-chosen CE budgets, paired comparisons, exact-gold-family bootstrap.
Run after v6 eval files are complete. This never scores a model.
"""
import argparse, csv, json, random, statistics, sys
from collections import defaultdict
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--project',default='/root/autodl-tmp/MultiHop-RAG')
p.add_argument('--v6-root',default='results/v6_focus_750_musa')
p.add_argument('--out',default='results/v7_8h_budget')
p.add_argument('--tolerance-pp',type=float,default=0.5,help='DEV quality tolerance from best CE@K (percentage points)')
p.add_argument('--bootstrap',type=int,default=1000)
a=p.parse_args()
project=Path(a.project).resolve()
sys.path.insert(0,str(project/'mhrag_overnight_v5'))
from common import gold_keys, gold_rows
v6=Path(a.v6_root);v6=v6 if v6.is_absolute() else project/v6
out=Path(a.out);out=out if out.is_absolute() else project/out
out.mkdir(parents=True,exist_ok=True)
gold=gold_rows(str(project/'dataset/MultiHopRAG.json'),0,20261027)

K=(20,30,60,100)

def read(name):
    fp=v6/f'eval_{name}'/'per_query.csv'
    if not fp.is_file():
        print('SKIP missing evaluation:',fp,flush=True)
        return None
    with fp.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    need=['index','split','type','baseline_ce100_all']+[f'ce{k}_all' for k in K]
    missing=[k for k in need if not rows or k not in rows[0]]
    if missing:raise RuntimeError(f'{fp}: missing columns {missing}')
    return rows

def gold_family(row):
    i=int(row['index'])
    if i not in gold:return ('unknown',str(i))
    # Same complete gold evidence set => same family; overlap-only families are not merged.
    return tuple(sorted(str(x) for x in gold_keys(gold[i]))) or ('ungrounded',str(i))

def cluster_ci(rows,a_col,b_col,rep=1000,seed=20261028):
    families=defaultdict(list)
    for r in rows: families[gold_family(r)].append(float(r[a_col])-float(r[b_col]))
    arrays=list(families.values());n=len(arrays)
    if not n:return None
    rng=random.Random(seed);samples=[]
    for _ in range(rep):
        s=total=0
        for j in range(n):
            group=arrays[rng.randrange(n)];s+=sum(group);total+=len(group)
        samples.append(s/total)
    samples.sort()
    return [round(samples[int(rep*.025)]*100,3),round(samples[min(rep-1,int(rep*.975))]*100,3)],n

report={
    'purpose':'exploratory, cached-score, oracle question type budget study',
    'holdout_caveat':'Existing holdout repeatedly inspected in v5; NOT fresh independent confirmation',
    'metrics':'document-level All-Gold@10; CE@K means rerank only initial K raw candidates using cached scores',
    'tolerance_pp':a.tolerance_pp,'methods':{}
}
for name in ('bm25_750','hybrid_750'):
    rows=read(name)
    if not rows:continue
    dev=[r for r in rows if r['split']=='dev']
    hold=[r for r in rows if r['split']=='holdout']
    if not dev or not hold:raise RuntimeError(f'{name} missing DEV or HOLDOUT')
    types=sorted(set(r['type'] for r in dev))
    selected={};type_table=[]
    for typ in types:
        rs=[r for r in dev if r['type']==typ]
        vals={k:sum(float(r[f'ce{k}_all']) for r in rs)/len(rs) for k in K}
        best=max(vals.values())
        within=[k for k in K if vals[k]>=best-a.tolerance_pp/100-1e-12]
        selected[typ]=min(within)
        type_table.append({'type':typ,'n_dev':len(rs),'budget':selected[typ],
                           **{f'dev_ce{k}_all':round(vals[k],6) for k in K}})
    detail=[];summary={}
    for split_name,rr in [('dev',dev),('holdout',hold)]:
        policy=[];base=[];full=[];fixed60=[];budgets=[]
        for r in rr:
            k=selected[r['type']]
            v=float(r[f'ce{k}_all']);b=float(r['baseline_ce100_all'])
            f=float(r['ce100_all']);s=float(r['ce60_all'])
            policy.append(v);base.append(b);full.append(f);fixed60.append(s);budgets.append(k)
            detail.append({'method':name,'split':split_name,'index':r['index'],'type':r['type'],
                           'gold_family':str(gold_family(r)),'budget':k,'policy_all':v,
                           'baseline_all':b,'ce60_all':s,'ce100_all':f})
        n=len(rr);delta=[v-b for v,b in zip(policy,base)]
        # grouping bootstrap directly over already-computed policy scores
        ci_rows=[{**r, '__policy':str(policy[j])} for j,r in enumerate(rr)]
        ci,n_families=cluster_ci(ci_rows,'__policy','baseline_ce100_all',a.bootstrap)
        summary[split_name]={
            'n':n,'gold_families_exact':n_families,
            'baseline_all':round(statistics.mean(base),6),
            'ce60_all':round(statistics.mean(fixed60),6),
            'ce100_all':round(statistics.mean(full),6),
            'policy_all':round(statistics.mean(policy),6),
            'delta_vs_baseline_pp':round(statistics.mean(delta)*100,3),
            'cluster_bootstrap_95ci_pp':ci,
            'better_vs_baseline':sum(v>b for v,b in zip(policy,base)),
            'worse_vs_baseline':sum(v<b for v,b in zip(policy,base)),
            'avg_candidate_pairs':round(statistics.mean(budgets),3),
            'theoretical_pair_reduction_pct':round(100-statistics.mean(budgets),3)
        }
    report['methods'][name]={'dev_chosen_budgets':selected,'by_type_dev':type_table,'results':summary}
    with (out/f'{name}_per_query.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(detail[0]));writer.writeheader();writer.writerows(detail)
    print('\nMETHOD',name,'DEV-budget',selected)
    for split,z in summary.items():
        print(split,'n',z['n'],'families',z['gold_families_exact'],
              'policy',z['policy_all'],'baseline',z['baseline_all'],
              'delta_pp',z['delta_vs_baseline_pp'],'cluster_CI_pp',z['cluster_bootstrap_95ci_pp'],
              'avg_pairs',z['avg_candidate_pairs'])
if not report['methods']:
    raise SystemExit('No completed V6 per_query.csv found. Finish v6 eval first; no inference was run.')
(out/'V7_BUDGET_REPORT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print('SAVED',out/'V7_BUDGET_REPORT.json')
