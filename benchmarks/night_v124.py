#!/usr/bin/env python3
"""Finite, resumable, read-only HTTP evaluation of v12.4 on RTX 4090.

Gold document IDs/reference answers never enter HTTP requests. All arms run on
one service with one frozen database. Repeat variants are not independent data.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,random,re,sqlite3,statistics,subprocess,sys,time,urllib.request
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

ARMS={'rag':('hybrid_ce',False),'agentic':('hybrid_ce',True),'agentic_hybrid':('hybrid',True)}
FOCUS=('v9_032','v9_042','v9_089','v9_097','v9_095','compare_083_084_not_comparable')

def now():return datetime.now(timezone.utc).isoformat()
def jhash(obj):return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def readj(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def writej(p,obj):
    p=Path(p); tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def load_cases(path):
    obj=readj(path);cases=obj['cases']
    if len({r['case_id'] for r in cases})!=len(cases):raise ValueError('重复 case_id')
    for c in cases:
        if not isinstance(c['question'],str) or not isinstance(c['gold_ids'],list):raise ValueError('题库内容不合法')
    return cases

def corpus_fingerprint(db_path):
    db=Path(db_path).resolve()
    if not db.is_file():raise FileNotFoundError(f'找不到知识库: {db}')
    with sqlite3.connect('file:'+db.as_posix()+'?mode=ro',uri=True,timeout=30) as con:
        n_docs=con.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
        n_chunks=con.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
        doc_rows=con.execute('SELECT id,filename,sha256 FROM documents ORDER BY id').fetchall()
        integrity=con.execute('PRAGMA quick_check').fetchone()[0]
    if integrity!='ok':raise RuntimeError(f'知识库校验未通过: {integrity}')
    return {'documents':n_docs,'chunks':n_chunks,'documents_sha256':jhash(doc_rows),'db':str(db)}

def api_get(url,timeout):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'Accept':'application/json'}),timeout=timeout) as r:
        return json.load(r)
def api_chat(url,payload,timeout):
    data=json.dumps(payload,ensure_ascii=False).encode('utf-8')
    req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=timeout) as r:
        o=json.load(r)
    if not isinstance(o,dict) or o.get('error') or not isinstance(o.get('sources'),list):
        raise RuntimeError(f'不完整 HTTP 响应: {str(o)[:200]}')
    return o

def gpu_snapshot():
    try:
        s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],timeout=8,text=True,stderr=subprocess.DEVNULL).splitlines()[0]
        a=[float(x.strip()) for x in s.split(',')[:4]]
        return {'time':now(),'used_MiB':a[0],'total_MiB':a[1],'temperature_C':a[2],'utilization_pct':a[3]}
    except (OSError,ValueError,IndexError,subprocess.SubprocessError):
        return {'time':now(),'used_MiB':None,'total_MiB':None}

def make_jobs(cases,arms,seed,pilot):
    if pilot:
        cases=[r for r in cases if r['phase']=='baseline' and r['parent_id'] in FOCUS]
        arms=['rag','agentic']
    rng=random.Random(seed);cases=list(cases);rng.shuffle(cases)
    jobs=[]
    for c in cases:
        for arm in rng.sample(arms,len(arms)):
            jobs.append({'job_id':c['case_id']+'::'+arm,'case':c,'arm':arm})
    return jobs,arms

def score(c,o):
    src=o.get('sources',[])
    found=sorted({int(s['document_id']) for s in src if isinstance(s.get('document_id'),int)})
    gold=set(c['gold_ids'])
    steps=[e for e in (o.get('trace') or {}).get('steps',[]) if e.get('action')=='search']
    ans=str(o.get('answer') or '')
    cites=[int(x) for x in re.findall(r'\[(\d+)\]',ans)]
    trace=o.get('trace') or {}
    return {'retrieved_document_ids':found,'matched_gold':sorted(gold&set(found)),
            'gold_coverage':len(gold&set(found))/len(gold) if gold else None,
            'all_gold':gold.issubset(set(found)) if gold else None,
            'citation_numbers_valid':bool(cites) and all(1<=x<=len(src) for x in cites),
            'citation_count':len(cites),
            'search_calls':len(steps),'zero_gain_search_calls':sum(not s.get('new_citations') for s in steps),
            'stop_reason':trace.get('stop_reason'),'evidence_gaps':trace.get('evidence_gaps'),
            'uncovered_entities':trace.get('uncovered_entities'),
            'planner_failures':trace.get('planner_errors',trace.get('planner_failures',0))}

def percentile(vals,p):
    vals=sorted(vals)
    return vals[round((len(vals)-1)*p)] if vals else None

def summary(records,jobs,manifest):
    latest={}
    for r in records:
        if r['status']=='ok':latest[r['job_id']]=r
    by_arm=defaultdict(list)
    for r in latest.values():by_arm[(r['phase'],r['arm'])].append(r)
    groups={}
    for (phase,arm),rs in sorted(by_arm.items()):
        covers=[r['gold_coverage'] for r in rs if r.get('gold_coverage') is not None]
        calls=sum(r.get('search_calls',0) for r in rs)
        groups[f'{phase}/{arm}']={'n':len(rs),'coverage_mean':statistics.mean(covers) if covers else None,
            'p50_ms':percentile([r['elapsed_ms'] for r in rs],0.5),
            'p95_ms':percentile([r['elapsed_ms'] for r in rs],0.95),
            'search_calls':calls,'no_gain':sum(r.get('zero_gain_search_calls',0) for r in rs),
            'stop_reasons':dict(Counter(r.get('stop_reason') for r in rs if r.get('stop_reason')))}
    pairs={}
    for arm in ('agentic','agentic_hybrid'):
        deltas=[]; times=[]
        for case in {j['case']['case_id'] for j in jobs if j['case']['phase']=='baseline'}:
            a,b=latest.get(case+'::'+arm),latest.get(case+'::rag')
            if a and b and a['gold_coverage'] is not None and b['gold_coverage'] is not None:
                deltas.append(a['gold_coverage']-b['gold_coverage']);times.append(a['elapsed_ms']-b['elapsed_ms'])
        if deltas:
            rng=random.Random(42)
            boot=sorted(statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(2000))
            pairs[arm]={'n_unique_baseline_cases':len(deltas),'coverage_delta_mean':statistics.mean(deltas),
                        'bootstrap_95pct_ci':[boot[49],boot[1949]],'mean_extra_ms':statistics.mean(times)}
    return {'attempts':len(records),'planned':len(jobs),'completed':len(latest),
            'errors':sum(r['status']=='error' for r in records),
            'groups':groups,'paired':pairs,'corpus':manifest['corpus']}

def reporting(out,records,jobs,manifest,state):
    s=summary(records,jobs,manifest)
    writej(out/'summary.json',s)
    writej(out/'status.json',{'state':state,'updated':now(),'run_label':manifest['run_label'],
        'completed':s['completed'],'planned':s['planned'],'attempts':s['attempts'],'errors':s['errors']})
    latest={}
    for r in records:
        if r['status']=='ok':latest[r['job_id']]=r
    with (out/'paired_cases.csv').open('w',newline='',encoding='utf-8-sig') as fp:
        w=csv.writer(fp);w.writerow(['phase','case_id','split','rag_coverage','agentic_coverage','agentic_hybrid_coverage',
           'rag_ms','agentic_ms','agentic_hybrid_ms','agentic_docs','rag_docs','agentic_stop'])
        for cid,c in sorted({(j['case']['case_id'],j['case']['phase']):j['case'] for j in jobs}.items()):
            rows={a:latest.get(cid[0]+'::'+a,{}) for a in ARMS}
            w.writerow([c['phase'],cid[0],c['split']]+[rows[a].get('gold_coverage') for a in ARMS]
             +[rows[a].get('elapsed_ms') for a in ARMS]
             +[rows['agentic'].get('retrieved_document_ids'),rows['rag'].get('retrieved_document_ids'),rows['agentic'].get('stop_reason')])
    lines=['# v12.4 今晚实验自动报告','',f'- 状态：**{state}**','- 完成：%s/%s；错误尝试：%s' %(s['completed'],s['planned'],s['errors']),
           '- 正式比较只以 91 道 baseline 独立问题为样本；stress 为同题改写，catalog 为定位审计。',
           '- Gold 文献覆盖/引用编号合法并不等于医学事实正确。','',
           '## 分组', '', '|组|样本|覆盖率|P50(ms)|P95(ms)|无增益搜索|','|---|---:|---:|---:|---:|---:|']
    for key,g in s['groups'].items():
        lines.append(f"|{key}|{g['n']}|{g['coverage_mean']:.2%}|{g['p50_ms']:.0f}|{g['p95_ms']:.0f}|{g['no_gain']}/{g['search_calls']}|")
    lines+=['','## 配对比较（Agentic − RAG）','']
    for arm,p in s['paired'].items():
        ci=p['bootstrap_95pct_ci'];lines.append(f"- {arm}: {p['n_unique_baseline_cases']} 个独立配对，覆盖差 {p['coverage_delta_mean']:+.2%}，题目重采样95%区间 [{ci[0]:+.2%}, {ci[1]:+.2%}]，平均额外延迟 {p['mean_extra_ms']:.0f}ms。")
    lines+=['','## 重点人工核验','',
           '- v9_032：Gold 指定文献 56 的命中与证据充分停止是否合理。',
           '- v9_042、v9_097：是否仍出现 planner_error。',
           '- v9_089：Focus Pro/BESS 两篇论文是否均有直接证据。',
           '- compare_083_084_not_comparable：是否混淆主动脉夹层与 ACS。',
           '- 文章 Results 和 References 的归属，医学数值/效果不能仅靠 Gold 文献覆盖判断。','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return s

def execute(args,client_get=api_get,client_chat=api_chat,gpu=gpu_snapshot):
    cases=load_cases(args.cases)
    arms=[x.strip() for x in args.arms.split(',') if x.strip()]
    if not arms or len(set(arms))!=len(arms) or any(a not in ARMS for a in arms):raise ValueError('无效实验分组')
    jobs,arms=make_jobs(cases,arms,args.seed,args.pilot)
    if args.dry_run:
        return {'dry_run':True,'jobs':len(jobs),'case_count':len({j['case']['case_id'] for j in jobs}),
             'arms':arms,'phase_counts':dict(Counter(j['case']['phase'] for j in jobs))}
    corpus=corpus_fingerprint(args.db)
    if corpus['documents']!=116 or corpus['chunks']!=14629:
        raise RuntimeError(f"知识库不是冻结的 116篇/14629块：{corpus}")
    host=args.url.rstrip('/')
    if not re.match(r'^http://(?:127\.0\.0\.1|localhost):\d+$',host):
        raise ValueError('为防止数据外发，仅允许 localhost HTTP 服务')
    stats=client_get(host+'/api/stats',10)
    if stats.get('device')!='cuda' or stats.get('llm_configured') is not True:
        raise RuntimeError('CUDA/LLM 未就绪')
    if stats.get('documents')!=corpus['documents'] or stats.get('chunks')!=corpus['chunks']:
        raise RuntimeError('服务和本地评测数据库不一致')
    if Path(stats.get('db','')).resolve()!=Path(args.db).resolve():
        raise RuntimeError('评测 DB 不是当前 HTTP 服务的知识库')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    mpath=out/'manifest.json'
    sig=jhash({'cases':cases,'arms':arms,'seed':args.seed,'pilot':args.pilot,'corpus':corpus['documents_sha256']})
    if mpath.exists():
        manifest=readj(mpath)
        if manifest['signature']!=sig or manifest['url']!=host or manifest['run_label']!=args.run_label:
            raise RuntimeError('配置或版本已变化；禁止混入同一个实验目录')
    else:
        manifest={'created':now(),'signature':sig,'url':host,'run_label':args.run_label,'corpus':corpus,
          'max_hours':args.hours,'deadline':time.time()+3600*args.hours,
          'pilot':args.pilot,'arms':arms,'plan_length':len(jobs),'cases_sha256':jhash(cases)}
        writej(mpath,manifest)
    all_records=[];path=out/'results.jsonl'
    if path.exists():
        with path.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():all_records.append(json.loads(line))
    done={r['job_id'] for r in all_records if r['status']=='ok'}
    failed=Counter(r['job_id'] for r in all_records if r['status']=='error')
    state='completed';error_streak=0
    with path.open('a',encoding='utf-8') as fp:
        for job in jobs:
            if job['job_id'] in done or failed[job['job_id']]>=2:continue
            if time.time()+args.timeout+15>manifest['deadline']:
                state='time_cap';break
            if args.max_jobs and len(done)>=args.max_jobs:
                state='job_cap';break
            c=job['case'];arm=job['arm'];mode,agentic=ARMS[arm]
            payload={'question':c['question'],'mode':mode,'agentic':agentic,'use_llm':True}
            ts=time.perf_counter()
            try:
                result=client_chat(host+'/api/chat',payload,args.timeout)
                row={'status':'ok','job_id':job['job_id'],'case_id':c['case_id'],'parent_id':c['parent_id'],
                 'phase':c['phase'],'split':c['split'],'arm':arm,'question':c['question'],
                 'gold_ids':c['gold_ids'],'reference':c.get('reference',''),
                 'answer':result.get('answer',''),'sources':result.get('sources',[]),
                 'trace':result.get('trace',{}),'elapsed_ms':round((time.perf_counter()-ts)*1000,2),
                 **score(c,result)}
                done.add(job['job_id']);error_streak=0
            except Exception as exc:
                row={'status':'error','job_id':job['job_id'],'case_id':c['case_id'],
                     'parent_id':c['parent_id'],'phase':c['phase'],'split':c['split'],
                     'arm':arm,'error':f'{type(exc).__name__}: {str(exc)[:350]}',
                     'elapsed_ms':round((time.perf_counter()-ts)*1000,2)}
                failed[job['job_id']]+=1;error_streak+=1
            row['time']=now()
            fp.write(json.dumps(row,ensure_ascii=False)+'\n');fp.flush();os.fsync(fp.fileno())
            all_records.append(row)
            print(f"[{len(done)}/{len(jobs)}] {row['status']} {c['case_id']} {arm} {row['elapsed_ms']:.0f}ms gold={row.get('gold_coverage')}",flush=True)
            if len(all_records)%args.checkpoint_every==0:
                reporting(out,all_records,jobs,manifest,'running')
                gpu_row=gpu()
                with (out/'gpu_metrics.jsonl').open('a',encoding='utf-8') as g:g.write(json.dumps(gpu_row)+'\n')
                if gpu_row.get('total_MiB') and gpu_row['total_MiB']-gpu_row['used_MiB']<256:
                    state='gpu_low_margin';break
            if error_streak>=3:state='consecutive_errors';break
        else:
            if len(done)!=len(jobs):state='completed_with_errors'
    report=reporting(out,all_records,jobs,manifest,state)
    print(f"END {state}: completed {report['completed']}/{report['planned']} errors={report['errors']}",flush=True)
    return report|{'state':state}

def cli(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--cases',type=Path,default=Path(__file__).with_name('v121_frozen_cases.json'))
    p.add_argument('--db',type=Path,default=Path('mhrag_gpu_chat_musa/data/knowledge.sqlite'))
    p.add_argument('--url',default='http://127.0.0.1:7866')
    p.add_argument('--run-label',default='v12.4_real_4090')
    p.add_argument('--out',type=Path,default=Path('results/night_v124_20261031'))
    p.add_argument('--arms',default='rag,agentic,agentic_hybrid')
    p.add_argument('--seed',type=int,default=20261031)
    p.add_argument('--hours',type=float,default=9)
    p.add_argument('--timeout',type=int,default=240)
    p.add_argument('--checkpoint-every',type=int,default=10)
    p.add_argument('--max-jobs',type=int,default=0)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if not 0<a.hours<=24 or not 10<=a.timeout<=600 or a.checkpoint_every<1:p.error('hours/timeout/checkpoint 超限')
    try:
        result=execute(a)
        if a.dry_run:print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0 if a.dry_run or result['state'] in ('completed','time_cap','job_cap') else 2
    except Exception as e:
        print(f'EXPERIMENT FAILED: {type(e).__name__}: {e}',file=sys.stderr);return 2
if __name__=='__main__':sys.exit(cli())
