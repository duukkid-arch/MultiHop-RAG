#!/usr/bin/env python3
"""No-GPU answer-shape checks on blinded CSV; not medical scoring.

Never accesses private unblinding keys, medical gold or online services.
"""
import argparse,csv,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mhrag_v13.question_coverage import question_spec, norm


def flags(question,answer):
    kind=question_spec(question)['kind']
    a=norm(answer)
    if not a:return ['empty_answer']
    fs=[]
    if kind=='universal_safety':
        if not any(w in a for w in ('不能证明','未能证明','没有证明','并未证明','不意味着','不代表','没有足够','不足以','不能据此','无法证明','不支持','没有证据')):
            fs.append('no_direct_universal_yes_no')
    if kind=='comparison':
        if not any(w in a for w in ('不同','区别','相比','分别','而另一','不可直接','不能直接','不具可比','不宜比较','无法直接','无法比较')):
            fs.append('no_explicit_comparison')
    if kind=='document_identification':
        if not any(w in a for w in ('研究设计','随机','综述','指南','观察性','回顾','病例','临床试验','荟萃','系统性','研究类型')):
            fs.append('study_type_not_obvious')
        if not any(w in a for w in ('研究发现','结论','结果','表明','发现','显示','相关','支持','提出')):
            fs.append('finding_not_obvious')
    if kind=='enumeration' and not any(w in a for w in ('电视','视频','游戏','社交媒体','手机','电脑','活动包括','活动类型')):
        fs.append('requested_categories_not_listed')
    # Reference-list contamination risk; diagnostics only, not an automatic clinical verdict.
    if re.search(r'\b(?:CrossRef|PubMed)\b',answer) and re.search(r'\b(?:doi|\d{4},\s*\d+)',answer,re.I):
        fs.append('possible_reference_list_contamination')
    return fs


def audit(questions,rows):
    by_id={x['case_id']:x['question'] for x in questions['cases']}
    output=[]
    for r in rows:
        case_id=r['case_id'];q=by_id.get(case_id)
        if q is None:raise ValueError('case not in questions: '+case_id)
        f=flags(q,r.get('answer',''))
        output.append({'case_id':case_id,'response_id':r.get('response_id',''),
                       'question_kind':question_spec(q)['kind'],
                       'flags':'|'.join(f),'requires_manual_check':'yes',
                       'medically_verified':'NO','answer_preview':r.get('answer','')[:220].replace('\n',' ')})
    return output


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--questions',required=True)
    ap.add_argument('--review-csv',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out)
    if out.exists():raise SystemExit('REFUSED: output exists')
    q=json.loads(Path(args.questions).read_text(encoding='utf-8'))
    with open(args.review_csv,encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if not rows:raise SystemExit('No review rows')
    result=audit(q,rows)
    out.mkdir(parents=True)
    with (out/'answer_shape_flags.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    from collections import Counter
    freq=Counter(flag for r in result for flag in r['flags'].split('|') if flag)
    report={'rows':len(rows),'questions':len(q['cases']),'flags':dict(freq),
            'rows_with_at_least_one_flag':sum(bool(r['flags']) for r in result),
            'clinical_accuracy':'NOT_EVALUATED','method':'heuristic_screening_only_not_ground_truth',
            'blinding_preserved':'Does not read private mapping key'}
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
