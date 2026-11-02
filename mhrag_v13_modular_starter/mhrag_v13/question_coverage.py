"""DEV-only conservative question-coverage gate.

Does NOT infer clinical truth or consume gold labels. Uses text-level heuristics and
source identity to prevent a quote-valid but irrelevant answer from being labelled
complete. All statuses are preliminary and require medical adjudication.
"""
from __future__ import annotations
import re
import unicodedata


def norm(s):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(s or ''))).strip().casefold()


def _has(text, *words):
    t=norm(text)
    return any(norm(w) in t for w in words)


def question_spec(question):
    q=norm(question)
    if any(x in q for x in ('每位', '所有', '必然', '人人', '全部')) and any(x in q for x in ('是否', '有没有', '可否', '能否', '证明')):
        kind='universal_safety'
    elif ('比较' in q or '对比' in q or '两篇' in q or '两项' in q or '分别' in q or '有何不同' in q or '交集与区别' in q) and not ('哪些屏幕活动' in q):
        kind='comparison'
    elif any(x in q for x in ('名为', '定位文件', '正文摘要', '找到名为')):
        kind='document_identification'
    elif ('是否' in q or '有没有' in q or '可否' in q) and any(x in q for x in ('报告', '计算', '统一', '降低百分比')):
        kind='evidence_limit'
    elif '哪些' in q or '什么活动类型' in q:
        kind='enumeration'
    else:
        kind='specific_answer'
    return {'kind':kind,'requires_distinct_studies':kind=='comparison',
            'question':question}


def _statement_kind(text):
    s=norm(text)
    if _has(s,'无法根据','无法判断','不能确定','证据不足','no_evidence'):
        return 'abstention'
    if _has(s,'没有证明','未能证明','不能证明','并未证明','不能据此','不意味着','不代表','不可以','不可直接','不足以','无法计算','不能计算','未报告','没有报告','不支持','无法支持','不能推断'):
        return 'negative_direct'
    if _has(s,'证明','报告','显示','结果','差异','显著','减少','提高','增加','治疗','干预','测量','评估','比较','对照','风险'):
        return 'substantive'
    return 'background'


def _compare_marker(text):
    return _has(text,'相比','相较','不同','区别','异同','分别','而另一','然而','不可直接','不能直接','不具有可比','不能比较','不宜直接')


def _same_claim_key(claim):
    return (claim.get('study_id'), norm(claim.get('text')))


def evaluate(question,accepted,identity=None,routing=None):
    """Return transparent, fail-conservative task coverage. Never a clinical score.

    accepted: prechecked claim entries {'claim':...}; they remain UNVERIFIED medically.
    Every model text is untrusted; do not interpret citations as clinical correctness.
    """
    spec=question_spec(question)
    identity=identity or {}
    routing=routing or {}
    seen=set(); unique=[]; duplicates=[]
    for row in accepted:
        claim=row.get('claim') or {}
        if not isinstance(claim,dict):continue
        key=_same_claim_key(claim)
        if key in seen:
            duplicates.append({'study_id':claim.get('study_id'),'text':claim.get('text')})
            continue
        seen.add(key);unique.append(row)
    relevant=[];off_topic=[]
    q=norm(question); kind=spec['kind']
    for entry in unique:
        claim=entry['claim'];t=norm(claim.get('text')); sk=_statement_kind(t)
        slot=claim.get('slot')
        good=False
        if kind=='universal_safety':
            # Generic RCT / disease definitions NEVER answer all-patients claims.
            good=sk=='negative_direct' and _has(t,'所有','每位','全部','人人','必然','停药','药物','获益','受益','适用','人人')
        elif kind=='comparison':
            # One fact about one paper alone is supporting material only, NOT full comparison.
            good=sk!='background' and (
                slot in ('results','primary_outcome') or
                _has(t,'研究对象','受试','患儿','儿童','患者','对照','干预','疾病','综合征','注意力','认知','疗效','机制','治疗','综述','指南','随机','类型','年龄','比较','方法'))
        elif kind=='document_identification':
            good=sk!='abstention' and (_has(t,'研究','综述','试验','随机','疾病','目的','结论','结果','发现','疗法','观察','分析') and slot in ('overview','results','design'))
            # Bare "是一篇综述" or disease definition does not answer question.
            if re.fullmatch(r'(?:这|本|该)?(?:篇)?(?:文章|论文|文献)?(?:是|属于|为)?(?:一篇|一项)?(?:综述|研究|随机对照试验|临床试验)(?:文章)?[。.]?',t):good=False
        elif kind=='evidence_limit':
            good=sk=='negative_direct' and _has(t,'无法','不能','缺少','不具备','不足','未报告','未提供','没有')
        elif kind=='enumeration':
            good=sk!='abstention' and _has(t,'电视','视频','游戏','社交','手机','电脑','屏幕活动','活动类型','类别') and not ('关联' in t and len(t)<35)
        else:
            good=sk in ('negative_direct','substantive') and slot in ('overview','results','design','primary_outcome')
        if good:relevant.append(entry)
        else:off_topic.append(entry)
    all_relevant_docs={r['claim'].get('study_id') for r in relevant if type(r['claim'].get('study_id')) is int}
    all_docs={r['claim'].get('study_id') for r in unique if type(r['claim'].get('study_id')) is int}
    reasons=[]
    if not unique:reasons.append('no_prechecked_claims')
    if duplicates:reasons.append('duplicate_claims_removed')
    if off_topic:reasons.append('generic_or_off_topic_claims')
    if routing.get('provisional') or (identity and not identity.get('complete',False)):
        reasons.append('study_identity_not_fully_confirmed')
    if kind=='comparison':
        if len(all_relevant_docs)<2:reasons.append('missing_two_study_specific_evidence')
        if not any(_compare_marker(e['claim'].get('text','')) for e in relevant):
            reasons.append('no_explicit_comparison_conclusion')
    if kind=='document_identification':
        texts=' '.join(norm(x['claim'].get('text')) for x in relevant)
        # Minimal content dimensions; title/filename alone is insufficient.
        if not _has(texts,'疾病','问题','研究','治疗','症状','目的','主题','关注','探讨','评估','观察','针对'):
            reasons.append('missing_study_question')
        if not _has(texts,'综述','指南','试验','随机','前瞻','回顾','病例','观察性','荟萃','分析','研究设计','实验'):
            reasons.append('missing_study_type')
        if not _has(texts,'发现','结果','表明','显示','指出','关联','结论','显著','有效','无效','支持','提出'):
            reasons.append('missing_main_finding')
    if kind=='universal_safety' and not relevant:
        reasons.append('no_direct_reply_to_universal_claim')
    if kind=='evidence_limit' and not relevant:
        reasons.append('no_direct_evidence_limit_conclusion')
    if kind=='enumeration' and not relevant:
        reasons.append('no_requested_categories_listed')
    reasons=list(dict.fromkeys(reasons))
    # Do NOT mark complete if identity is unresolved or if a question dimension is missing.
    # No status means physician-verified.
    if not relevant:status='no_task_aligned_claim'
    elif reasons:status='partial_task_coverage'
    else:status='candidate_task_covered_NOT_MEDICALLY_VERIFIED'
    return {'schema_version':'question-coverage-v0.9','kind':kind,'status':status,
            'reasons':reasons,'unique_prechecked_claims':len(unique),
            'duplicate_count':len(duplicates),'off_topic_count':len(off_topic),
            'task_aligned_claims':len(relevant),'prechecked_study_ids':sorted(all_docs),
            'task_aligned_study_ids':sorted(all_relevant_docs),
            'medical_accuracy':'NOT_EVALUATED'}, relevant


class CoverageController:
    """A separate v0.9 wrapper; never mutates v0.8 evidence or underlying controller."""
    def __init__(self,base):self.base=base
    def run(self,question,*,evidence_only=False):
        data=self.base.run(question,evidence_only=evidence_only)
        if evidence_only:return {**data,'question_coverage':{'status':'not_applicable_evidence_only'}}
        report,aligned=evaluate(question,(data.get('claims') or {}).get('accepted',[]),data.get('identity'),data.get('routing'))
        original=data.get('decision') or {}
        # Never upgrade provisional claim support into clinically validated answer.
        new_decision=dict(original)
        if report['status']!='candidate_task_covered_NOT_MEDICALLY_VERIFIED':
            new_decision['decision']='partial_or_abstain' if aligned else 'abstain'
            new_decision['reason']='question_coverage_not_satisfied'
        # Do not include generic/duplicate fact claims as substitute for answer.
        lines=[]
        for e in aligned:
            c=e['claim']
            lines.append(f"- {c['text']} [文献 {c['study_id']}，证据 {c['evidence_id']}]（来源与数值初检，非医学核验）")
        if not lines:
            answer='未找到足以直接回应问题核心要求的已初检论断；不以泛泛背景事实代替答案。'
        else:
            answer='\n'.join(lines)
            if report['status']!='candidate_task_covered_NOT_MEDICALLY_VERIFIED':
                answer+='\n回答仍不完整，缺少：'+', '.join(report['reasons'])+'。'
            answer+='\n注意：引用和数值初检不等于医学事实核验；尚需原文及人工审核。'
        return {**data,'answer':answer,'upstream_answer':data.get('answer'),
                'upstream_decision':original,'decision':new_decision,
                'question_coverage':report,
                'schema_version':'v13-question-coverage-experiment-0.9'}
