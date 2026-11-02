"""v1.0 DEV question-coverage gate: reject aim-only findings and spurious comparisons.

No gold labels, benchmark IDs, or clinical truths used. All output statuses are
question-coverage candidates, not medical factual accuracy. Additive sidecar.
"""
from __future__ import annotations
import re
from .question_coverage import evaluate as evaluate_v09, CoverageController as _OldController, norm, _has

AIM = re.compile(r'(?:旨在|旨以|目的(?:是|在于)|目标(?:是|为)|力图|试图|计划|尝试|aims?\s+to|seeks?\s+to|intends?\s+to|objective\s+(?:is|was))', re.I)
FINDING = re.compile(r'(?:研究发现|结果(?:表明|显示|发现|提示)|研究(?:表明|显示|指出)|数据(?:表明|显示)|结论(?:指出|认为|显示)|(?:显著|不显著)(?:差异|改善|降低|增加)|未发现(?:显著|统计)|观察到|主要发现(?:为|是)|发现(?:了|到)|\b(?:found|showed|demonstrated|observed|reported)\b)', re.I)
META = re.compile(r'^(?:这|本|该)?(?:篇)?(?:文章|论文|文献)?(?:是|属于|为)?(?:一篇|一项)?(?:综述|研究|随机对照试验|临床试验)(?:文章)?[。.]?$', re.I)
NO_DIRECT_EFFECT = re.compile(r'(?:不能|无法|不足以|不支持|未报告|缺乏|尚无|不可|不宜|没有)(?:.{0,48})(?:直接比较|统一(?:计算|估计)|合并|算出|推断|证明|效应量|百分比|死亡率|必然|全部|所有|停药|获益)', re.I)
CROSS_TYPE = re.compile(r'(?:急性冠脉综合征|急性冠状动脉综合征|acute coronary syndrome|ACS)', re.I)
AORTIC = re.compile(r'(?:主动脉夹层|aortic dissection|TBAD|Type B aortic)', re.I)
EQUATE = re.compile(r'(?:同一种(?:疾病|病)|相同(?:疾病|诊断)|均是同一|属于同一(?:疾病|病))')


def _is_purpose_only(s):
    t=norm(s)
    # A review can be about 'results' without *reporting* a finding.
    return bool(AIM.search(t) and not FINDING.search(t))


def _is_finding(s):
    t=norm(s)
    return bool(FINDING.search(t) and not _is_purpose_only(t))


def _study_specific(s):
    t=norm(s)
    if META.fullmatch(t) or _is_purpose_only(t):
        return False
    if re.search(r'(?:randomi[sz]ed|随机|双盲|对照组|实验组|干预组|患儿|参与者|受试者|脑|症状|疗效|结局|时间|置信区间|\bCI\b|\bp\s*[=<]|系统性红斑狼疮|心肌|ADHD|TBAD)',t,re.I):
        return True
    return bool(_is_finding(t))


def question_kind(q):
    t=norm(q)
    # Classify explicit study-document identification before generic answers,
    # even when the sentence says '定位这篇文章' instead of '定位文件'.
    if (any(x in t for x in ('研究问题','研究的问题','主要发现','主要结论')) and
        any(x in t for x in ('文章','论文','文献','文件','综述')) and
        any(x in t for x in ('研究类型','研究设计','研究问题','研究的问题','正文摘要','定位','找到','主要发现'))):
        return 'document_identification'
    # More specific evidence-limit request must win over generic 'all' / universal.
    if (('是否' in t or '能否' in t or '可以' in t or '可否' in t) and
        any(k in t for k in ('统一死亡率','统一效应','计算','合并效应','直接比较治疗效果','死亡率降低百分比','相对于全部传统'))):
        return 'evidence_limit'
    return None


def evaluate(question, accepted, identity=None, routing=None):
    base, aligned=evaluate_v09(question,accepted,identity,routing)
    kind=question_kind(question) or base['kind']
    reasons=list(base['reasons'])
    relevant=list(aligned)
    def add(r):
        if r not in reasons: reasons.append(r)
    def drop(r):
        if r in reasons:reasons.remove(r)
    if kind=='document_identification':
        # Aims/objectives are not outcomes. Require a separate asserted finding.
        docs=[e for e in accepted if isinstance(e,dict) and isinstance(e.get('claim'),dict)]
        if not any(_is_finding(e['claim'].get('text','')) for e in docs):
            add('missing_asserted_main_finding_not_just_study_aim')
        # Do not treat aim-only sentences as standalone substantive findings.
        relevant=[e for e in relevant if not _is_purpose_only(e['claim'].get('text',''))]
        if docs and any(_is_purpose_only(e['claim'].get('text','')) for e in docs):
            add('purpose_is_not_an_observed_finding')
    if kind=='comparison':
        # Two source docs must both provide concrete study-specific content.
        specific={e['claim'].get('study_id') for e in relevant
                  if isinstance(e['claim'].get('study_id'),int) and _study_specific(e['claim'].get('text',''))}
        if len(specific)<2: add('missing_two_study_specific_content')
        # An unrelated 'compared' word or a one-study result does not prove a
        # two-study contrast. This is a conservative task-coverage guard.
        if not any(re.search(r'(?:两项|两篇|两者|相比|相较|不同|区别|不可直接|不能直接|不能比较|不宜直接|各自|分别)',
                             str(e['claim'].get('text',''))) for e in relevant):
            add('missing_explicit_two_study_contrast')
        relevant=[e for e in relevant if _study_specific(e['claim'].get('text',''))]
    if kind=='evidence_limit':
        # Model must directly address non-comparability, missing outcome or
        # insufficient data, not merely cite a paper's topic.
        relevant=[e for e in accepted if isinstance(e,dict) and isinstance(e.get('claim'),dict)
                  and NO_DIRECT_EFFECT.search(str(e['claim'].get('text','')))]
        if not relevant:add('no_direct_evidence_limit_conclusion')
        # Old wrong-kind universal-safety reasons are not applicable.
        drop('no_direct_reply_to_universal_claim')
        if not relevant: add('no_task_aligned_claim')
    if CROSS_TYPE.search(question) and AORTIC.search(question):
        for e in accepted:
            t=str((e.get('claim') or {}).get('text',''))
            if CROSS_TYPE.search(t) and AORTIC.search(t) and EQUATE.search(t):
                add('potential_disease_identity_conflation')
                relevant=[x for x in relevant if x is not e]
    if not relevant:
        status='no_task_aligned_claim'
    elif reasons:
        status='partial_task_coverage'
    else:
        status='candidate_task_covered_NOT_MEDICALLY_VERIFIED'
    base={**base,'schema_version':'question-coverage-v1.0', 'kind':kind,
          'status':status,'reasons':reasons,'task_aligned_claims':len(relevant),
          'task_aligned_study_ids':sorted({e['claim']['study_id'] for e in relevant
                                if type(e['claim'].get('study_id')) is int}),
          'medical_accuracy':'NOT_EVALUATED'}
    return base,relevant


class CoverageController(_OldController):
    def run(self,question,*,evidence_only=False):
        # v0.9 already applies the old gate; reconstruct its immutable prechecked
        # accepted entries and apply stricter question-level checks afterwards.
        old=self.base.run(question,evidence_only=evidence_only)
        if evidence_only:return {**old,'question_coverage':{'status':'not_applicable_evidence_only'}}
        report,aligned=evaluate(question,(old.get('claims') or {}).get('accepted',[]),
                                old.get('identity'),old.get('routing'))
        orig=old.get('decision') or {}
        final=dict(orig)
        if report['status']!='candidate_task_covered_NOT_MEDICALLY_VERIFIED':
            final['decision']='partial_or_abstain' if aligned else 'abstain'
            final['reason']='question_coverage_v10_not_satisfied'
        lines=[]
        seen=set()
        for entry in aligned:
            c=entry['claim']; key=(c.get('study_id'),norm(c.get('text')))
            if key in seen: continue
            seen.add(key)
            lines.append(f"- {c['text']} [文献 {c['study_id']}，证据 {c['evidence_id']}]（引文/数值初检，非医学核验）")
        if not lines:
            answer='没有足以直接回答本题的已初检结论。可标注证据缺口，不能用背景知识冒充结果。'
        else:
            answer='\n'.join(lines)
            if report['status']!='candidate_task_covered_NOT_MEDICALLY_VERIFIED':
                answer+='\n回答尚不完整：'+', '.join(report['reasons'])+'。'
            answer+='\n本回答未经过医学专家核验。'
        return {**old,'answer':answer,'upstream_answer':old.get('answer'),
                'upstream_decision':orig,'decision':final,
                'question_coverage':report,
                'schema_version':'v13-question-coverage-experiment-1.0'}
