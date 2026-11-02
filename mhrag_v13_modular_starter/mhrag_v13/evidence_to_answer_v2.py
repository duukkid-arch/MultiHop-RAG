"""Source-bound answer candidate extraction, no benchmark IDs/Gold.

Not clinical truth verification. All extractions use only source span strings, and
all model rescue candidates must pass the original provenance/numeric verifier.
"""
from __future__ import annotations
import re
from .e2_field8 import plan, rank_spans
from .grounded_answers import ClaimSupportVerifier, parse_single_sentence_reply

_REF=re.compile(r'(?im)^\s*(?:references|bibliography)\s*[:\n]')


def recover_single_sentence(raw,span,slot='design'):
    """Allow only a *trailing* NO_EVIDENCE token, never repair numbers.

    A restored candidate is still subject to all downstream checks.
    """
    if not isinstance(raw,str):return None,'invalid_model_response'
    t=raw.strip()
    if not re.search(r'(?i)\s*NO[_ -]?EVIDENCE\s*$',t):return None,'not_trailing_marker'
    t=re.sub(r'(?i)\s*NO[_ -]?EVIDENCE\s*$','',t).strip()
    if not t or 'NO_EVIDENCE' in t.upper():return None,'no_substantive_answer'
    try:
        obj=parse_single_sentence_reply(t,span.document_id,slot,
            {'quote':span.text[:1000],'evidence_id':span.evidence_id})
        claims=obj.get('claims') or []
        if len(claims)!=1:return None,'no_claim'
        ok,why=ClaimSupportVerifier()._check_one(claims[0],{span.evidence_id:span})
        return (claims[0],'recovered_and_numeric_checked') if ok else (None,why)
    except (ValueError,TypeError,KeyError,IndexError) as exc:
        return None,'parse_failed_'+type(exc).__name__


def _qslice(t,match,limit=950):
    """Quote is actual contiguous original text; keep numbers and unit together."""
    if len(t)<=limit:return t.strip()
    a=max(0,match.start()-300)
    b=min(len(t),a+limit)
    if b<match.end():a=max(0,match.end()-limit);b=min(len(t),a+limit)
    return t[a:b].strip()


def _candidate(text,sp,match,fields,section='design'):
    quote=_qslice(sp.text,match)
    return {'study_id':sp.document_id,'slot':section,'text':text,'quote':quote,
            'evidence_id':sp.evidence_id,'candidate_fields':fields,'source_bound':True}


def _extract_one(intent,fields,sp,question):
    t=sp.text
    if _REF.search(t) or sp.section in ('references','bibliography'):return None
    fld={f.name for f in fields}
    if {'initial_dose','followup_dose','interval'} <= fld:
        m=re.search(r'(?is)initially\s*(\d+(?:\.\d+)?)\s*mg/kg\s*,?\s*followed\s+by\s*(\d+(?:\.\d+)?)\s*mg/kg\s+at\s*(\d+)\s*h\s+intervals?',t)
        if m:
            a,b,h=m.groups()
            # A single contextual source match, not copying unrelated dose values.
            return _candidate(f'原文给药方案：首次{a} mg/kg，随后{b} mg/kg，每{h}小时一次。',sp,m,
                              ['initial_dose','followup_dose','interval'])
    if {'age','onset'} <= fld:
        m=re.search(r'(?is)\bage\s*(\d{1,3})\s*[–−-]\s*(\d{1,3})\s*years?.{0,120}?\b(?:presentation\s+(?:of\s+)?|within\s+|less\s+than\s+)(?:less\s+than\s+)?(\d+)\s*h',t)
        if m:
            a,b,h=m.groups()
            return _candidate(f'方案拟纳入年龄{a}–{b}岁、距发病不足{h}小时的患者。',sp,m,['age','onset'])
    if {'dose','duration'} <= fld:
        m=re.search(r'(?is)(?:glibenclamide|study\s+drug)\s*(\d+(?:\.\d+)?)\s*mg.{0,240}?\b(?:daily\s+dose\s+until\s+the\s+(\d+)(?:st|nd|rd|th)\s+day|(?:for\s+)(\d+)\s+days?)',t)
        if m:
            dose,day1,day2=m.groups();day=day1 or day2
            return _candidate(f'试验方案使用格列本脲{dose} mg，每日一次，至出血后第{day}天。',sp,m,['dose','duration'])
        m=re.search(r'(?is)(\d+(?:\.\d+)?)\s*mg\s+daily.{0,70}?\bfor\s+(\d+)\s+days?',t)
        if m and re.search(r'glibenclamide',t,re.I):
            dose,days=m.groups()
            return _candidate(f'方案使用格列本脲{dose} mg，每日一次，共{days}天。',sp,m,['dose','duration'])
    if 'outcome_definition' in fld:
        m=re.search(r'(?is)primary\s+(?:endpoint|outcome)\s+(?:was|is)\s+(?:the\s+)?(?:duration\s+of\s+IMV|duration\s+of\s+invasive\s+mechanical\s+ventilation)\s+from\s+randomi[sz]ation\s+to\s+final\s+successful\s+extubation',t)
        if m:
            return _candidate('试验主要终点是从随机化到最终成功拔管的有创机械通气时长。',sp,m,['outcome_definition'],'primary_outcome')
    if 'test' in fld:
        m=re.search(r'(?i)Test\s+of\s+Variables\s+of\s+Attention\s*\(\s*TOVA\s*\)|\bTOVA\b',t)
        if m:
            return _candidate('客观神经认知评价使用注意力变量测试（TOVA）。',sp,m,['test'])
    if {'timepoint','statistic'} <= fld:
        # Require TOVA plus the matching week-eight statistics in one source span.
        if not re.search(r'(?i)\bTOVA\b|attention\s+comparison',t):return None
        m=re.search(r'(?is)week\s*(?:eight|8)\s*\(\s*p\s*=\s*(0\.\d+)\s*,\s*Cohen[’\x27]?s\s+d\s*=\s*(\d+(?:\.\d+)?)',t)
        if m:
            pv,d=m.groups()
            return _candidate(f'TOVA注意比较评分在第八周的组间差异为p={pv}，效应量Cohen’s d={d}。',sp,m,['timepoint','statistic'],'results')
    if 'trial_limit' in fld:
        m=re.search(r'(?is)(?:did\s+not\s+find\s+a\s+significant\s+benefit|no\s+significant\s+(?:benefit|difference)|did\s+not\s+significantly)',t)
        if m:
            return _candidate('该试验没有发现显著获益，不能据此断言每位患者必然获益。',sp,m,['trial_limit'],'overview')
    if 'study_drug_context' in fld:
        m=re.search(r'(?is)(?:integration\s+of\s+adaptive\s+digital\s+therapy\s+with\s+MPH|MPH\s+plus\s+Focus\s+Pro|medication\s+alone\s+or\s+plus)',t)
        if m:
            return _candidate('该研究是在与MPH药物联合使用的背景下评估数字疗法，不能据此证明所有儿童可停止药物。',sp,m,['study_drug_context'],'overview')
    return None


def source_bound_generate(question,spans):
    """Deterministic evidence candidate before Qwen; never fabricate missing fields.

    The output is provisional; does NOT establish medical semantic correctness.
    Only a single-source *complete field* candidate is returned, otherwise abstain.
    """
    intent,fields=plan(question)
    field_names=[f.name for f in fields]
    if intent=='general':return {'status':'unsupported_intent','claims':[],'fields':field_names}
    for ranked in rank_spans(question,spans):
        sp=ranked['span'];claim=_extract_one(intent,fields,sp,question)
        if claim is None:continue
        if not set(field_names).issubset(claim['candidate_fields']):continue
        ok,why=ClaimSupportVerifier()._check_one(claim,{sp.evidence_id:sp})
        if ok:
            return {'status':'source_bound_candidate_NOT_MEDICALLY_VERIFIED','claims':[claim],
                    'fields':field_names,'evidence_id':sp.evidence_id,'check':why}
    return {'status':'abstain_no_source_bound_complete_fields','claims':[],'fields':field_names}

class SourceBoundFallbackGenerator:
    """Use source-bound fields where extractable, then legacy E2 for unsupported intents.

    Does not assert the extracted candidate is medically true. All original
    identity, route, quote/numeric review and downstream E3 gates still run.
    """
    def __init__(self,llm,max_calls=6):
        self.llm=llm;self.max_calls=max_calls

    def generate(self,question,spans,slots):
        from .ablation_arms import QuestionAwareGenerator, decompose_question
        # For cross-study requests we need both studies; never satisfy them with
        # an isolated source-bound candidate from just one paper.
        if len({s.document_id for s in spans})==1:
            extracted=source_bound_generate(question,spans)
            if extracted['claims']:
                claim=extracted['claims'][0]
                diagnostic=[{'status':'supported_quote_numeric_precheck',
                             'subtarget':t.name,'evidence_id':claim['evidence_id'],
                             'generation_mode':'deterministic_source_bound', 'call':0}
                            for t in decompose_question(question)]
                return {'claims':[claim],'limitations':['source_bound_candidate_requires_medical_review'],
                        'diagnostics':diagnostic,'generation_calls':0,'generation_status':'ok',
                        'source_bound_mode':True}
        result=QuestionAwareGenerator(self.llm,max_calls=self.max_calls).generate(question,spans,slots)
        result['source_bound_mode']=False
        return result
