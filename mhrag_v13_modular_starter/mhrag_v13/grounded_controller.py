"""New v13 sidecar controller: retrieval -> original-text slots -> regeneration ->
quote/numeric/model checks -> conservative decision. Existing /api/chat unchanged.
"""
from __future__ import annotations
from dataclasses import asdict
import time
from .controller import AdaptiveController
from .answer import EvidencePacketAnswerer
from .verify import StructuralEvidenceGate
from .evidence_slots import SQLiteEvidenceHarvester, requested_slots
from .grounded_answers import ClaimDraftGenerator, ClaimSupportVerifier, EvidenceStopController, render_answer
from .routing import decide_route

class EvidenceFusion:
    """Deduplicate by original (doc,chunk), keep local and API provenance separate."""
    @staticmethod
    def fuse(api_sources, local_spans):
        grouped={}
        for src in api_sources:
            did=src.get('document_id');cid=src.get('chunk_id')
            if type(did) is not int:continue
            entry=grouped.setdefault(did,{})
            key=(did,cid) if type(cid) is int else (did,src.get('page'),str(src.get('snippet',''))[:80])
            if key not in entry:
                entry[key]={'document_id':did,'chunk_id':cid,'page':src.get('page'),
                            'origins':['api'],'section':'unclassified','excerpt':str(src.get('snippet',''))[:600]}
        for src in local_spans:
            entry=grouped.setdefault(src.document_id,{})
            key=(src.document_id,src.chunk_id)
            if key in entry:
                value=entry[key];value['origins'].append('sqlite_local_open')
                value['section']=src.section;value['excerpt']=src.text[:1000]
                value['evidence_id']=src.evidence_id
            else:
                entry[key]={'document_id':src.document_id,'chunk_id':src.chunk_id,
                            'evidence_id':src.evidence_id,'page':src.page,
                            'section':src.section,'origins':['sqlite_local_open'],
                            'excerpt':src.text[:1000]}
        return {'api_sources':[s for s in api_sources],
                'local_spans':[s.to_dict() for s in local_spans],
                'study_evidence':{str(d):list(nodes.values()) for d,nodes in sorted(grouped.items())},
                'api_doc_ids':sorted({s['document_id'] for s in api_sources if type(s.get('document_id')) is int}),
                'local_doc_ids':sorted({s.document_id for s in local_spans})}

class GroundedController:
    def __init__(self,*,resolver,retriever,db_path,
                 generator=None,claim_verifier=None,harvester=None,
                 stop_controller=None,trace=None,max_http_calls=4):
        self.resolver=resolver
        self.harvester=harvester or SQLiteEvidenceHarvester(db_path)
        self.db_path=db_path
        self.generator=generator
        self.claim_verifier=claim_verifier or ClaimSupportVerifier()
        self.stop=stop_controller or EvidenceStopController()
        self.retrieval=AdaptiveController(resolver,retriever,StructuralEvidenceGate(),
                        EvidencePacketAnswerer(),trace=trace,
                        max_http_calls=max_http_calls)

    def run(self,question,*,evidence_only=False):
        started=time.monotonic()
        # Existing v13 retriever issues real API calls; never reuses a previous answer
        # as the final synthesized answer.
        first=self.retrieval.run(question)
        identity=self.resolver.resolve(question)
        route=decide_route(identity, first['sources'], self.db_path)
        spans,slots=self.harvester.harvest(route.document_ids,question)
        fusion=EvidenceFusion.fuse(first['sources'],spans)
        required=requested_slots(question)
        draft={'claims':[],'limitations':['evidence_only'] if evidence_only else [],
               'diagnostics':[], 'generation_status':'skipped', 'generation_calls':0}
        generation_error=None
        if not evidence_only and route.document_ids and spans and self.generator is not None:
            try:
                draft=self.generator.generate(question,spans,slots)
            except (ValueError,TypeError,KeyError,IndexError,OSError,RuntimeError) as exc:
                generation_error=type(exc).__name__
                draft={'claims':[],'limitations':['model_generation_failed'],
                       'diagnostics':[{'status':'model_error','error_type':type(exc).__name__,
                                       'error_message':str(exc)[:180]}],
                       'generation_status':'failed','generation_calls':0}
        review=self.claim_verifier.review(draft.get('claims',[]),spans)
        decision=self.stop.decide(identity,slots,review,required)
        if route.mode != 'confirmed_named' and not evidence_only:
            if review.get('accepted'):
                decision={'decision':'partial_or_abstain',
                          'reason':('unresolved_named_studies' if route.mode=='partial_named'
                                    else 'provisional_api_sources_not_named_study_identity'),
                          'missing_slots':[], 'unverified_slots':[],
                          'routing_provisional':route.provisional}
            else:
                decision={'decision':'abstain','reason':route.reason,
                          'missing_slots':[], 'unverified_slots':[],
                          'routing_provisional':route.provisional}
        if evidence_only:
            decision={**decision,'decision':'evidence_audit_only','reason':'no_answer_generated'}
        answer=render_answer(review,decision) if not evidence_only else '仅输出结构化证据供审计；未生成医学答案。'
        if review.get('accepted') and route.mode == 'provisional_api' and not evidence_only:
            answer='注意：以下仅来自 API 检索候选文献，尚未确认其就是问题所指研究。\n'+answer
        elif review.get('accepted') and route.mode == 'partial_named' and not evidence_only:
            answer='注意：以下仅涉及已确认的部分研究，未解析的研究不得据此推断。\n'+answer
        return {
            'schema_version':'v13-closure-route-0.7',
            'routing':route.to_dict(),
            'answer':answer,
            'identity':{'required_ids':list(identity.ids),'unresolved':list(identity.unresolved),
                        'complete':identity.complete},
            'requested_slots':list(required),
            'slots':[x.to_dict() for x in slots],
            'decision':decision,
            'draft_claim_count':len(draft.get('claims',[])),
            'accepted_claim_count':len(review['accepted']),
            'rejected_claim_count':len(review['rejected']),
            'claims':review,
            'limitations':draft.get('limitations',[]),
            'generation_error':generation_error or ('NoClaimsReturned' if not evidence_only and draft.get('generation_calls',0)>0 and not draft.get('claims') else None),
            'generation_status':draft.get('generation_status','unknown'),
            'generation_calls':draft.get('generation_calls',0),
            'generation_diagnostics':draft.get('diagnostics',[]),
            'evidence':fusion,
            'retrieval':{'assessment':first['assessment'],'http_calls':first['http_calls'],
                         'attempts':first['attempts'],'wall_ms':first['wall_ms'],
                         'api_document_coverage':first['observed_ids']},
            'medical_claims_verified':False,
            'human_medical_review_required':True,
            'wall_ms':round((time.monotonic()-started)*1000,1),
            'note':'Provisional API source != named study identification. Local Results != API retrieval. Quote/numeric/LLM review != physician fact verification.'
        }
