"""Public, stable contracts for v13 plugins. No model/DB dependencies."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Sequence, Any

@dataclass(frozen=True)
class Study:
    document_id: int
    title: str
    anchor: str
    confidence: float = 1.0

@dataclass(frozen=True)
class Resolution:
    studies: tuple[Study, ...] = ()
    unresolved: tuple[str, ...] = ()
    anchors: tuple[str, ...] = ()
    @property
    def complete(self) -> bool:
        return bool(self.studies) and not self.unresolved
    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(sorted({s.document_id for s in self.studies}))

@dataclass(frozen=True)
class Source:
    document_id: int
    chunk_id: int | None
    page: str
    title: str
    snippet: str
    origin: str = 'api'
    @classmethod
    def from_api(cls, data: dict) -> 'Source':
        did = data.get('document_id')
        if type(did) is not int or did < 1:
            raise ValueError('Invalid document_id in API source')
        cid = data.get('chunk_id')
        if type(cid) is not int: cid = None
        return cls(did, cid, str(data.get('page','')),
                   str(data.get('title','')), str(data.get('snippet','')), 'api')

@dataclass
class SearchResult:
    sources: list[Source]
    answer: str = ''
    elapsed_ms: float = 0
    trace: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Assessment:
    status: str
    required_ids: tuple[int, ...]
    observed_ids: tuple[int, ...]
    missing_ids: tuple[int, ...]
    suspicious_ids: tuple[int, ...]
    claim_facts_verified: bool = False  # NEVER automatically True without a grounded claim verifier
    @property
    def structurally_complete(self) -> bool:
        return self.status == 'STUDIES_LOCATED_FACTS_UNVERIFIED'

class Resolver(Protocol):
    def resolve(self, question: str) -> Resolution: ...
class Retriever(Protocol):
    def search(self, question: str, *, agentic: bool = False,
               document_ids: Sequence[int] | None = None) -> SearchResult: ...
class EvidenceVerifier(Protocol):
    def assess(self, resolution: Resolution, sources: Sequence[Source],
               question: str) -> Assessment: ...
class Answerer(Protocol):
    def compose(self, question: str, sources: Sequence[Source],
                assessment: Assessment, fallback_answer: str = '') -> str: ...
class TraceSink(Protocol):
    def emit(self, event: dict[str, Any]) -> None: ...
class SectionReader(Protocol):
    def open(self, doc_id: int, query: str, limit: int = 2) -> list[Source]: ...
