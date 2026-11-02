"""MultiHop-RAG v13 sidecar modular prototype; safe to run alongside v12.4."""
from .controller import AdaptiveController
from .contracts import Study,Resolution,Source,SearchResult,Assessment
from .plugin import Registry
__all__=['AdaptiveController','Study','Resolution','Source','SearchResult','Assessment','Registry']
