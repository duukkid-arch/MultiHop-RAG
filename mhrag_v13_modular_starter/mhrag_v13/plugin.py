"""Explicit dependency injection: plugins are registered in Python, not eval'd from user input."""
from __future__ import annotations

class Registry:
    def __init__(self):
        self._factories = {}
    def register(self, category: str, name: str, factory):
        if not category or not name or not callable(factory):
            raise ValueError('category, name and callable factory required')
        key=(category, name)
        if key in self._factories:
            raise ValueError(f'duplicate plugin: {key}')
        self._factories[key] = factory
    def create(self, category: str, name: str, **kwargs):
        key=(category, name)
        if key not in self._factories:
            raise KeyError(f'unknown plugin {key}; available: {self.names(category)}')
        return self._factories[key](**kwargs)
    def names(self, category: str):
        return sorted(n for c,n in self._factories if c == category)
