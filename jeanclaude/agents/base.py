from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    @abstractmethod
    async def run(self, *args: Any, **kwargs: Any) -> Any: ...
