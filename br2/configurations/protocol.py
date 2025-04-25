from typing import Protocol
from pathlib import Path
import json


class ConfigProtocol(Protocol):
    path: Path
    info: dict[str, ...]

    def load(self) -> None: ...

    def save(self) -> None: ...
