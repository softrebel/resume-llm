from pydantic import BaseModel
from typing import Callable


class Step(BaseModel):
    name: str | None = None
    prompt: str
    response: str | None = None
    order: int | None = None
    transform: Callable | None = None


class Pipeline(BaseModel):
    steps: list[Step] = []
