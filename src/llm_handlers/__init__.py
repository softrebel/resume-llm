from abc import ABC, abstractmethod
from src._core.pipe import Pipeline


class BaseLLMHandler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def send_prompt(self, prompt: str):
        pass

    def run(self, pipeline: Pipeline) -> dict:
        output = {}
        for step in pipeline.steps:
            response = self.send_prompt(step.prompt)
            if step.transform:
                response = step.transform(response)
            output[step.name] = output.get(step.name, "") + response
        return output
