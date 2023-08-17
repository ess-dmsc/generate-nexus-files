import itertools
from typing import List

try:
    from nx_component import NXComponent
except ModuleNotFoundError:
    from examples.nmx.nx_component import NXComponent


class NXSample(NXComponent):
    def __init__(self, name: str, sample_name: str, instrument_name: str):
        super().__init__(name)
        self.name = name
        self.sample_name = sample_name
        self.instrument_name = instrument_name

    def to_json(self):
        context = {
            "j2_name": self.name,
            "j2_sample_name": self.sample_name,
            "j2_instrument_name": self.instrument_name,
            "j2_sample_transformations": self.transformations,
        }
        return self._render(
            "examples/nmx", "template_NXsample.json.j2", **context
        )
