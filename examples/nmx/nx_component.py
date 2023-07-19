from copy import deepcopy
from typing import Any, Dict, List

import jinja2


class NXComponent:
    def __init__(self, name):
        self.name = name
        self._transformations: List[Dict[str, Any]] = []

    @property
    def transformations(self):
        """Return list of transformations. The earliest transformation is the first element of the list,
        and the last transformation (the one the component depends on) is the last."""
        output = deepcopy(self._transformations)
        for idx, transformation in enumerate(output):
            if idx != 0:
                if "config" in output[idx-1]:
                    parent_name = output[idx-1]['config']['name']
                else:
                    parent_name = output[idx-1]['name']
                transformation["attributes"].append(
                    {
                        "dtype": "string",
                        "name": "depends_on",
                        "values": f"/entry/instrument/{self.name}/transformations/{parent_name}",
                    }
                )
        return output

    def _render(self, template_dir: str, template_file_name: str, **context):
        def get_item(dictionary, key):
            return dictionary.get(key)

        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir + "/"))
        env.filters["get_item"] = get_item
        return env.get_template(template_file_name).render(context)

    def rotate(
        self,
        name: str,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        units: str = "degrees",
    ):
        self._transformations.append(
            self._build_rotation(name, x=x, y=y, z=z, units=units)
        )
        return self

    def _build_rotation(self, name: str, x=0.0, y=0.0, z=0.0, units="degrees"):
        vector = [1.0 if x else 0.0, 1.0 if y else 0.0, 1.0 if z else 0.0]
        assert (
            sum([1 for i in (x, y, z) if i]) <= 1
        ), f"You can only rotate on one axis at a time, got x={x}, y={y}, z={z}"
        return {
            "module": "dataset",
            "config": {"name": f"{name}", "type": "double", "values": x + y + z},
            "attributes": [
                {
                    "dtype": "string",
                    "name": "transformation_type",
                    "values": "rotation",
                },
                {"dtype": "string", "name": "units", "values": units},
                {"dtype": "string", "name": "vector", "values": vector},
            ],
        }

    def rotate_from_nxlog(
        self,
        name: str,
        vector: List[float],
        schema: str,
        topic: str,
        source: str,
        units: str = "degrees",
        offset: List[float] = None,
    ):
        self._transformations.append(
            self._build_nxlog_rotation(name, vector, schema, topic, source, units=units, offset=offset)
        )
        return self

    def _build_nxlog_rotation(
        self,
        name: str,
        vector: List[float],
        schema: str,
        topic: str,
        source: str,
        units: str = "degrees",
        offset: List[float] = None,
    ):
        x, y, z = vector[:3]
        assert (
            sum([1 for i in (x, y, z) if i]) <= 1
        ), f"You can only rotate on one axis at a time, got x={x}, y={y}, z={z}"
        output = {
            "name": name,
            "type": "group",
            "attributes": [
                {"dtype": "string", "name": "NX_class", "values": "NXlog"},
                {
                    "dtype": "string",
                    "name": "transformation_type",
                    "values": "rotation",
                },
                {"dtype": "string", "name": "units", "values": units},
                {"dtype": "double", "name": "vector", "values": vector},
            ],
            "children": [
                {
                  "module": schema,
                  "config": {
                    "source": source,
                    "topic": topic,
                    "dtype": "double",
                    "value_units": units
                  },
                  "attributes": [
                    {
                      "name": "units",
                      "dtype": "string",
                      "values": units
                    }
                  ]
                }
            ]
        }
        if offset:
            output["attributes"].append(
                {"dtype": "double", "name": "offset", "values": offset},
            )
        return output

    def translate(
        self,
        name: str,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        units: str = "m",
    ):
        self._transformations.append(
            self._build_translation(name, x=x, y=y, z=z, units=units)
        )
        return self

    def _build_translation(self, name: str, x=0.0, y=0.0, z=0.0, units="m"):
        vector = [1.0 if x else 0.0, 1.0 if y else 0.0, 1.0 if z else 0.0]
        assert (
            sum([1 for i in (x, y, z) if i]) <= 1
        ), f"You can only translate on one axis at a time, got x={x}, y={y}, z={z}"
        return {
            "module": "dataset",
            "config": {"name": f"{name}", "type": "double", "values": x + y + z},
            "attributes": [
                {
                    "dtype": "string",
                    "name": "transformation_type",
                    "values": "translation",
                },
                {"dtype": "string", "name": "units", "values": units},
                {"dtype": "string", "name": "vector", "values": vector},
            ],
        }

    def translate_from_nxlog(
        self,
        name: str,
        vector: List[float],
        schema: str,
        topic: str,
        source: str,
        units: str = "m",
        offset: List[float] = None,
    ):
        self._transformations.append(
            self._build_nxlog_translation(name, vector, schema, topic, source, units=units, offset=offset)
        )
        return self

    def _build_nxlog_translation(
        self,
        name: str,
        vector: List[float],
        schema: str,
        topic: str,
        source: str,
        units: str = "m",
        offset: List[float] = None,
    ):
        x, y, z = vector[:3]
        assert (
            sum([1 for i in (x, y, z) if i]) <= 1
        ), f"You can only translate on one axis at a time, got x={x}, y={y}, z={z}"
        output = {
            "name": name,
            "type": "group",
            "attributes": [
                {"dtype": "string", "name": "NX_class", "values": "NXlog"},
                {
                    "dtype": "string",
                    "name": "transformation_type",
                    "values": "translation",
                },
                {"dtype": "string", "name": "units", "values": units},
                {"dtype": "double", "name": "vector", "values": vector},
            ],
            "children": [
                {
                  "module": schema,
                  "config": {
                    "source": source,
                    "topic": topic,
                    "dtype": "double",
                    "value_units": units
                  },
                  "attributes": [
                    {
                      "name": "units",
                      "dtype": "string",
                      "values": units
                    }
                  ]
                }
            ]
        }
        if offset:
            output["attributes"].append(
                {"dtype": "double", "name": "offset", "values": offset},
            )
        return output
