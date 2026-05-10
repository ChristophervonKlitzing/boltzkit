from docutils import nodes
from docutils.parsers.rst import Directive
import importlib
import pkgutil
import inspect

import boltzkit.targets as targets_pkg
from boltzkit.targets.base import (
    BaseTarget,
    DensityProvider,
    SampleProvider,
    DatasetProvider,
)


def iter_target_classes():
    """Yield (name, class) for all target implementations excluding base."""
    for mod in pkgutil.iter_modules(targets_pkg.__path__):
        if mod.name == "base":
            continue

        module = importlib.import_module(f"{targets_pkg.__name__}.{mod.name}")

        for obj_name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseTarget)
                and obj is not BaseTarget
                and obj.__module__.startswith(targets_pkg.__name__)
            ):
                yield obj_name, obj


def yes_no(val: bool) -> str:
    return "✅" if val else "❌"


class TargetMatrixDirective(Directive):
    has_content = False

    def run(self):
        table = nodes.table()

        tgroup = nodes.tgroup(cols=4)
        table += tgroup

        for _ in range(4):
            tgroup += nodes.colspec(colwidth=1)

        header = nodes.thead()
        row = nodes.row()

        for text in [
            "Name",
            "Admits Density",
            "Admits Sampling",
            "Provides Dataset(s)",
        ]:
            entry = nodes.entry()
            entry += nodes.paragraph(text=text)
            row += entry

        header += row
        tgroup += header

        body = nodes.tbody()

        for name, cls in iter_target_classes():
            row = nodes.row()

            row += self._cell(name)
            row += self._cell(issubclass(cls, DensityProvider))
            row += self._cell(issubclass(cls, SampleProvider))
            row += self._cell(issubclass(cls, DatasetProvider))

            body += row

        tgroup += body
        return [table]

    def _cell(self, value):
        entry = nodes.entry()
        symbol = yes_no(value) if isinstance(value, bool) else value
        entry += nodes.paragraph(text=symbol)
        return entry


def setup(app):
    app.add_directive("target-matrix", TargetMatrixDirective)
    return {"version": "1.0"}
