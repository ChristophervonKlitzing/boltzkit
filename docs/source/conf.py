import os
import sys


import importlib
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList


class AutoClassAll(Directive):
    required_arguments = 1  # module name

    def run(self):
        module_name = self.arguments[0]
        module = importlib.import_module(module_name)

        content = []

        for name in getattr(module, "__all__", []):
            full_name = f"{module_name}.{name}"

            content.extend(
                [
                    f".. autoclass:: {full_name}",
                    "   :members:",
                    "",
                ]
            )

        # Turn the generated text into real docutils nodes
        node = nodes.section()
        node.document = self.state.document

        self.state.nested_parse(StringList(content), self.content_offset, node)

        return node.children


def setup(app):
    app.add_directive("autoclass_all", AutoClassAll)


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "boltzkit"
copyright = "2026, Christopher von Klitzing"
author = "Christopher von Klitzing"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # for Google/Numpy docstrings
    "sphinx.ext.autosummary",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

sys.path.insert(0, os.path.abspath("../../src"))


autosummary_generate = True
