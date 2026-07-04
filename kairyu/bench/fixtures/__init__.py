"""Committed tiny fixture datasets (synthetic — never real benchmark content).

They keep the default CPU test suite and `--offline-fixtures` runs hermetic:
every adapter's request-building and scoring paths execute without network,
tokens, or the [bench] extra.
"""
