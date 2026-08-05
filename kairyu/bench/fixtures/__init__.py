"""Committed tiny fixture datasets (synthetic — never real benchmark content).

They keep the default CPU test suite and `--offline-fixtures` runs hermetic:
every adapter's request-building and scoring paths execute without network,
target tokens, or a dataset cache. Adapter-owned scorer dependencies still
fail closed when the ``bench`` extra is not installed.
"""
