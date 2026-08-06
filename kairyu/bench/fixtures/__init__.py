"""Committed package data for hermetic benchmark execution.

Eleven tiny synthetic stand-ins keep the default CPU tests and
``--offline-fixtures`` runs hermetic.  The fixed five-row structured-output
conformance corpus and the separately licensed judge-calibration corpus are
real score-bearing package resources, not substitutes for downloaded benchmark
populations.  Adapter-owned scorer dependencies still fail closed when the
``bench`` extra is not installed.
"""
