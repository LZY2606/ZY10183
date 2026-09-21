"""频弧辨证台 (EIS circuit discrimination workbench) core package."""

from .circuit import (
    CircuitError,
    canonical,
    elements,
    parse,
    render,
    validate,
    ELEMENTS,
    KIND_SERIES,
    KIND_PARALLEL,
    KIND_ELEMENT,
)
from .impedance import impedance, impedance_simple
from .fitting import FitProblem, FitResult, solve
from .dataio import parse_table, DATA_HEADERS

__all__ = [
    "CircuitError",
    "canonical",
    "elements",
    "parse",
    "render",
    "validate",
    "ELEMENTS",
    "KIND_SERIES",
    "KIND_PARALLEL",
    "KIND_ELEMENT",
    "impedance",
    "impedance_simple",
    "FitProblem",
    "FitResult",
    "solve",
    "parse_table",
    "DATA_HEADERS",
]
