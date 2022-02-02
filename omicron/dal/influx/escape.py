import re

import numpy as np
from numpy.typing import ArrayLike

"""
follow these rules to escape field values:

Measurement    Comma, Space
Tag key    Comma, Equals Sign, Space
Tag value    Comma, Equals Sign, Space
Field key    Comma, Equals Sign, Space
Field value    Double quote, Backslash
"""


def escape_field_value(values: ArrayLike):
    for char in ("\\", '"'):
        values = [v.replace(char, f"\\{char}") for v in values]
    return values


def escape_tag_name(tag_names: ArrayLike):
    return _escape_comma_equal_space(tag_names)


def escape_tag_value(tag_values: ArrayLike):
    return _escape_comma_equal_space(tag_values)


def escape_field_name(field_names: ArrayLike):
    return _escape_comma_equal_space(field_names)


def escape_measurement(measurement: str):
    return measurement.replace(",", "\\,").replace(" ", "\\ ")


def _escape_comma_equal_space(values: ArrayLike):
    # elementwise re.sub is 3~5 times slower than direct python replace
    # pat = re.compile(r"([,= ])")
    for char in (",", "=", " "):
        # values = np.char.replace(values, char, f"\{char}")
        values = [value.replace(char, f"\\{char}") for value in values]
        # values = [pat.sub(r"\\\1", value) for value in values]

    return values
