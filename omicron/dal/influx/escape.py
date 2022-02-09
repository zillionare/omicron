import warnings

# Special characters documentation:
# https://docs.influxdata.com/influxdb/v1.4/write_protocols/line_protocol_reference/#special-characters
# Although not in the official docs, new line characters are removed in order to avoid issues.
# Go implementation: https://github.com/influxdata/influxdb/blob/master/pkg/escape/strings.go

# for field key
KEY_ESCAPE = str.maketrans({"\\": "\\\\", ",": r"\,", " ": r"\ ", "=": r"\=", "\n": ""})

# for both tag names and tag values
TAG_ESCAPE = str.maketrans({"\\": "\\\\", ",": r"\,", " ": r"\ ", "=": r"\=", "\n": ""})

# for field value in str-type
STR_ESCAPE = str.maketrans({"\\": "\\\\", '"': r"\"", "\n": ""})
MEASUREMENT_ESCAPE = str.maketrans({"\\": "\\\\", ",": r"\,", " ": r"\ ", "\n": ""})


def escape(string: str, escape_pattern):
    """Assistant function for string escaping

    Args:
        string: string to be escaped
        escape_pattern: str.maketrans() pattern
    """
    return string.translate(escape_pattern)
