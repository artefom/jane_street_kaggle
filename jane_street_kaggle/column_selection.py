import re

__all__ = ['FEATURE_PATTERN', 'TARGET_PATTERN', 'get_columns_by_pattern']

FEATURE_PATTERN = re.compile(r'^feature_\d+$')
TARGET_PATTERN = re.compile(r'^resp(:?_\d+)?$')


def get_columns_by_pattern(columns, pattern: re.Pattern):
    rv = list()
    for column_name in columns:
        match = pattern.match(column_name)
        if match is None:
            continue
        rv.append(column_name)
    return rv
