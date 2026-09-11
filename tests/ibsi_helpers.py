"""Reference-driven benchmark selection, independent of extractor output keys."""

import csv
import math
from decimal import Decimal


def matches_reference(actual, reference, tolerance):
    """Compare finite values, applying reference precision only at zero tolerance.

    Use at least three significant figures, retaining finer precision written
    in the reference. Trailing zeros in plain integers are place holders;
    decimal/scientific notation can explicitly preserve those digits.
    A zero reference with zero tolerance requires an exact zero.
    """
    actual, value, margin = float(actual), float(reference), float(tolerance)
    if not all(math.isfinite(x) for x in (actual, value, margin)) or margin < 0:
        return False
    if margin != 0:
        return value - margin <= actual <= value + margin
    if value == 0:
        return actual == 0
    text = str(reference).strip().lower()
    decimal = Decimal(text)
    if '.' not in text and 'e' not in text:
        decimal = decimal.normalize()
    precision = max(3, len(decimal.as_tuple().digits))
    return float(f'{actual:.{precision}g}') == value


IBSI_I_UNAVAILABLE = frozenset(
    {
        'morph_vol_dens_ombb',
        'morph_area_dens_ombb',
        'morph_vol_dens_mvee',
        'morph_area_dens_mvee',
        'ivh_auc',
    }
)


def load_references(path, tag_column, value_column, *, delimiter, phase, config=None):
    references = {}
    seen = set()
    with open(path, newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream, delimiter=delimiter):
            tag = row[tag_column]
            key = (row.get('filter_id'), tag)
            if not tag or key in seen:
                raise ValueError(f'Missing or duplicate reference tag: {key}')
            seen.add(key)
            value, tolerance = row[value_column], row['tolerance']
            unavailable = (phase == 'I' and tag in IBSI_I_UNAVAILABLE) or (
                phase == 'II' and key == ('8.B', 'stat_qcod')
            )
            if unavailable and (value or tolerance):
                raise ValueError(f'Reference availability changed; review benchmark exception: {key}')
            if not value or not tolerance:
                if not (unavailable and value == tolerance == ''):
                    raise ValueError(f'Unexpected blank reference: {key}')
            elif not (math.isfinite(float(value)) and math.isfinite(float(tolerance)) and float(tolerance) >= 0):
                raise ValueError(f'Invalid reference value/tolerance: {key}')
            if config is None or row.get('filter_id') == config:
                references[tag] = row
    if not references:
        raise ValueError(f'Empty reference selection: {config or path}')
    return references


def select_ibsi_i_references(references, aggr_dim, aggr_method):
    """Select published non-diagnostic rows for one explicit aggregation mode."""
    valid_modes = {
        ('2D', 'AVER'),
        ('2D', 'SLICE_MERG'),
        ('2.5D', 'DIR_MERG'),
        ('2.5D', 'MERG'),
        ('3D', 'AVER'),
        ('3D', 'MERG'),
    }
    if (aggr_dim, aggr_method) not in valid_modes:
        raise ValueError(f'Invalid IBSI aggregation mode: {aggr_dim}, {aggr_method}')
    method = {'AVER': 'averaged', 'SLICE_MERG': 'slice-merged', 'DIR_MERG': 'direction-merged', 'MERG': 'merged'}[
        aggr_method
    ]
    result = {}
    for tag, row in references.items():
        family = row['family']
        if family.startswith('Diagnostics') or tag in IBSI_I_UNAVAILABLE:
            continue
        if '(' in family:
            suffix = (
                f'({aggr_dim}, {method})' if family.startswith(('Co-occurrence', 'Run length')) else f'({aggr_dim})'
            )
            if not family.endswith(suffix):
                continue
        result[tag] = row
    if not result:
        raise ValueError('Empty aggregation reference selection')
    if any(row['family'].startswith('Co-occurrence') for row in references.values()) and not any(
        row['family'].startswith('Co-occurrence') for row in result.values()
    ):
        raise ValueError(f'No texture references for aggregation mode: {aggr_dim}, {aggr_method}')
    return result
