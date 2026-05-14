from ..exceptions import DataStructureError
from .glcm import GLCMFeatureGroup
from .gldzm import GLDZMFeatureGroup
from .glrlm import GLRLMFeatureGroup
from .glszm import GLSZMFeatureGroup
from .intensity import (
    IntensityHistogramFeatureGroup,
    IntensityStatisticsFeatureGroup,
    IVHFeatureGroup,
    LocalIntensityFeatureGroup,
)
from .morphology import MorphologyCorrelationFeatureGroup, MorphologyFeatureGroup
from .ngldm import NGLDMFeatureGroup
from .ngtdm import NGTDMFeatureGroup

DEFAULT_GROUP_ORDER = (
    'morphology',
    'local_intensity',
    'intensity_statistics',
    'intensity_histogram',
    'glcm',
    'glrlm',
    'glszm',
    'gldzm',
    'ngtdm',
    'ngldm',
    'ivh',
    'morphology_correlation',
)


FEATURE_GROUPS = {
    group.family: group
    for group in (
        MorphologyFeatureGroup(),
        MorphologyCorrelationFeatureGroup(),
        LocalIntensityFeatureGroup(),
        IntensityStatisticsFeatureGroup(),
        IntensityHistogramFeatureGroup(),
        IVHFeatureGroup(),
        GLCMFeatureGroup(),
        GLRLMFeatureGroup(),
        GLSZMFeatureGroup(),
        GLDZMFeatureGroup(),
        NGTDMFeatureGroup(),
        NGLDMFeatureGroup(),
    )
}


def _normalize_selection(selection, label):
    if selection is None:
        return None

    if isinstance(selection, str):
        values = [selection]
    else:
        values = list(selection)

    if not values:
        raise ValueError(f'No {label} were selected.')

    return values


def all_supported_families(context):
    return [family for family in DEFAULT_GROUP_ORDER if FEATURE_GROUPS[family].supports(context)]


def default_families(context):
    return [family for family in DEFAULT_GROUP_ORDER if FEATURE_GROUPS[family].default_enabled(context)]


def resolve_groups(context, families=None, features=None):
    if families is not None and features is not None:
        raise ValueError("Use either 'families' or 'features', not both.")

    if features is not None:
        requested_features = _normalize_selection(features, 'features')
        matched_groups = []
        canonical_features = []
        seen_groups = set()
        seen_features = set()

        for feature in requested_features:
            matched = False
            for family in DEFAULT_GROUP_ORDER:
                group = FEATURE_GROUPS[family]
                if not group.supports(context):
                    continue
                aliases = group.feature_aliases(context)
                if feature in aliases:
                    matched = True
                    if family not in seen_groups:
                        matched_groups.append(group)
                        seen_groups.add(family)
                    canonical = aliases[feature]
                    if canonical not in seen_features:
                        canonical_features.append(canonical)
                        seen_features.add(canonical)
                    break
            if not matched:
                raise ValueError(f"Feature '{feature}' is not supported for the current extraction settings.")

        return matched_groups, canonical_features

    requested_families = _normalize_selection(families, 'families')
    if requested_families is None:
        selected = default_families(context)
    elif len(requested_families) == 1 and requested_families[0] == 'all':
        selected = all_supported_families(context)
    else:
        selected = requested_families

    groups = []
    seen_families = set()
    for family in selected:
        if family in seen_families:
            continue
        group = FEATURE_GROUPS.get(family)
        if group is None:
            raise ValueError(f"Feature family '{family}' is not supported.")
        if not group.supports(context):
            raise DataStructureError(f"Feature family '{family}' is not supported for the current image shape.")
        groups.append(group)
        seen_families.add(family)

    return groups, None
