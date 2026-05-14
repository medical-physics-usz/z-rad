AGGR_METHOD_SUFFIX = {
    'AVER': 'avg',
    'DIR_MERG': 'avg',
    'SLICE_MERG': 'comb',
    'MERG': 'comb',
}


def normalized_aggr_dim(aggr_dim):
    return '2_5D' if aggr_dim == '2.5D' else aggr_dim


def format_cm_rlm_feature_names(feature_names, aggr_dim, aggr_method):
    dim_suffix = normalized_aggr_dim(aggr_dim)
    method_suffix = AGGR_METHOD_SUFFIX[aggr_method]
    return tuple(f'{feature}_{dim_suffix}_{method_suffix}' for feature in feature_names)


def format_texture_feature_names(feature_names, aggr_dim):
    dim_suffix = normalized_aggr_dim(aggr_dim)
    return tuple(f'{feature}_{dim_suffix}' for feature in feature_names)
