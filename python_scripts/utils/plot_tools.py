
_PRETTY_VAR_NAMES_MATH = {
    "rts": r"$RT$",
    "max_sto": r"$S_{max}$",
    "rel_inf_corr": r"$r(D, I)$",
    "storage_pre": r"$S_{t-1}$",
    "release_pre": r"$D_{t-1}$",
    "inflow": r"$NI_t$",
}

_PRETTY_VAR_NAMES = {
    "rts": "Residence Time",
    "max_sto": "Max. Storage",
    "rel_inf_corr": "Pearson Cor. for Inflow & Release",
    "storage_pre": "Lag-1 Storage",
    "release_pre": "Lag-1 Release",
    "inflow": "Net Inflow",
}


def get_pretty_var_name(var, math=True):
    if math:
        return _PRETTY_VAR_NAMES_MATH.get(var, var)
    else:
        return _PRETTY_VAR_NAMES.get(var, var)