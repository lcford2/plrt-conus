import numpy as np
import pandas as pd
from utils.io import load_feather

# data, meta = load_resopsus_data()
data = load_feather(
    "../data/model_ready/resops_3yr.feather", index_keys=("res_id", "date")
)

# resers = meta.index
resers = {
    "1020": ["release"],
    "1042": ["release"],
    "1170": ["release"],
    "1777": ["storage"],
    "572": ["release"],
    "616": ["storage"],
    "629": ["storage"],
    "7214": ["release"],
    "870": ["storage"],
    "929": ["release"],
}

for res, fix_vars in resers.items():
    rdf = data.loc[pd.IndexSlice[res, :], ["storage", "release", "net_inflow"]]
    for var in fix_vars:
        q1 = rdf[var].quantile(0.25)
        q3 = rdf[var].quantile(0.75)
        iqr = q3 - q1
        lb = q1 - 1.5 * iqr
        ub = q3 + 1.5 * iqr
        rdf.loc[(rdf[var] < lb) | (rdf[var] > ub), var] = np.nan
        rdf[var] = rdf[var].interpolate()
    rdf["storage_pre"] = rdf["storage"]
    # axes = rdf.plot(subplots=True)
    # plt.suptitle(res)
    # plt.show()
