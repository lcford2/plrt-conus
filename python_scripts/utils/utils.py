
def my_groupby(df, keys):
    return df.groupby(keys, group_keys=False)
