from sklearn.metrics import mean_squared_error, r2_score


def nrmse(actual, model):
    return mean_squared_error(actual, model, squared=False) / actual.mean()


def nnse(actual, model):
    return 1 / (2 - r2_score(actual, model))


def get_nse(df, actual, model, grouper=None):
    if grouper:
        scores = df.groupby(grouper).apply(
            lambda x: r2_score(x[actual], x[model])
        )
    else:
        scores = r2_score(df[actual], df[model])
    scores.name = "NSE"
    return scores


def get_nnse(df, actual, model, grouper=None):
    if grouper:
        scores = df.groupby(grouper).apply(lambda x: nnse(x[actual], x[model]))
    else:
        scores = nnse(df[actual], df[model])
    scores.name = "NNSE"
    return scores


def get_rmse(df, actual, model, grouper=None):
    if grouper:
        scores = df.groupby(grouper).apply(
            lambda x: mean_squared_error(x[actual], x[model], squared=False)
        )
    else:
        scores = mean_squared_error(df[actual], df[model], squared=False)
    scores.name = "RMSE"
    return scores


def get_nrmse(df, actual, model, grouper=None):
    if grouper:
        scores = df.groupby(grouper).apply(lambda x: nrmse(x[actual], x[model]))
    else:
        scores = nrmse(df[actual], df[model])
    scores.name = "RMSE"
    return scores
