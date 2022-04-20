import pandas as pd
from functools import reduce


def single_filter(s, f):
    if isinstance(f, list):
        w = s.isin(f)
    else:
        w = s == f
    assert w.sum() > 0, f'Selecting for {f} does not leaves any data.'
    return w


def selector(df, selectors):
    if len(selectors) > 0:
        return reduce(lambda a,b: a & b, (single_filter(df[k], v) for k, v in selectors.items()))
    else:
        return pd.Series(True, index=df.index)
