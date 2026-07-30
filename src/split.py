"""Three time-ordered blocks carved out of the training file.

Why the split is chronological and not random
---------------------------------------------
The obvious way to make a train/validation/test split is to shuffle the loans
and cut. Loan ids never overlap, so it looks out-of-sample, and it is the setup
behind most reported numbers on this dataset. It does not work here.

Lending Club's credit box changed over time -- a dollar lent in 2013 returned
about 10% above Treasuries, a dollar lent in 2018 lost 5% against them -- and a
shuffled split puts loans from the same origination month on both sides of the
line. The model is then fitted on 2016 loans and asked to price other 2016 loans,
which lets it learn the 2016 base rate instead of learning what makes a borrower
risky. Measured on this data, that single choice inflates the predictive
correlation from 0.02 to 0.13 and the Sharpe ratio of the resulting rule from
1.2 to 1.7. The gap is not the model's skill; it is the model reading a calendar
it was never handed.

So the blocks are contiguous in time: the model learns from the past, its
stopping point is chosen on a later period, and every rule is settled on a
period later still. That is the order in which a lender actually encounters
information.

The blocks
----------
The cut is on VINTAGE COUNT, not loan count, and the difference matters. LC's
volume is heavily back-loaded, so a 60/20/20 cut on loan count puts the 60% mark
at 2016-01 and leaves the final block only 15 origination months. A Sharpe ratio
computed on 15 vintages has an effective sample size near 0.3 -- not enough to
choose a grade cut on, let alone a timing rule. Splitting on vintage count gives
each block a real span of calendar time:

    train  2007-06 .. 2013-09    76 vintages    112,228 loans
    valid  2013-10 .. 2015-11    26 vintages    384,438 loans
    test   2015-12 .. 2018-01    26 vintages    380,114 loans

`train` fits the model. `valid` stops it -- XGBoost needs a held-out series to
decide when to stop adding trees, and using the training block for that quietly
overfits the stopping point. `test` is where every decision in this project gets
made: which grades to lend to, whether the risk-free gate helps, how the timing
dial is parameterised.

`lending_club_2020_test.csv` appears in none of this. It is opened once, at the
end, with every rule already frozen.
"""
import numpy as np
import pandas as pd

TRAIN_FRAC, VALID_FRAC = 0.60, 0.20
BLOCKS = ("train", "valid", "test")


def vintage_boundaries(df, train_frac=TRAIN_FRAC, valid_frac=VALID_FRAC):
    """The two origination months where the blocks change over."""
    months = np.sort(df["issue_month"].unique())
    i_train = int(round(len(months) * train_frac))
    i_valid = int(round(len(months) * (train_frac + valid_frac)))
    return pd.Timestamp(months[i_train]), pd.Timestamp(months[i_valid])


def assign(df, train_frac=TRAIN_FRAC, valid_frac=VALID_FRAC):
    """Label every loan `train`, `valid` or `test` by its origination month."""
    b1, b2 = vintage_boundaries(df, train_frac, valid_frac)
    issue = df["issue_month"]
    return pd.Series(
        np.select([issue < b1, issue < b2], ["train", "valid"], default="test"),
        index=df.index, name="block",
    )


def split(df, **kwargs):
    """(train, valid, test) frames, contiguous in time and disjoint in loans."""
    block = assign(df, **kwargs)
    return tuple(df[block == name].reset_index(drop=True) for name in BLOCKS)


def summary(df, **kwargs):
    """One row per block: how much time and how much money it covers."""
    block = assign(df, **kwargs)
    rows = []
    for name in BLOCKS:
        sub = df[block == name]
        rows.append({
            "블록": name,
            "빈티지수": sub["issue_month"].nunique(),
            "시작": f"{sub['issue_month'].min():%Y-%m}",
            "종료": f"{sub['issue_month'].max():%Y-%m}",
            "대출건수": len(sub),
            "대출금액": float(sub["funded_amnt"].sum()),
            "건수비중": len(sub) / len(df),
        })
    return pd.DataFrame(rows)
