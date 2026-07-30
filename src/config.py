from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

TRAIN_CSV = RAW_DIR / "lending_club_2020_train.csv"
TEST_CSV = RAW_DIR / "lending_club_2020_test.csv"
DGS3_CSV = RAW_DIR / "DGS3.csv"
DGS5_CSV = RAW_DIR / "DGS5.csv"

# term(months) -> maturity-matched risk-free series
TERM_TO_RF_COL = {36: "DGS3", 60: "DGS5"}

# loans with a final, realized outcome (needed to compute realized return R_i)
MATURED_STATUSES = [
    "Fully Paid",
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
]

# outcome is a loss (used for downside / classification diagnostics)
LOSS_STATUSES = [
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
]

# columns needed to build target/return + cohort/grouping, kept separate from model features
OUTCOME_COLS = [
    "id", "term", "issue_d", "grade", "sub_grade", "loan_status", "funded_amnt",
    "total_pymnt", "total_rec_prncp", "total_rec_int", "recoveries",
    "collection_recovery_fee", "last_pymnt_d",
]

# ex-ante (available at the time of the funding decision) features used by the model.
# excludes anything that is only known after origination (payments, balances, hardship,
# settlement, last_*, collections outcomes, etc.) to avoid leakage into the return prediction.
FEATURE_COLS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "addr_state", "dti",
    "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "initial_list_status", "application_type",
    "annual_inc_joint", "dti_joint", "verification_status_joint",
    "tot_coll_amt", "tot_cur_bal",
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m",
    "mths_since_rcnt_il", "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m",
    "max_bal_bc", "all_util", "total_rev_hi_lim", "inq_fi", "total_cu_tl",
    "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt",
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc", "mths_since_recent_bc_dlq",
    "mths_since_recent_inq", "mths_since_recent_revol_delinq",
    "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats",
    "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "pub_rec_bankruptcies", "tax_liens", "tot_hi_cred_lim", "total_bal_ex_mort",
    "total_bc_limit", "total_il_high_credit_limit", "revol_bal_joint",
    "collections_12_mths_ex_med", "mths_since_last_major_derog", "policy_code",
]

# Many of the newer credit-bureau fields (added by LC around 2015-2016) are
# 100% missing for pre-2012 vintages and ~0% missing from 2016 onward. A tree
# model can use "is this NaN" as a near-perfect proxy for "how old is this
# loan", which would let it smuggle in calendar-time information (and any
# era-specific base rate baked into it) without ever seeing a date feature.
# This is the "safe" subset: features whose missingness rate doesn't swing by
# more than 30pp between 2007-2010 and 2016-2018 vintages, verified in
# src/final_model_safe.py to still reproduce the model's performance and its
# cyclical acceptance-rate behavior -- i.e. that behavior is coming from
# genuine borrower-level signal (grade, rate, dti, revol_util, fico, delinquency
# history, ...), not from an era-via-missingness shortcut.
SAFE_FEATURE_COLS = [
    "loan_amnt", "int_rate", "installment", "grade", "sub_grade",
    "home_ownership", "annual_inc", "verification_status", "purpose",
    "addr_state", "dti", "delinq_2yrs", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "initial_list_status",
    "application_type", "annual_inc_joint", "dti_joint",
    "verification_status_joint", "chargeoff_within_12_mths", "delinq_amnt",
    "mths_since_recent_bc_dlq", "tax_liens", "revol_bal_joint",
    "collections_12_mths_ex_med", "mths_since_last_major_derog", "policy_code",
    "term_months", "emp_length_num", "credit_history_months", "fico_avg",
]

CATEGORICAL_COLS = [
    "term", "grade", "sub_grade", "emp_length", "home_ownership",
    "verification_status", "purpose", "addr_state",
    "initial_list_status", "application_type", "verification_status_joint",
]

# Lending Club's servicing cut, taken off every dollar the borrower repays. Net-of-fee
# is the only return an investor actually receives, so it belongs in the measurement
# rather than in a footnote.
SERVICING_FEE = 0.01

RANDOM_STATE = 42

# data snapshot date (max observed last_pymnt_d / last_credit_pull_d is
# 2020-12 / 2020-10). A vintage is only usable for return/Sortino evaluation
# once its FULL nominal term has elapsed by this date -- otherwise the
# "matured" subset for that vintage is just whichever loans happened to
# resolve fastest (early charge-offs + early full prepayments), which is not
# a representative sample of that vintage's eventual outcome.
SNAPSHOT_DATE = pd.Timestamp("2021-01-01")
