import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from scipy.stats import chi2_contingency


def model_eval_cv(models, categorical_clm, numerical_clm, X_train, y_train, kfold=5):
    accuracy = []
    model_name = []
    column_trans = make_column_transformer(
        (OneHotEncoder(), categorical_clm),
        (MinMaxScaler(), numerical_clm),
        remainder="passthrough",
    )

    for model in models:
        pipe = Pipeline([("preprocessor", column_trans), ("classifier", model)])
        cv_result = cross_val_score(
            pipe, X_train, y_train, cv=kfold, scoring="f1_macro"
        )
        accuracy.append(cv_result.mean())
        model_name.append(model.__class__.__name__)

    model_summary = pd.DataFrame({"Model": model_name, "F1_Macro": accuracy})

    return model_summary


def cramers_V(var1, var2):
    crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))
    stat = chi2_contingency(crosstab)[0]
    obs = np.sum(crosstab)
    mini = min(crosstab.shape) - 1
    return np.sqrt(stat / (obs * mini))


def matrix_build(data_labeled):
    rows = []

    for var1 in data_labeled:
        col = []
        for var2 in data_labeled:
            cramers = cramers_V(data_labeled[var1], data_labeled[var2])
            col.append(round(cramers, 2))
        rows.append(col)

    cramers_results = np.array(rows)
    df_cor = pd.DataFrame(
        cramers_results, columns=data_labeled.columns, index=data_labeled.columns
    )

    return df_cor
