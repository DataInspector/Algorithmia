# Copyright 2019 H2O.ai; Proprietary License;  -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
from numpy import nan
from scipy.special._ufuncs import expit
from scoring_h2oai_experiment_4e86cfd8_ba96_11e9_b4c6_0242ac110002 import Scorer

#
# The format of input record to the Scorer.score() method is as follows:
#

# ------------------------------------------------------------
# Name        Type      Range                                 
# ------------------------------------------------------------
# sepal_len   float32   [4.400000095367432, 7.699999809265137]
# sepal_wid   float32   [2.0, 4.199999809265137]              
# petal_len   float32   [1.0, 6.699999809265137]              
# petal_wid   float32   [0.10000000149011612, 2.5]            
# ------------------------------------------------------------


#
# Create a singleton Scorer instance.
# For optimal performance, create a Scorer instance once, and call score() or score_batch() multiple times.
#
scorer = Scorer()


#
# To score one row at a time, use the Scorer.score() method (this can seem slow due to one-time overhead):
#

print('---------- Score Row ----------')
print(scorer.score([
    '5.199999809265137',  # sepal_len
    '2.0',  # sepal_wid
    '1.2999999523162842',  # petal_len
    '1.5',  # petal_wid
]))
print(scorer.score([
    '4.5',  # sepal_len
    '2.0',  # sepal_wid
    '1.2000000476837158',  # petal_len
    '0.20000000298023224',  # petal_wid
]))
print(scorer.score([
    '4.400000095367432',  # sepal_len
    '2.0',  # sepal_wid
    '1.0',  # petal_len
    '1.399999976158142',  # petal_wid
]))
print(scorer.score([
    '4.5',  # sepal_len
    '2.5999999046325684',  # sepal_wid
    '3.299999952316284',  # petal_len
    '1.5',  # petal_wid
]))
print(scorer.score([
    '5.400000095367432',  # sepal_len
    '3.0',  # sepal_wid
    '1.899999976158142',  # petal_len
    '0.10000000149011612',  # petal_wid
]))


#
# To score a batch of rows, use the Scorer.score_batch() method (much faster than repeated one-row scoring):
#
print('---------- Score Frame ----------')
columns = [
    pd.Series(['5.199999809265137', '4.5', '4.400000095367432', '4.5', '5.400000095367432', '4.400000095367432', '4.5', '4.5', '5.0', '4.800000190734863'], name='sepal_len', dtype='float32'),
    pd.Series(['2.0', '2.0', '2.0', '2.5999999046325684', '3.0', '2.799999952316284', '3.0', '2.799999952316284', '2.299999952316284', '2.200000047683716'], name='sepal_wid', dtype='float32'),
    pd.Series(['1.2999999523162842', '1.2000000476837158', '1.0', '3.299999952316284', '1.899999976158142', '1.399999976158142', '1.2000000476837158', '3.299999952316284', '1.899999976158142', '1.2000000476837158'], name='petal_len', dtype='float32'),
    pd.Series(['1.5', '0.20000000298023224', '1.399999976158142', '1.5', '0.10000000149011612', '0.6000000238418579', '1.399999976158142', '0.20000000298023224', '0.6000000238418579', '1.2999999523162842'], name='petal_wid', dtype='float32'),
]
df = pd.concat(columns, axis=1)
print(scorer.score_batch(df))

##  Recommended workflow with datatable (fast and consistent with training):
import datatable as dt
dt.Frame(df).to_csv("test.csv")          # turn above dummy frame into a CSV (for convenience)
test_dt = dt.fread("test.csv", na_strings=['', '?', 'None', 'nan', 'NA', 'N/A', 'unknown', 'inf', '-inf', '1.7976931348623157e+308', '-1.7976931348623157e+308'])           # parse test set CSV file into datatable (with consistent NA handling)
preds_df = scorer.score_batch(test_dt)   # make predictions (pandas frame)
dt.Frame(preds_df).to_csv("preds.csv")   # save pandas frame to CSV using datatable


#
# The following lines demonstrate how to obtain per-feature prediction contributions per row. These can be
# very helpful in interpreting the model's predictions for individual observations (rows).
# Note that the contributions are in margin space (link space), so for binomial models the application of the
# final logistic function is omitted, while for multinomial models, the application of the final softmax function is
# omitted and for regression models the inverse link function is omitted (such as exp/square/re-normalization/etc.).
# This ensures that we can provide per-feature contributions that add up to the model's prediction.
# To simulate the omission of the transformation from margin/link space back to the probability or target space,
# and to get the predictions in the margin/link space, enable the output_margin flag. To get the prediction
# contributions, set pred_contribs=True. Note that you cannot provide both flags at the same time.
#

print('---------- Get Per-Feature Prediction Contributions for Row ----------')
print(scorer.score([
    '5.199999809265137',  # sepal_len
    '2.0',  # sepal_wid
    '1.2999999523162842',  # petal_len
    '1.5',  # petal_wid
], pred_contribs=True))


print('---------- Get Per-Feature Prediction Contributions for Frame ----------')
pred_contribs = scorer.score_batch(df, pred_contribs=True)  # per-feature prediction contributions
print(pred_contribs)


#
# The following lines demonstrate how to perform feature transformations without scoring.
# You can use this capability to transform input rows and fit models on the transformed frame
#   using an external ML tool of your choice, e.g. Sparkling Water or H2O.
#

#
# To transform a batch of rows (without scoring), use the Scorer.fit_transform_batch() method:
# This method fits the feature engineering pipeline on the given training frame, and applies it on the validation set,
# and optionally also on a test set.
#

# Transforms given datasets into enriched datasets with Driverless AI features')
#    train - for model fitting (do not use parts of this frame for parameter tuning)')
#    valid - for model parameter tuning')
#    test  - for final model testing (optional)')

print('---------- Transform Frames ----------')

# The target column 'species' has to be present in all provided frames.
df['species'] = pd.Series(['Iris-versicolor', 'Iris-versicolor', 'Iris-setosa', 'Iris-virginica', 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica', 'Iris-virginica', 'Iris-setosa', 'Iris-virginica'], dtype='object')

#  For demonstration only, do not use the same frame for train, valid and test!
train_munged, valid_munged, test_munged = \
  scorer.fit_transform_batch(train_frame=df, valid_frame=df, test_frame=df)
print(train_munged)  # for model fitting (use entire frame, no cross-validation)
print(valid_munged)  # for model validation (parameter tuning)
print(test_munged)   # for final pipeline testing (one time)

#
# To retrieve the original feature column names, use the Scorer.get_column_names() method:
# This method retrieves the input column names 
#

print('---------- Retrieve column names ----------')
print(scorer.get_column_names())

#
# To retrieve the transformed column names, use the Scorer.get_transformed_column_names() method:
# This method retrieves the transformed column names
#

print('---------- Retrieve transformed column names ----------')
print(scorer.get_transformed_column_names())

