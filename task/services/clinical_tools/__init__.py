"""IVF clinical tool logic (numpy only, no web framework).

cleaning:        outcome-blind deleakage and outlier rules for the training CSV.
similar:         similar-patient retrieval; strategies are pluggable (STRATEGIES).
predict:         pregnancy probability from age, AMH and Gn days (logistic regression).
train_predictor: writes the model artifact; Predictor only reads it.

The HTTP services that expose these as tools live in ivf_services/.
"""
