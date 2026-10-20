"""IVF clinical tools served over HTTP for tool-calling agents (Claude Code via MCP).

cleaning: outcome-blind deleakage and outlier rules for the training CSV.
similar:  similar-patient cohort by female age and AMH, with a minimum cohort size.
predict:  pregnancy probability from age, AMH and Gn days (logistic regression);
          train_predictor writes the model artifact, Predictor only reads it.
similar_service / predict_service: two independent HTTP services (ports 9201 / 9202).
specs:    MCP tool spec files for the Claw task format.
"""
