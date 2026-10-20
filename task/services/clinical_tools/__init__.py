"""IVF clinical tools served over HTTP for tool-calling agents (Claude Code via MCP).

cleaning: outcome-blind deleakage and outlier rules for the training CSV.
similar:  similar-patient cohort by female age and AMH, with a minimum cohort size.
predict:  clinical pregnancy probability from age and AMH (logistic regression).
service:  HTTP endpoints and the tool spec files for the MCP bridge.
"""
