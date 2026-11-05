"""FastAPI services for the IVF clinical tools (one process per tool).

similar/server.py  ivf_similar_patients on port 9201
predict/server.py  ivf_predict_outcome on port 9202
cases/server.py    ivf_query_cases on port 9203 (training-case database, structured queries)
case_sql/server.py ivf_case_sql on port 9204 (training-case database, read-only SQL)
_base.py           shared logging, bounded computation, audit, health and errors
specs.py           writes the MCP tool specs and the system prompt tool section
"""
