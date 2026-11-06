"""FastAPI service for the IVF clinical tools (all tools in one process).

server.py          one process, one port (9201): every tool at POST /tools/<tool name>
similar/tool.py    ivf_similar_patients
predict/tool.py    ivf_predict_outcome
cases/tool.py      ivf_query_cases (training-case database, structured queries)
case_sql/tool.py   ivf_case_sql (training-case database, read-only SQL)
_base.py           shared logging, bounded computation, audit, health and errors
specs.py           writes the MCP tool specs and the system prompt tool section
"""
