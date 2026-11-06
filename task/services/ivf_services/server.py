"""IVF tool service: every tool in one process on one port, each at its own path by tool name.

  POST /tools/ivf_similar_patients   similar training cases by female age and AMH
  POST /tools/ivf_predict_outcome    pregnancy probability, 95% interval and each input's effect
  POST /tools/ivf_query_cases        structured queries on the training-case database
  POST /tools/ivf_case_sql           one read-only SELECT on the training-case database
  GET  /health                       status, the tools, each tool's data (scope, cases, model), load
  GET  /ivf/audit                    tool calls since start;  POST /ivf/reset clears them

A tool is a module (similar/tool.py, ...) declaring TOOL, TOOL_DESCRIPTION, REQUEST_MODEL, STATE, and
load(), health(state), call(state, request), summary(response); tools with the same STATE share what
load() returns (the two case-database tools share one database). Adding a tool = a module in TOOLS.

Environment:
  IVF_TOOLS_PORT     9201
  data and models: see each tool module (IVF_TRAIN_CSV, IVF_MODEL / IVF_MODEL_NAME, IVF_CASE_TABLE,
                     IVF_EXCLUDE_CASE hides the evaluated case's fold from every tool that reads cases)
  logging, concurrency (shared by all tools): see ivf_services/_base.py
"""

import asyncio
import inspect
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ivf_services._base import ServiceKit, run
from ivf_services.case_sql import tool as case_sql_tool
from ivf_services.cases import tool as cases_tool
from ivf_services.predict import tool as predict_tool
from ivf_services.similar import tool as similar_tool

SERVICE = "ivf_tools"
PORT_ENV, DEFAULT_PORT = "IVF_TOOLS_PORT", 9201
TOOLS: tuple[ModuleType, ...] = (similar_tool, predict_tool, cases_tool, case_sql_tool)
PATH_PREFIX = "/tools/"


def tool_path(tool: ModuleType) -> str:
    return PATH_PREFIX + tool.TOOL


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tool_states = {}
    for tool in TOOLS:
        if tool.STATE in app.state.tool_states:
            continue
        try:
            app.state.tool_states[tool.STATE] = await asyncio.to_thread(tool.load)
        except Exception:
            kit.log.exception("failed to start: could not load the data of %s", tool.TOOL)
            raise
    info = {tool.TOOL: tool.health(app.state.tool_states[tool.STATE]) for tool in TOOLS}
    kit.health_info.update(tool_info=info)
    kit.log.info("started: %s", "; ".join(f"{name} {detail}" for name, detail in info.items()))
    yield
    kit.shutdown()


app = FastAPI(title="IVF Clinical Tools", lifespan=lifespan)
kit = ServiceKit(app, SERVICE, [tool.TOOL for tool in TOOLS])


def _endpoint(tool: ModuleType):
    path = tool_path(tool)

    async def endpoint(req) -> Any:
        state = app.state.tool_states[tool.STATE]
        try:
            resp = await kit.compute(tool.call, state, req)
        except ValueError as exc:
            if not getattr(tool, "CLIENT_ERROR", None):
                raise
            body = {"error": tool.CLIENT_ERROR, "message": str(exc)}
            kit.record(path, req.model_dump(), body, status=400)
            return JSONResponse(status_code=400, content=body)
        kit.record(path, req.model_dump(), resp, summary=f"{tool.TOOL} {tool.summary(resp)}")
        return resp

    # the request model is the body type, so FastAPI validates it (422) as for a hand-written route
    endpoint.__signature__ = inspect.Signature(
        [inspect.Parameter("req", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=tool.REQUEST_MODEL)])
    endpoint.__name__ = tool.TOOL
    return endpoint


for _tool in TOOLS:
    app.add_api_route(tool_path(_tool), _endpoint(_tool), methods=["POST"], name=_tool.TOOL,
                      summary=_tool.TOOL_DESCRIPTION[:80])


if __name__ == "__main__":
    run(app, PORT_ENV, DEFAULT_PORT)
