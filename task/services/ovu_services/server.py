"""Oocyte-yield tool service: every tool in one process on one port, each at /tools/<tool name>.

  POST /tools/ovu_predict_oocytes   group probabilities, expected count, each input's effect
  POST /tools/ovu_similar_cases     the most similar training cycles and their oocyte counts
  POST /tools/ovu_case_sql          one read-only SELECT on the training-case database
  POST /tools/ovu_data_overview     what the database holds
  POST /tools/ovu_sql_examples      example queries
  GET  /health                      status, the tools, each tool's data, load
  GET  /ivf/audit                   tool calls since start;  POST /ivf/reset clears them

  OVU_DATA_DIR=outputs/ovu PYTHONPATH=task/services python task/services/ovu_services/server.py

Environment: OVU_TOOLS_PORT (9211); OVU_DATA_DIR and OVU_HIDE_FOLD (ovu_services/tools.py);
logging and concurrency as the IVF service (ivf_services/_base.py).
"""

import asyncio
import inspect
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from ivf_services._base import ServiceKit, run
from ovu_services.tools import build_tools

SERVICE = "ovu_tools"
PORT_ENV, DEFAULT_PORT = "OVU_TOOLS_PORT", 9211
PATH_PREFIX = "/tools/"
TOOLS = build_tools()                    # reads the model file and the specs for the request schemas


def tool_path(tool: Any) -> str:
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


app = FastAPI(title="OVU Clinical Tools", lifespan=lifespan)
kit = ServiceKit(app, SERVICE, [tool.TOOL for tool in TOOLS])


def _endpoint(tool: Any):
    path = tool_path(tool)

    async def endpoint(req) -> Any:
        state = app.state.tool_states[tool.STATE]
        try:
            resp = await kit.compute(tool.call, state, req)
        except ValueError as exc:
            body = {"error": tool.CLIENT_ERROR, "message": str(exc)}
            kit.record(path, req.model_dump(exclude_none=True), body, status=400)
            return JSONResponse(status_code=400, content=body)
        kit.record(path, req.model_dump(exclude_none=True), resp, summary=f"{tool.TOOL} {tool.summary(resp)}")
        return resp

    endpoint.__signature__ = inspect.Signature(
        [inspect.Parameter("req", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=tool.REQUEST_MODEL)])
    endpoint.__name__ = tool.TOOL
    return endpoint


for _tool in TOOLS:
    app.add_api_route(tool_path(_tool), _endpoint(_tool), methods=["POST"], name=_tool.TOOL,
                      summary=_tool.TOOL_DESCRIPTION[:80])


if __name__ == "__main__":
    run(app, PORT_ENV, DEFAULT_PORT)
