import asyncio
import os
from pathlib import Path

from e2b import AsyncSandbox

# 自建 E2B 需要 API_URL + SANDBOX_URL；从 e2b_eval/.env 读取（不覆盖已导出的变量）
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip().removeprefix("export ").strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

SANDBOX_ID = "iex36rarmysffjz28htjp"
# BlackboxServer 的运行时根目录；每次起 OpenClaw 都会新建一个 bbs-<timestamp> 子目录
BBS_ROOT = os.environ.get("BBS_RUNTIME_ROOT", "/workspace_sandbox/blackbox_server_runtime")


async def main() -> None:
    sb = await AsyncSandbox.connect(SANDBOX_ID, timeout=3600)

    # 动态定位本次运行真正的 OpenClaw workspace（bbs-* 名字每次都变）
    r = await sb.commands.run(
        f"ls -d {BBS_ROOT}/bbs-*/home/.openclaw/workspace/AGENTS.md 2>/dev/null | head -1; true"
    )
    agents_md = r.stdout.strip()
    if not agents_md:
        raise SystemExit(f"no OpenClaw AGENTS.md under {BBS_ROOT}/bbs-*/home/.openclaw/workspace")
    workspace = agents_md[: -len("/AGENTS.md")]
    oc_home = workspace[: -len("/workspace")]
    print("OpenClaw workspace:", workspace)
    print("OpenClaw HOME     :", oc_home)

    # 实际注入给模型的 AGENTS.md（= /workspace/AGENTS.md + 运行时追加的工具桥接段）
    print("\n===== .openclaw/workspace/AGENTS.md =====")
    print(await sb.files.read(agents_md))

    # 对照：仓库上传到 /workspace 的那份（少了运行时追加段）
    print("\n===== /workspace/AGENTS.md (repo copy) =====")
    print(await sb.files.read("/workspace/AGENTS.md"))

    # OpenClaw 家目录结构：config / session / logs / 空壳 SOUL 等
    for cmd in (
        f"ls -la {workspace}",
        f"ls -la {oc_home}",
        f"ls -la {oc_home}/agents/main/sessions 2>/dev/null",
    ):
        res = await sb.commands.run(f"{cmd}; true")
        print(f"\n$ {cmd}")
        print(res.stdout, end="")


asyncio.run(main())
