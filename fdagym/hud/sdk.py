"""HUD SDK Environment for public FDA-Gym. Lazy-imports ``hud`` / FastMCP.

Default ``import fdagym`` (and pyright on ``fdagym``) does not load this module's
SDK imports. Call ``load_hud_environment()`` after ``uv sync --extra hud``.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import AsyncGenerator
from typing import Any

from fdagym.env import TOOL_BUDGET, tool_hard_cap
from fdagym.hud.session import PublicEpisode
from fdagym.tools import TOOLS

ENV_NAME = "fda-gym"
ENV_VERSION = "0.1.0"
MCP_CAPABILITY_NAME = "fda-gym-tools"
MCP_PATH = "/mcp"
_HUD_MISSING = "FDA-Gym HUD adapter needs the [hud] extra: uv sync --extra hud"

_current: PublicEpisode | None = None
_mcp_task: asyncio.Task | None = None
_loaded: tuple[Any, Any] | None = None


def _tool_meta(name: str) -> dict:
    for spec in TOOLS:
        function = spec["function"]
        if function["name"] == name:
            return function
    raise KeyError(name)


def _strip(arguments: dict) -> dict:
    return {key: value for key, value in arguments.items() if value is not None}


def current_episode() -> PublicEpisode:
    if _current is None:
        raise RuntimeError("no public FDA-Gym episode is active")
    return _current


def bind_episode(episode: PublicEpisode) -> PublicEpisode | None:
    global _current
    previous = _current
    _current = episode
    return previous


def unbind_episode(previous: PublicEpisode | None) -> None:
    global _current
    _current = previous


def _dispatch(name: str, arguments: dict) -> dict:
    return current_episode().call_tool(name, _strip(arguments))


def build_mcp_server():
    """FastMCP server whose tools call ``PublicEpisode`` / ``FDAEnv.step``."""
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(_HUD_MISSING) from exc

    server = FastMCP(name=MCP_CAPABILITY_NAME)
    label = _tool_meta("search_drug_label")
    section = _tool_meta("get_label_section")
    events = _tool_meta("query_adverse_events")
    approvals = _tool_meta("search_approvals")
    submit = _tool_meta("submit_answer")

    @server.tool(name=label["name"], description=label["description"])
    def search_drug_label(
        generic_name: str,
        has_section: str | None = None,
        page: int | None = None,
    ) -> dict:
        return _dispatch("search_drug_label", {"generic_name": generic_name, "has_section": has_section, "page": page})

    @server.tool(name=section["name"], description=section["description"])
    def get_label_section(set_id: str, section: str) -> dict:
        return _dispatch("get_label_section", {"set_id": set_id, "section": section})

    @server.tool(name=events["name"], description=events["description"])
    def query_adverse_events(
        generic_name: str,
        mode: str,
        serious: bool | None = None,
        year: int | None = None,
    ) -> dict:
        return _dispatch(
            "query_adverse_events",
            {"generic_name": generic_name, "mode": mode, "serious": serious, "year": year},
        )

    @server.tool(name=approvals["name"], description=approvals["description"])
    def search_approvals(
        generic_name: str | None = None,
        application_number: str | None = None,
        sponsor_name: str | None = None,
        page: int | None = None,
    ) -> dict:
        return _dispatch(
            "search_approvals",
            {
                "generic_name": generic_name,
                "application_number": application_number,
                "sponsor_name": sponsor_name,
                "page": page,
            },
        )

    @server.tool(name=submit["name"], description=submit["description"])
    def submit_answer(answer: Any, citations: list[dict]) -> dict:
        return _dispatch("submit_answer", {"answer": answer, "citations": citations})

    return server


async def mcp_tool_names(server=None) -> list[str]:
    mcp = server if server is not None else build_mcp_server()
    tools = await mcp.list_tools()
    return [tool.name for tool in tools]


def _require_hud():
    try:
        from hud.capabilities import Capability
        from hud.environment import Environment
    except ImportError as exc:
        raise ImportError(_HUD_MISSING) from exc
    return Environment, Capability


async def _wait_tcp(host: str, port: int, timeout: float = 8.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        try:
            _reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(f"FDA-Gym MCP server did not bind on {host}:{port}") from None
            await asyncio.sleep(0.05)


def _ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def load_hud_environment():
    """Build the HUD ``Environment`` and the ``public`` task template factory."""
    global _loaded
    if _loaded is not None:
        return _loaded

    Environment, Capability = _require_hud()
    env = Environment(name=ENV_NAME, version=ENV_VERSION)

    @env.initialize
    async def _start_tools() -> None:
        global _mcp_task
        if _mcp_task is not None and not _mcp_task.done():
            return
        port = _ephemeral_port()
        server = build_mcp_server()
        _mcp_task = asyncio.create_task(
            server.run_http_async(
                host="127.0.0.1",
                port=port,
                path=MCP_PATH,
                show_banner=False,
                stateless_http=True,
            )
        )
        await _wait_tcp("127.0.0.1", port)
        env.add_capability(Capability.mcp(name=MCP_CAPABILITY_NAME, url=f"http://127.0.0.1:{port}{MCP_PATH}"))

    @env.shutdown
    async def _stop_tools() -> None:
        global _mcp_task
        task = _mcp_task
        _mcp_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    @env.template(
        id="public",
        description="One published FDA-Gym public.jsonl task (FG-9101–FG-9193).",
    )
    async def public_task(task_id: str) -> AsyncGenerator[str | float, str | None]:
        episode = PublicEpisode(task_id)
        previous = bind_episode(episode)
        try:
            answer = yield episode.prompt()
            reward = episode.grade(answer)
            yield float(reward["total"])
        finally:
            unbind_episode(previous)

    _loaded = (env, public_task)
    return _loaded


def mint_public_tasks(public_path=None):
    """Concrete HUD tasks, one per public.jsonl row."""
    from fdagym.hud.session import public_task_ids

    _env, public_task = load_hud_environment()
    tasks = []
    for task_id in public_task_ids(public_path=public_path):
        task = public_task(task_id=task_id)
        task.slug = task_id
        task.columns = {"split": "public"}
        task.agent_config = {"max_steps": tool_hard_cap(TOOL_BUDGET) + 1}
        tasks.append(task)
    return tasks
