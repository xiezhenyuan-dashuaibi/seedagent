from __future__ import annotations

import argparse
import importlib.util
import os
from typing import Any, Iterable

from dotenv import load_dotenv
from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep

try:
    import readline  # noqa: F401
except ModuleNotFoundError:
    pass


DEFAULT_AUTHORIZED_IMPORTS = [
    "abc",
    "array",
    "base64",
    "bisect",
    "calendar",
    "cmath",
    "copy",
    "csv",
    "dataclasses",
    "decimal",
    "difflib",
    "enum",
    "fractions",
    "functools",
    "hashlib",
    "heapq",
    "html",
    "json",
    "operator",
    "pathlib",
    "pprint",
    "string",
    "textwrap",
    "typing",
    "uuid",
    "zoneinfo",
    # Lightweight, commonly useful third-party packages when installed.
    "numpy.*",
    "pandas.*",
    "matplotlib.*",
    "PIL.*",
    "pydantic.*",
    "yaml.*",
    "bs4.*",
    "lxml.*",
]

try:
    from .context_logger import ModelInputLogger
    from .routed_agent import SeedAgent
    from .runtime_events import RuntimeEventRouter, TerminalRenderer
    from .streaming_model import SeedStreamingOpenAIModel
except ImportError:
    from context_logger import ModelInputLogger
    from routed_agent import SeedAgent
    from runtime_events import RuntimeEventRouter, TerminalRenderer
    from streaming_model import SeedStreamingOpenAIModel


def _first_env(names: Iterable[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _installed_imports(imports: Iterable[str]) -> list[str]:
    installed: list[str] = []
    for module_name in imports:
        base_name = module_name.split(".")[0]
        if importlib.util.find_spec(base_name) is not None:
            installed.append(module_name)
    return installed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SeedAgent terminal runner.")
    parser.add_argument("prompt", nargs="?", default=None, help="Optional one-shot prompt. Omit for terminal chat.")
    parser.add_argument("--model-id", default=_first_env(["OPENAI_MODEL", "MODEL_ID"]) or "gpt-4o-mini")
    parser.add_argument("--api-key", default=_first_env(["OPENAI_API_KEY", "MODEL_API_KEY"]))
    parser.add_argument(
        "--api-base",
        default=_first_env(["OPENAI_API_BASE", "OPENAI_BASE_URL", "OPENAI_API_BASE_URL", "MODEL_API_BASE"]),
        help="OpenAI-compatible API base URL, for example https://api.openai.com/v1.",
    )
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("TERMINAL_AGENT_MAX_STEPS", "8")))
    parser.add_argument("--imports", nargs="*", default=[], help="Extra Python imports authorized for generated code.")
    parser.add_argument("--show-code", action="store_true", help="Show generated Python code in the terminal.")
    parser.add_argument(
        "--log-model-inputs",
        action="store_true",
        default=True,
        help="Overwrite seedagent/logs/model_inputs.md with the latest exact model.generate() input. Enabled by default.",
    )
    parser.add_argument(
        "--no-log-model-inputs",
        action="store_false",
        dest="log_model_inputs",
        help="Disable exact model input logging.",
    )
    parser.add_argument("--once", action="store_true", help="Run one prompt then exit.")
    return parser.parse_args()


def _authorized_imports(args: argparse.Namespace) -> list[str]:
    imports = [*DEFAULT_AUTHORIZED_IMPORTS, *args.imports]
    seen = set()
    unique_imports = []
    for module_name in imports:
        if module_name not in seen:
            seen.add(module_name)
            unique_imports.append(module_name)
    return _installed_imports(unique_imports)


def _make_model_input_logger(args: argparse.Namespace) -> ModelInputLogger | None:
    if not args.log_model_inputs:
        return None
    model_input_logger = ModelInputLogger(run_name="seedagent")
    print(f"model-inputs> {model_input_logger.markdown_path}")
    return model_input_logger


def build_seed_agent(args: argparse.Namespace, router: RuntimeEventRouter) -> SeedAgent:
    model_input_logger = _make_model_input_logger(args)
    model = SeedStreamingOpenAIModel(
        model_id=args.model_id,
        api_base=args.api_base,
        api_key=args.api_key,
        router=router,
        model_input_logger=model_input_logger,
    )
    return SeedAgent(
        tools=[],
        model=model,
        router=router,
        max_steps=args.max_steps,
        additional_authorized_imports=_authorized_imports(args),
        code_block_tags=("<python_run>", "</python_run>"),
        stream_outputs=False,
        verbosity_level=0,
    )


build_terminal_agent = build_seed_agent


def _run_once(agent: SeedAgent, router: RuntimeEventRouter, prompt: str, *, reset: bool) -> Any:
    router.emit("run_started", task=prompt)
    final_output = None
    for event in agent.run(prompt, stream=True, reset=reset):
        if isinstance(event, (ActionStep, PlanningStep)):
            continue
        if isinstance(event, FinalAnswerStep):
            final_output = event.output
    return final_output


def _interactive_loop(agent: SeedAgent, router: RuntimeEventRouter) -> None:
    print("SeedAgent")
    print("Commands: /exit, /quit, /reset")
    print()
    first_turn = True
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            break
        if prompt == "/reset":
            agent.memory.reset()
            agent.monitor.reset()
            first_turn = True
            print("memory cleared")
            continue

        try:
            _run_once(agent, router, prompt, reset=first_turn)
            first_turn = False
        except Exception as exc:
            router.emit("error", message=str(exc))


def main() -> int:
    load_dotenv()
    args = _parse_args()
    if not args.api_key:
        print("Missing API key. Set OPENAI_API_KEY or pass --api-key.")
        return 2

    router = RuntimeEventRouter()
    router.subscribe(TerminalRenderer(show_code=args.show_code))
    agent = build_seed_agent(args, router)

    if args.prompt:
        _run_once(agent, router, args.prompt, reset=True)
        return 0
    if args.once:
        print("--once requires a prompt.")
        return 2

    _interactive_loop(agent, router)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
