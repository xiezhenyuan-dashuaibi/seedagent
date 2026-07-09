from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any

from smolagents import CodeAgent
from smolagents.agents import ActionOutput
from smolagents.memory import ActionStep, PlanningStep, TaskStep, ToolCall
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import LogLevel, Timing
from smolagents.utils import (
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    extract_code_from_text,
    parse_code_blobs,
    truncate_content,
)

try:
    from .runtime_events import RuntimeEventRouter
except ImportError:
    from runtime_events import RuntimeEventRouter


AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parent
SHELL_SANDBOX_DIR = AGENT_DIR / "sandbox_workspace"
SHELL_TIMEOUT_SECONDS = 60
AGENT_PYTHON = Path(sys.executable).resolve()
AGENT_PYTHON_SCRIPTS_DIR = AGENT_PYTHON.parent

SHELL_DENIED_PATTERNS = [
    r"\bRemove-Item\b",
    r"\brm\b",
    r"\bdel\b",
    r"\berase\b",
    r"\brmdir\b",
    r"\brd\b",
    r"\bMove-Item\b",
    r"\bmv\b",
    r"\bren\b",
    r"\brename\b",
    r"\bcopy\b",
    r"\bCopy-Item\b",
    r"\bSet-Content\b",
    r"\bAdd-Content\b",
    r"\bOut-File\b",
    r"\bNew-Item\b",
    r"\bmkdir\b",
    r"\bInvoke-WebRequest\b",
    r"\biwr\b",
    r"\bcurl\b",
    r"\bcurl\.exe\b",
    r"\bwget\b",
    r"\bInvoke-RestMethod\b",
    r"\birm\b",
    r"\bnpm\b",
    r"\bpnpm\b",
    r"\byarn\b",
    r"\bStart-Process\b",
    r"\bStart-Job\b",
    r"\bschtasks\b",
    r"\breg\b",
    r"\bnetsh\b",
    r"\bssh\b",
    r"\bscp\b",
    r"\bpython(?:\.exe)?\s+automation[\\/]+actions\b",
    r"\.\\\.venv\\Scripts\\python\.exe\s+automation[\\/]+actions\b",
    r">",
    r">>",
    r"\|",
    r"&",
    r";",
]

SHELL_ALLOWED_FIRST_TOKENS = {
    "dir",
    "echo",
    "get-childitem",
    "get-content",
    "ls",
    "pip",
    "pip.exe",
    "pwd",
    "python",
    "python.exe",
    ".\\.venv\\scripts\\python.exe",
    "rg",
    "select-string",
    "type",
    "where",
    "where.exe",
    "whoami",
}

PIP_COMMAND_PATTERN = re.compile(
    r"^(?:(?:python(?:\.exe)?|\.\\\.venv\\scripts\\python\.exe)\s+-m\s+pip|(?:pip(?:\.exe)?))\s+",
    re.IGNORECASE,
)

PROTOCOL_RETRY_LIMIT = 3


ROUTED_SYSTEM_PROMPT_TEMPLATE = """# 任务背景说明

## 基本信息
你是 SeedAgent，一个运行在标签路由终端运行时中的 AI Agent。你的回复会被宿主程序作为文本流监控；当你输出完整的 <commentary>、<python_run>、<shell_run> 或 <final> 标签块时，闭合标签会触发对应能力：展示进度、执行 Python、执行安全 shell 命令或结束当前轮次。执行结果会作为下一次模型输入中的 Observation 返回给你，因此你应主动使用这些标签完成观察、思考、行动和验证，辅助用户完成任务。

## 工具/能力协议##
标签即工具，打上指定的标签即代表使用指定的工具，你的能力包括以下工具：

<commentary>消息</commentary>
- 运行时效果：立刻把 `消息` 流式展示给用户，同时你继续工作。
- 只用于简短的进度更新。

<python_run>
Python 代码
</python_run>
- 运行时效果：在生成到 </python_run> 时停止模型输出，执行其中的 Python 代码，
  并把执行结果/日志作为下一次模型调用中的 Observation。
- 用于计算、解析、数据处理，以及只需要 Python 的检查，禁止进行如机器学习等超大计算量的任务。
- Python 状态会在多次 Python 运行之间保持。
- 允许导入：{authorized_imports}

<shell_run>命令</shell_run>
- 运行时效果：在生成到 </shell_run> 时停止模型输出，在沙盒 shell 进程中执行这一行命令，
  并把 stdout/stderr/退出码作为下一次模型调用中的 Observation。
- 用于安全的文档只读检查、简单命令行检查、检查当前 Python 环境，以及在当前 agent 虚拟环境中调整包。
- Shell 命令会在隔离的沙盒工作区运行，不会在项目根目录运行。
- Shell 命令必须是一行且非交互式。

<final>消息</final>
- 运行时效果：进行单轮的直接对话回复，或者把本轮agent所做的事项总结回复用户，并结束当前轮次，将发言权权交还给用户。
- 在任务完成、需要用户许可/输入、向用户提问，或所需能力被策略阻止时使用，总之就是需要用户输入信息时使用。

标签使用注意事项：
- 每次回复都需要善于使用标签（工具），将回复装进各个标签化的模块中（正式回复时第一个非空白字符必须是 `<`，否则会请求重新生成）。
- 写在 <commentary>、<python_run>、<shell_run> 或 <final> 之外的文本只是普通文本，不会执行任何操作，也不会向用户展示。
- 用普通文字写出的 shell 命令不会执行，例如 `date` 或 `python -m pip list`。
  必须把命令放进 <shell_run>...</shell_run>。
- 写在 <python_run>...</python_run> 之外的 Python 代码不会执行。
- <commentary> 只是可见进度更新。只有 commentary 不算完成一步。
- 只有在任务完成、证据足够、没有允许的动作能继续推进，或需要提问/停止时，才使用 <final>。

## 行为逻辑纪律

澄清不明确的请求：
- 如果用户目标、目标对象、必要输入、约束或成功标准不清楚，不要基于猜测执行。
- 在 <final>...</final> 中提出简洁的澄清问题，并把控制权交还给用户。
- 只询问选择下一步所必需的信息。优先提出一到三个具体问题。
- 如果任务有一部分已经清楚，并且可以在不改变状态的前提下安全检查有用上下文，可以先检查；否则先澄清。
- 用户回答后，继续对齐需求，直到任务清楚才开始行动。

保持积极：
- 回答前，判断用户请求是否能通过各种手段来推进，实在做不到才给用户提意见，不要在本身能帮用户实现任务的情况下一上来就给用户提意见，尽量积极主动自行帮助用户完成任务。
- 你的能力范围很广，因为你可以写并运行 Python 和安全 shell 命令。用这些能力来推导、验证、检查、
  转换和测试，而不是只依赖记忆或当前可见信息。
- 按“观察-思考-行动”的循环工作：先观察用户请求和已有 Observations，再思考下一步最有用的动作，
  然后执行一个动作并等待新的 Observation；根据新观察继续循环，直到可以可靠结束。

严禁事项：
- 不要尝试绕过沙盒限制。沙盒策略会阻止破坏性文件系统操作、凭据访问、网络探测、后台任务、GUI 启动器、注册表/系统更改、路径逃逸。

简单示例：

示例循环 1：用户询问时，该任务根据原本上下文的信息是无法回答的，但可以通过各种工具来获取外界信息，从而实现。
User: 现在几点？
Assistant:
<commentary>我会从 Python 运行时读取当前时间。</commentary>
<python_run>
from datetime import datetime
print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
</python_run>
Observation: 执行日志包含 `2026-07-08 17:30:00`。
Assistant:
<final>当前运行时的时间是 2026-07-08 17:30:00。</final>

示例循环 2：缺少库，安装后重试，该任务展现了多轮调用工具并进行观察-思考-行动的循环。
User: 给 hello 生成一个二维码。
Assistant:
<commentary>我会先用 Python 试一下。</commentary>
<python_run>
import qrcode
img = qrcode.make("hello")
print("created")
</python_run>
Observation: ModuleNotFoundError: No module named 'qrcode'
Assistant:
<commentary>所需的 Python 包缺失，我会把它安装到当前虚拟环境中。</commentary>
<shell_run>python -m pip install qrcode</shell_run>
Observation: 退出码 0。
Assistant:
<commentary>包已经安装好，我会重新运行 Python 任务。</commentary>
<python_run>
import qrcode
img = qrcode.make("hello")
print("QR code created for hello")
</python_run>
Observation: 执行日志包含 `QR code created for hello`。
Assistant:
<final>我已成功为 `hello` 创建二维码。</final>

示例循环 3：请求不清楚，先澄清再行动，该任务展现了作为Agent的主体性。
User: 帮我处理这个文件。
Assistant:
<final>请告诉我要处理哪个文件，以及你想要什么结果：总结、清洗、转换、分析，还是其他内容？</final>
User: 那个销售 CSV。汇总月收入并指出异常。
Assistant:
<commentary>我会分析 CSV 结构并计算月收入。</commentary>
<python_run>
# 在这里检查提供的 CSV 数据或文件路径，然后计算请求的汇总。
print("monthly revenue summary ready")
</python_run>
Observation: 执行日志包含计算出的汇总。
Assistant:
<final>我已汇总月收入并识别出异常。</final>

现在开始。
"""


def _extract_tag_text(text: str, opening_tag: str, closing_tag: str) -> str | None:
    pattern = re.escape(opening_tag) + r"(.*?)" + re.escape(closing_tag)
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    return "\n\n".join(match.strip() for match in matches).strip()


def _extract_tag_blocks(text: str, tag_name: str) -> list[str]:
    pattern = rf"<{re.escape(tag_name)}>(.*?)</{re.escape(tag_name)}>"
    return [match.strip() for match in re.findall(pattern, text, re.DOTALL) if match.strip()]


def _tag_is_present(text: str, opening_tag: str) -> bool:
    try:
        return re.search(opening_tag, text, re.DOTALL) is not None
    except re.error:
        return opening_tag in text


def _has_any_code_block(text: str, code_block_tags: tuple[str, str]) -> bool:
    return (
        _extract_tag_text(text, code_block_tags[0], code_block_tags[1]) is not None
        or "```python" in text
        or "```py" in text
    )


def _has_shell_run_block(text: str) -> bool:
    return _extract_tag_text(text, "<shell_run>", "</shell_run>") is not None


def _has_action_or_final_block(text: str, code_block_tags: tuple[str, str]) -> bool:
    return (
        _extract_tag_text(text, "<final>", "</final>") is not None
        or _has_shell_run_block(text)
        or _has_any_code_block(text, code_block_tags)
    )


def _protocol_retry_message(output_text: str, code_block_tags: tuple[str, str]) -> str | None:
    stripped_text = output_text.lstrip()
    if not stripped_text:
        return (
            "Your previous response was empty. Regenerate the response now. "
            "The first non-whitespace character must be `<`, and you must write exactly one valid routed block."
        )
    if not stripped_text.startswith("<"):
        return (
            "Your previous response did not start with a routing tag. Regenerate the response now. "
            "The first non-whitespace character must be `<`. Use exactly one valid block: "
            "<commentary>...</commentary> followed by a run block when you are still working, "
            "<python_run>...</python_run>, <shell_run>...</shell_run>, or <final>...</final>."
        )
    if not _has_action_or_final_block(stripped_text, code_block_tags):
        return (
            "Your previous response used only <commentary> or another non-action tag. That is not a complete step. "
            "Regenerate the response now and choose the real outcome yourself: "
            "<python_run>...</python_run>, <shell_run>...</shell_run>, or <final>...</final>. "
            "You may put one brief <commentary>...</commentary> before a run block, but commentary alone is invalid."
        )
    return None


INVALID_FINAL_PATTERNS = [
    r"不能(?:直接)?访问.*(?:标签|运行时|runtime)",
    r"cannot access.*(?:tag|runtime)",
    r"can't access.*(?:tag|runtime)",
    r"date` command unavailable",
    r"你也可以.*(?:终端|terminal|Python|python)",
    r"you can.*(?:terminal|python|shell)",
    r"```(?:bash|python|py|powershell|sh)",
]


def _invalid_final_reason(final_text: str) -> str | None:
    for pattern in INVALID_FINAL_PATTERNS:
        if re.search(pattern, final_text, flags=re.IGNORECASE | re.DOTALL):
            return f"Final answer violates runtime protocol: matched `{pattern}`."
    return None


def _first_shell_token(command: str) -> str:
    stripped = command.strip()
    if not stripped:
        return ""
    try:
        return shlex.split(stripped, posix=False)[0].strip("\"'").lower()
    except ValueError:
        return stripped.split()[0].strip("\"'").lower()


def _validate_shell_command(command: str) -> None:
    stripped = command.strip()
    if not stripped:
        raise ValueError("Shell command is empty.")
    if "\n" in stripped or "\r" in stripped:
        raise ValueError("Shell commands must be a single non-interactive line.")

    lowered = stripped.lower()
    first_token = _first_shell_token(stripped)
    if first_token not in SHELL_ALLOWED_FIRST_TOKENS:
        raise ValueError(
            "Shell command blocked by sandbox policy: "
            f"'{first_token}' is not in the allowed command list."
        )

    for pattern in SHELL_DENIED_PATTERNS:
        if re.search(pattern, stripped, flags=re.IGNORECASE):
            raise ValueError(f"Shell command blocked by sandbox policy: denied pattern `{pattern}`.")

    if "pip" in lowered:
        if not PIP_COMMAND_PATTERN.search(stripped):
            raise ValueError("Shell command blocked by sandbox policy: pip must run inside the agent Python env.")
        try:
            parts = [part.strip("\"'") for part in shlex.split(stripped, posix=False)]
        except ValueError as exc:
            raise ValueError(f"Shell command blocked by sandbox policy: invalid pip command: {exc}") from exc
        lowered_parts = [part.lower() for part in parts]
        pip_index = lowered_parts.index("pip") if "pip" in lowered_parts else lowered_parts.index("pip.exe")
        pip_action = lowered_parts[pip_index + 1] if len(lowered_parts) > pip_index + 1 else ""
        if pip_action not in {"install", "uninstall", "show", "list", "freeze", "--version", "-v"}:
            raise ValueError("Shell command blocked by sandbox policy: unsupported pip action.")
        if pip_action in {"install", "uninstall"} and "-y" not in lowered_parts and "--yes" not in lowered_parts:
            if pip_action == "uninstall":
                raise ValueError("Shell command blocked by sandbox policy: pip uninstall must include -y.")
        if re.search(r"--(?:user|prefix|root|target|require-virtualenv)", stripped, flags=re.IGNORECASE):
            raise ValueError("Shell command blocked by sandbox policy: unsafe pip option.")

    repo_root = str(REPO_ROOT).lower()
    agent_dir = str(AGENT_DIR).lower()
    normalized = lowered.replace("/", "\\")
    if repo_root in normalized or agent_dir in normalized or ".." in normalized:
        raise ValueError("Shell command blocked by sandbox policy: path escapes are not allowed.")


def _run_shell_command(command: str, timeout_seconds: int = SHELL_TIMEOUT_SECONDS) -> tuple[int | None, str, str]:
    _validate_shell_command(command)
    SHELL_SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    shell_executable = os.environ.get("COMSPEC") if os.name == "nt" else os.environ.get("SHELL")
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(AGENT_PYTHON.parent.parent)
    env["PATH"] = str(AGENT_PYTHON_SCRIPTS_DIR) + os.pathsep + env.get("PATH", "")
    completed = subprocess.run(
        command,
        shell=True,
        executable=shell_executable,
        cwd=SHELL_SANDBOX_DIR,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    return completed.returncode, completed.stdout, completed.stderr


class SeedAgent(CodeAgent):
    """SeedAgent runtime built on CodeAgent with tag-routed execution events."""

    def __init__(self, *args: Any, router: RuntimeEventRouter, **kwargs: Any) -> None:
        self.runtime_router = router
        super().__init__(*args, **kwargs)
        self.tools.pop("final" + "_answer", None)

    def initialize_system_prompt(self) -> str:
        authorized_imports = (
            "任意已安装的包"
            if "*" in self.authorized_imports
            else ", ".join(self.authorized_imports)
        )
        return ROUTED_SYSTEM_PROMPT_TEMPLATE.format(
            authorized_imports=authorized_imports,
        )

    def write_memory_to_messages(
        self,
        summary_mode: bool = False,
    ) -> list[ChatMessage]:
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in self.memory.steps:
            if isinstance(memory_step, TaskStep):
                content = [{"type": "text", "text": str(memory_step.task)}]
                if memory_step.task_images:
                    content.extend([{"type": "image", "image": image} for image in memory_step.task_images])
                messages.append(ChatMessage(role=MessageRole.USER, content=content))
                continue

            if isinstance(memory_step, PlanningStep):
                messages.extend(memory_step.to_messages(summary_mode=summary_mode))
                continue

            if not isinstance(memory_step, ActionStep):
                messages.extend(memory_step.to_messages(summary_mode=summary_mode))
                continue

            if memory_step.model_output is not None and not summary_mode:
                messages.append(
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=[{"type": "text", "text": str(memory_step.model_output).strip()}],
                    )
                )

            if memory_step.observations_images:
                messages.append(
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[{"type": "image", "image": image} for image in memory_step.observations_images],
                    )
                )

            if memory_step.observations is not None:
                messages.append(
                    ChatMessage(
                        role=MessageRole.TOOL_RESPONSE,
                        content=[{"type": "text", "text": f"Observation:\n{memory_step.observations}"}],
                    )
                )

            if memory_step.error is not None:
                messages.append(
                    ChatMessage(
                        role=MessageRole.TOOL_RESPONSE,
                        content=[
                            {
                                "type": "text",
                                "text": (
                                    "Error:\n"
                                    + str(memory_step.error)
                                    + "\nNow retry with the runtime protocol and avoid repeating the same mistake.\n"
                                ),
                            }
                        ],
                    )
                )
        return messages

    def _handle_max_steps_reached(self, task: str) -> Any:
        step_start_time = time.time()
        final_answer = (
            "Reached the step limit before the model produced a valid routed final answer. "
            "No fallback model call was made, so seedagent/logs/model_inputs.md still shows the last real routed model input."
        )
        final_memory_step = ActionStep(
            step_number=self.step_number,
            error=AgentMaxStepsError("Reached max steps without a valid routed final answer.", self.logger),
            timing=Timing(start_time=step_start_time, end_time=time.time()),
        )
        final_memory_step.observations = (
            "Max steps reached. The agent did not produce <final>...</final> under the routed protocol. "
            "Fallback final-answer generation is disabled for this runtime."
        )
        final_memory_step.action_output = final_answer
        self._finalize_step(final_memory_step)
        self.memory.steps.append(final_memory_step)
        self.runtime_router.emit("error", message=final_answer)
        return final_answer

    def _step_stream(self, memory_step: ActionStep):
        memory_messages = self.write_memory_to_messages()

        stop_sequences = ["Observation:", "Calling tools:"]
        if self.code_block_tags[1] not in self.code_block_tags[0]:
            stop_sequences.append(self.code_block_tags[1])
        stop_sequences.append("</shell_run>")

        try:
            retry_messages: list[ChatMessage] = []
            output_text = ""
            chat_message: ChatMessage | None = None
            for retry_index in range(PROTOCOL_RETRY_LIMIT + 1):
                input_messages = memory_messages + retry_messages
                memory_step.model_input_messages = input_messages
                additional_args: dict[str, Any] = {}
                if self._use_structured_outputs_internally:
                    additional_args["response_format"] = self._codeagent_response_format()
                chat_message = self.model.generate(
                    input_messages,
                    stop_sequences=stop_sequences,
                    **additional_args,
                )
                output_text = (chat_message.content or "").strip()

                if not self._use_structured_outputs_internally:
                    if (
                        output_text
                        and _tag_is_present(output_text, self.code_block_tags[0])
                        and not output_text.strip().endswith(self.code_block_tags[1])
                    ):
                        output_text += self.code_block_tags[1]
                        chat_message.content = output_text

                retry_message = _protocol_retry_message(output_text, self.code_block_tags)
                if retry_message is None:
                    break
                if retry_index >= PROTOCOL_RETRY_LIMIT:
                    memory_step.model_output_message = chat_message
                    memory_step.token_usage = chat_message.token_usage
                    memory_step.model_output = output_text
                    memory_step.observations = (
                        "Protocol failure: the model did not produce a valid routed response after "
                        f"{PROTOCOL_RETRY_LIMIT + 1} attempts. Last retry instruction: {retry_message}"
                    )
                    memory_step.action_output = memory_step.observations
                    yield ActionOutput(output=memory_step.observations, is_final_answer=False)
                    return
                retry_messages.append(
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[{"type": "text", "text": retry_message}],
                    )
                )

            if chat_message is None:
                raise AgentGenerationError("Error in generating model output:\nNo model output was produced.", self.logger)
            memory_step.model_output_message = chat_message
            memory_step.token_usage = chat_message.token_usage
            memory_step.model_output = output_text
        except Exception as exc:
            raise AgentGenerationError(f"Error in generating model output:\n{exc}", self.logger) from exc

        if not _has_action_or_final_block(output_text, self.code_block_tags):
            memory_step.observations = "Protocol failure: model output passed retry loop without an action block or <final>."
            memory_step.action_output = memory_step.observations
            yield ActionOutput(output=memory_step.observations, is_final_answer=False)
            return

        final_text = _extract_tag_text(output_text, "<final>", "</final>")
        if final_text is not None:
            invalid_final_reason = _invalid_final_reason(final_text)
            if invalid_final_reason is not None:
                memory_step.observations = (
                    invalid_final_reason
                    + "\nThat final answer was based on a wrong capability assumption. "
                    + "You can execute Python with <python_run>...</python_run> and safe shell commands with <shell_run>...</shell_run>. "
                    + "The host runtime scans your output; the closing tag triggers the matching capability and writes its result back as an Observation. "
                    + "Do not tell the user to run commands manually. In the next response, use a runtime tag if it can make progress."
                )
                memory_step.action_output = memory_step.observations
                yield ActionOutput(output=memory_step.observations, is_final_answer=False)
                return
            memory_step.code_action = None
            memory_step.action_output = final_text
            self.runtime_router.emit("final_response", text=final_text)
            yield ActionOutput(output=final_text, is_final_answer=True)
            return

        shell_command = _extract_tag_text(output_text, "<shell_run>", "</shell_run>")
        if shell_command is not None:
            memory_step.code_action = None
            self.runtime_router.emit("shell_started", command=shell_command)
            tool_call = ToolCall(
                name="shell",
                arguments=shell_command,
                id=f"call_{len(self.memory.steps)}",
            )
            yield tool_call
            memory_step.tool_calls = [tool_call]

            try:
                return_code, stdout, stderr = _run_shell_command(shell_command)
            except ValueError as exc:
                return_code = None
                stdout = ""
                stderr = f"Blocked by shell sandbox policy: {exc}"
            except subprocess.TimeoutExpired as exc:
                return_code = None
                stdout = exc.stdout or ""
                stderr = (exc.stderr or "") + "\nShell command timed out."
            except Exception as exc:
                self.runtime_router.emit("error", message=str(exc))
                raise AgentExecutionError(str(exc), self.logger) from exc

            stdout = truncate_content(str(stdout))
            stderr = truncate_content(str(stderr))
            observation = (
                "Shell command:\n"
                + shell_command
                + "\nExit code:\n"
                + str(return_code)
                + "\nStdout:\n"
                + stdout
                + "\nStderr:\n"
                + stderr
            )
            memory_step.observations = observation
            memory_step.action_output = stdout
            self.runtime_router.emit(
                "shell_finished",
                command=shell_command,
                return_code=return_code,
                stdout=stdout,
                stderr=stderr,
            )
            yield ActionOutput(output=stdout, is_final_answer=False)
            return

        try:
            if self._use_structured_outputs_internally:
                code_action = json.loads(output_text)["code"]
                code_action = extract_code_from_text(code_action, self.code_block_tags) or code_action
            else:
                try:
                    code_action = parse_code_blobs(output_text, self.code_block_tags)
                except Exception:
                    if _has_any_code_block(output_text, self.code_block_tags) or _has_shell_run_block(output_text):
                        raise
                    memory_step.observations = (
                        "The response did not contain a valid runtime block or explicit <final>. "
                        "Continue with <python_run>, <shell_run>, or an explicit <final>."
                    )
                    memory_step.action_output = memory_step.observations
                    yield ActionOutput(output=memory_step.observations, is_final_answer=False)
                    return
            if re.search(r"(?<![\w.])final[_]answer\s*\(", code_action):
                raise ValueError("Use <final>...</final> to finish this turn, outside Python code.")
            memory_step.code_action = code_action
        except Exception as exc:
            error_msg = f"Error in code parsing:\n{exc}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger)

        self.runtime_router.emit("code_started", code=code_action)
        tool_call = ToolCall(
            name="python_interpreter",
            arguments=code_action,
            id=f"call_{len(self.memory.steps)}",
        )
        yield tool_call
        memory_step.tool_calls = [tool_call]

        self.runtime_router.emit("execution_started", code=code_action)
        self.logger.log_code(title="Executing parsed code:", content=code_action, level=LogLevel.DEBUG)
        try:
            code_output = self.python_executor(code_action)
            observation = "Execution logs:\n" + code_output.logs
        except Exception as exc:
            if hasattr(self.python_executor, "state") and "_print_outputs" in self.python_executor.state:
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if execution_logs:
                    memory_step.observations = "Execution logs:\n" + execution_logs
            self.runtime_router.emit("error", message=str(exc))
            raise AgentExecutionError(str(exc), self.logger)

        truncated_output = truncate_content(str(code_output.output))
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation
        memory_step.action_output = code_output.output

        self.runtime_router.emit(
            "execution_finished",
            output=code_output.output,
            logs=code_output.logs,
            is_final_answer=code_output.is_final_answer,
        )
        yield ActionOutput(output=code_output.output, is_final_answer=code_output.is_final_answer)

    @staticmethod
    def _codeagent_response_format() -> dict[str, Any]:
        from smolagents.models import CODEAGENT_RESPONSE_FORMAT

        return CODEAGENT_RESPONSE_FORMAT


RoutedCodeAgent = SeedAgent


__all__ = ["SeedAgent", "RoutedCodeAgent"]
