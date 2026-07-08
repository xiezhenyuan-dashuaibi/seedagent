# SeedAgent

SeedAgent is a tag-routed terminal agent seed for second-stage development.

It wraps a `smolagents.CodeAgent` with a small runtime protocol:

- `<commentary>...</commentary>` streams progress to the terminal.
- `<python_run>...</python_run>` executes Python and returns an Observation.
- `<shell_run>...</shell_run>` executes a tightly sandboxed one-line shell command and returns an Observation.
- `<final>...</final>` streams the final user-facing answer and ends the turn.

The goal is to provide a compact seed you can fork into business-specific agents without carrying a full application domain.

## Install

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create `.env`:

```env
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4o-mini
# OPENAI_API_BASE=https://api.openai.com/v1
```

## Run

Interactive chat:

```powershell
.\.venv\Scripts\python.exe -m seedagent.terminal_agent
```

One-shot prompt:

```powershell
.\.venv\Scripts\python.exe -m seedagent.terminal_agent "现在几点？"
```

Useful flags:

- `--show-code`: print generated Python code.
- `--no-log-model-inputs`: disable exact model input logging.
- `/reset`: clear current agent memory in terminal chat.
- `/exit` or `/quit`: exit terminal chat.

## Logs

By default, SeedAgent overwrites:

```text
seedagent/logs/model_inputs.md
```

with the latest exact messages passed into `model.generate()`.

## Safety Boundary

`<shell_run>` is intentionally limited:

- Commands run from `seedagent/sandbox_workspace`.
- Commands must be one line and non-interactive.
- Read-only inspection, simple environment checks, and package operations in the current virtualenv are allowed.
- Destructive filesystem operations, network probing, background jobs, GUI launchers, registry/system changes, path escapes, and automation scripts are blocked by default.

## Notes

SeedAgent is the seed/runtime layer. The underlying code execution agent still depends on the upstream `smolagents` package; this repository does not rename or vendor that dependency.
