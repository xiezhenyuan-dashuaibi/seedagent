from __future__ import annotations

try:
    from .terminal_agent import main
except ImportError:
    from terminal_agent import main


if __name__ == "__main__":
    raise SystemExit(main())
