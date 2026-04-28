from __future__ import annotations

# Backward-compatible thin entrypoint. Implementation lives under forecast/.
from forecast.all import *  # noqa: F401,F403
from forecast.runner import main


if __name__ == "__main__":
    main()
