"""
Server entry point for uv run / openenv serve.

Starts the EpiSteward FastAPI application via uvicorn on port 7860.
"""

import uvicorn


def main() -> None:
    """Launch the EpiSteward OpenEnv server."""
    uvicorn.run("episteward.api.server:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
