# Purple Agent (Short Interest)

This purple agent uses an LLM to decide how to call a `client_short.py` script
(path provided at runtime, e.g. `/opt/client_short.py`) for short interest data and
returns JSON in an A2A artifact.

## Run

```bash
uv sync
uv run src/server.py --host 127.0.0.1 --port 9010
```

## Make Targets

```bash
make run
make docker-build
make docker-run
```

## LLM Configuration

Set the OpenAI API key (required):

```bash
export OPENAI_API_KEY="..."
```

Optional model override (defaults to `gpt-4o-mini`):

```bash
export OPENAI_MODEL="gpt-4o-mini"
```

## MCP Tool Server (Optional)

The purple agent can call an MCP tool instead of running `client_short.py` directly.
Set `MCP_SERVER_COMMAND` to enable it (the agent will spawn the server per call):

```bash
export MCP_SERVER_COMMAND="python src/mcp_server.py"
```

If `MCP_SERVER_COMMAND` is not set, the agent will call `client_short.py` directly.

This MCP server exposes the tool:

- `finra_short_interest(symbol, settlement_date, issue_name?, client_short_path?, finra_client_id?, finra_client_secret?, timeout?)`

## FINRA Credentials

FINRA credentials must be supplied at runtime (they are not bundled with the agent):

```bash
export FINRA_CLIENT_ID="..."
export FINRA_CLIENT_SECRET="..."
```

Create your FINRA API credentials here:
https://developer.finra.org/docs#getting_started-the_api_console

## Docker

Build and run:

```bash
docker build -t purple-agent .
docker run -p 9010:9010 -e OPENAI_API_KEY="..." purple-agent --host 0.0.0.0 --port 9010
```

## GitHub Secrets

If you publish this repo with GitHub Actions, add:

- `OPENAI_API_KEY` (required for runtime tool use)
- `OPENAI_MODEL` (optional override)
- `FINRA_CLIENT_ID` (required to call FINRA)
- `FINRA_CLIENT_SECRET` (required to call FINRA)

## Expected Request (from green agent)

```json
{
  "task": "max_short_interest",
  "client_short_path": "/opt/client_short.py",
  "requested_settlement_date": "2025-05-15",
  "min_attempts": 3,
  "args": {
    "symbols": ["TSLA", "NVDA", "AAPL"],
    "settlement_date": "2025-05-15"
  },
  "finra_client_id": "...",
  "finra_client_secret": "...",
  "timeout": 60
}
```

Note: `client_short_path` must exist in the runtime. If you use Docker, mount the
script into the container (e.g. `-v /path/to/client_short.py:/opt/client_short.py`)
or bake it into a custom image.

## Response

The agent responds with a JSON artifact containing:

- `status` / `error`
- `requested_settlement_date`
- `results` (each symbol includes `attempts`, `chosen_date`,
  `currentShortPositionQuantity`, and `record`)
- `best_symbol` / `best_quantity` for `max_short_interest`
