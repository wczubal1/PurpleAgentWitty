# Purple Agent (Short Interest)

This purple agent uses an LLM to decide how to call
`/home/wczubal1/projects/tau2/brokercheck/client_short.py` for short interest data and
returns JSON in an A2A artifact.

## Run

```bash
uv sync
uv run src/server.py --host 127.0.0.1 --port 9010
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

## Expected Request (from green agent)

```json
{
  "task": "max_short_interest",
  "client_short_path": "/home/wczubal1/projects/tau2/brokercheck/client_short.py",
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

## Response

The agent responds with a JSON artifact containing:

- `status` / `error`
- `requested_settlement_date`
- `results` (each symbol includes `attempts`, `chosen_date`,
  `currentShortPositionQuantity`, and `record`)
- `best_symbol` / `best_quantity` for `max_short_interest`
