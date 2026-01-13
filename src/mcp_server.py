from typing import Any
import json
import os
import subprocess
import sys

from mcp.server.fastmcp import FastMCP


DEFAULT_CLIENT_SHORT_PATH = "/home/wczubal1/projects/tau2/brokercheck/client_short.py"

mcp = FastMCP("FINRA Short Interest")


def _normalize_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "rows", "results", "result", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _extract_short_position(
    payload: Any,
    symbol: str,
    settlement_date: str,
) -> tuple[Any | None, dict[str, Any] | None]:
    target_symbol = symbol.upper()
    for record in _normalize_records(payload):
        record_symbol = record.get("symbolCode")
        if not isinstance(record_symbol, str) or record_symbol.upper() != target_symbol:
            continue
        record_date = record.get("settlementDate")
        if not isinstance(record_date, str) or not record_date.startswith(settlement_date):
            continue
        return record.get("currentShortPositionQuantity"), record
    return None, None


def _run_client_short(
    *,
    client_short_path: str,
    symbol: str,
    settlement_date: str,
    issue_name: str | None,
    finra_client_id: str | None,
    finra_client_secret: str | None,
    timeout: int | None,
) -> Any:
    cmd = [
        sys.executable,
        client_short_path,
        "--symbol",
        symbol,
        "--settlement-date",
        settlement_date,
    ]
    if issue_name:
        cmd.extend(["--issue-name", issue_name])

    env = os.environ.copy()
    if finra_client_id:
        env["FINRA_CLIENT_ID"] = finra_client_id
    if finra_client_secret:
        env["FINRA_CLIENT_SECRET"] = finra_client_secret

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout or 60,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "Unknown error"
        raise RuntimeError(f"client_short.py failed: {message}")

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from client_short.py: {exc}") from exc


@mcp.tool()
def finra_short_interest(
    symbol: str,
    settlement_date: str,
    issue_name: str | None = None,
    client_short_path: str | None = None,
    finra_client_id: str | None = None,
    finra_client_secret: str | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    path = (
        client_short_path
        or os.environ.get("CLIENT_SHORT_PATH")
        or DEFAULT_CLIENT_SHORT_PATH
    )
    client_id = finra_client_id or os.environ.get("FINRA_CLIENT_ID")
    client_secret = finra_client_secret or os.environ.get("FINRA_CLIENT_SECRET")

    payload = _run_client_short(
        client_short_path=path,
        symbol=symbol,
        settlement_date=settlement_date,
        issue_name=issue_name,
        finra_client_id=client_id,
        finra_client_secret=client_secret,
        timeout=timeout,
    )
    quantity, record = _extract_short_position(payload, symbol, settlement_date)
    return {
        "symbol": symbol,
        "settlement_date": settlement_date,
        "currentShortPositionQuantity": quantity,
        "record": record,
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
