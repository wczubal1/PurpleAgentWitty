from typing import Any
from datetime import date, datetime, timedelta
import asyncio
import json
import os
import re
import shlex
import subprocess
import sys
import threading

from openai import OpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pydantic import BaseModel, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, DataPart
from a2a.utils import get_message_text


DEFAULT_CLIENT_SHORT_PATH = "/home/wczubal1/projects/tau2/brokercheck/client_short.py"
MIN_ATTEMPTS = 3
MAX_TOOL_ROUNDS = 20


class PurpleRequest(BaseModel):
    task: str | None = None
    client_short_path: str | None = None
    dataset_group: str | None = None
    dataset_name: str | None = None
    question: str | None = None
    args: dict[str, Any] | None = None
    finra_client_id: str | None = None
    finra_client_secret: str | None = None
    timeout: int | None = None
    min_attempts: int | None = None
    requested_settlement_date: str | None = None


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

def _extract_weekly_share(
    payload: Any,
    symbol: str,
    settlement_date: str,
) -> tuple[Any | None, dict[str, Any] | None]:
    target_symbol = symbol.upper()
    for record in _normalize_records(payload):
        record_symbol = (
            record.get("issueSymbolIdentifier")
            or record.get("symbolCode")
            or record.get("symbol")
        )
        if not isinstance(record_symbol, str) or record_symbol.upper() != target_symbol:
            continue
        record_date = record.get("weekStartDate") or record.get("summaryStartDate")
        if not isinstance(record_date, str) or not record_date.startswith(settlement_date):
            continue
        return record.get("totalWeeklyShareQuantity"), record
    return None, None


def _extract_treasury_volume(
    payload: Any,
    trade_date: str,
    years_to_maturity: str,
    benchmark: str,
) -> tuple[Any | None, dict[str, Any] | None]:
    for record in _normalize_records(payload):
        record_date = record.get("tradeDate")
        if not isinstance(record_date, str) or not record_date.startswith(trade_date):
            continue
        years = str(record.get("yearsToMaturity") or "").strip()
        record_benchmark = str(record.get("benchmark") or "").strip()
        if years == years_to_maturity and record_benchmark.lower() == benchmark.lower():
            return record.get("dealerCustomerVolume"), record
    return None, None


QUESTION_WEEKLY_KEYWORDS = ("weekly", "week", "weeklysummary", "weekly summary")
QUESTION_SHARE_KEYWORDS = ("share", "shares", "totalweeklysharequantity", "total weekly share")
QUESTION_SHORT_KEYWORDS = ("short interest", "short position", "current short")
QUESTION_TREASURY_KEYWORDS = ("treasury", "dealer customer volume", "on-the-run")
TREASURY_UPPER_BOUND_BUCKETS = {
    2: "<= 2 years",
    3: "> 2 years and <= 3 years",
    5: "> 3 years and <= 5 years",
    7: "> 5 years and <= 7 years",
    10: "> 7 years and <= 10 years",
}


def _infer_dataset_from_question(question: str | None) -> tuple[str | None, str | None]:
    if not question:
        return None, None
    lowered = question.lower()
    if any(key in lowered for key in QUESTION_TREASURY_KEYWORDS):
        return "fixedIncomeMarket", "treasuryDailyAggregates"
    if any(key in lowered for key in QUESTION_WEEKLY_KEYWORDS) and any(
        key in lowered for key in QUESTION_SHARE_KEYWORDS
    ):
        return "otcmarket", "weeklySummary"
    if any(key in lowered for key in QUESTION_SHORT_KEYWORDS):
        return "otcmarket", "consolidatedShortInterest"
    return None, None


def _is_treasury_max_question(question: str | None) -> bool:
    if not question:
        return False
    lowered = question.lower()
    return ("highest" in lowered or "max" in lowered) and "dealer customer volume" in lowered


def _has_treasury_bucket(question: str | None) -> bool:
    if not question:
        return False
    lowered = question.lower()
    if re.search(r">\\s*\\d+\\s*years\\s*and\\s*<=\\s*\\d+\\s*years", lowered):
        return True
    if re.search(r"(?:<=|up to)\\s*\\d+\\s*years", lowered):
        return True
    return False


def _parse_treasury_bucket(question: str | None) -> tuple[str, str]:
    if not question:
        return "<= 2 years", "On-the-run"
    lowered = question.lower()
    benchmark = "On-the-run"
    if "off-the-run" in lowered or "off the run" in lowered:
        benchmark = "Off-the-run"
    elif "on-the-run" in lowered or "on the run" in lowered:
        benchmark = "On-the-run"

    explicit = re.search(r">\\s*\\d+\\s*years\\s*and\\s*<=\\s*\\d+\\s*years", lowered)
    if explicit:
        return explicit.group(0).replace("  ", " "), benchmark
    bound_match = re.search(r"(?:<=|up to)\\s*(\\d+)\\s*years", lowered)
    if bound_match:
        bound = int(bound_match.group(1))
        bucket = TREASURY_UPPER_BOUND_BUCKETS.get(bound)
        if bucket:
            return bucket, benchmark
        return f"<= {bound} years", benchmark
    return "<= 2 years", benchmark


def _shift_year(value: str, years: int) -> str | None:
    parsed = _parse_date(value)
    if not parsed:
        return None
    try:
        return parsed.replace(year=parsed.year + years).strftime("%Y-%m-%d")
    except ValueError:
        adjusted = parsed - timedelta(days=1)
        return adjusted.replace(year=adjusted.year + years).strftime("%Y-%m-%d")


def _parse_date(value: str) -> date | None:
    trimmed = value.strip()
    if not trimmed:
        return None
    try:
        return datetime.strptime(trimmed, "%Y-%m-%d").date()
    except ValueError:
        return None


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return False


def _normalize_symbols(value: Any) -> list[str] | None:
    if not value:
        return None
    if isinstance(value, str):
        symbols = [part.strip().upper() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        symbols = [str(item).strip().upper() for item in value if str(item).strip()]
    else:
        return None
    return symbols or None


def _unwrap_response(payload: Any, task: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    nested = payload.get(task)
    if isinstance(nested, dict):
        return nested
    return payload

def _attach_dataset_info(
    payload: dict[str, Any],
    task: str,
    dataset_group: str | None,
    dataset_name: str | None,
) -> dict[str, Any]:
    nested = payload.get(task)
    if dataset_group:
        if "dataset_group" not in payload and "datasetGroup" not in payload:
            payload["dataset_group"] = dataset_group
        if isinstance(nested, dict) and "dataset_group" not in nested and "datasetGroup" not in nested:
            nested["dataset_group"] = dataset_group
    if dataset_name:
        if "dataset_name" not in payload and "datasetName" not in payload:
            payload["dataset_name"] = dataset_name
        if isinstance(nested, dict) and "dataset_name" not in nested and "datasetName" not in nested:
            nested["dataset_name"] = dataset_name
    return payload



def _missing_symbols(
    payload: dict[str, Any],
    symbols: list[str],
    min_attempts: int,
) -> list[str]:
    results = payload.get("results")
    if not isinstance(results, list):
        return [symbol.upper() for symbol in symbols]

    seen: set[str] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or item.get("symbolCode") or "").strip().upper()
        if not symbol:
            continue
        attempts = item.get("attempts")
        attempts_len = len(attempts) if isinstance(attempts, list) else 0
        if attempts_len >= min_attempts:
            seen.add(symbol)

    return [symbol.upper() for symbol in symbols if symbol.upper() not in seen]


def _build_system_prompt(min_attempts: int) -> str:
    return (
        "You are a purple agent. Decide which FINRA dataset to use based on the question. "
        "For short interest/short position questions, use consolidatedShortInterest. "
        "For weekly total shares questions, use weeklySummary and totalWeeklyShareQuantity. "
        "For Treasury dealer customer volume questions, use fixedIncomeMarket/treasuryDailyAggregates "
        "and return dealerCustomerVolume for the requested maturity bucket and benchmark "
        "(On-the-run/Off-the-run). If the question asks for the highest dealer customer volume, "
        "compare matching rows and return the maturity bucket with the max value. "
        "If a trade date falls on a weekend/holiday, use the closest available tradeDate and "
        "include attempts in the response. "
        "Use dataset_group/dataset_name from the request when provided. "
        "Data is reported twice per month (15th and month-end). "
        f"For each symbol, try at least {min_attempts} different settlement dates near the "
        "requested date to find the closest available date with data. "
        "Include dataset_name in your JSON response. Return JSON only (no markdown)."
    )


def _build_user_prompt(
    task: str,
    symbol: str | None,
    symbols: list[str] | None,
    requested_date: str,
    min_attempts: int,
    question: str | None,
    dataset_group: str | None,
    dataset_name: str | None,
) -> str:
    payload: dict[str, Any] = {
        "task": task,
        "requested_settlement_date": requested_date,
        "min_attempts": min_attempts,
    }
    if task == "treasury_daily_aggregate":
        payload["trade_date"] = requested_date
        expected_years, expected_benchmark = _parse_treasury_bucket(question)
        payload["expected_years_to_maturity"] = expected_years
        payload["expected_benchmark"] = expected_benchmark
        if _is_treasury_delta_question(question):
            payload["previous_trade_date"] = _shift_year(requested_date, -1) or ""
    if question:
        payload["question"] = question
    if dataset_group:
        payload["dataset_group"] = dataset_group
    if dataset_name:
        payload["dataset_name"] = dataset_name
    if symbol:
        payload["symbol"] = symbol
    if symbols:
        payload["symbols"] = symbols
    is_treasury_delta = (
        _is_treasury_delta_question(question) if task == "treasury_daily_aggregate" else False
    )
    is_treasury_max = (
        _is_treasury_max_question(question) if task == "treasury_daily_aggregate" else False
    )
    payload["response_format"] = {
        "max_short_interest": {
            "status": "ok|error",
            "requested_settlement_date": "YYYY-MM-DD",
            "dataset_name": "consolidatedShortInterest|weeklySummary",
            "best_symbol": "string",
            "best_quantity": "number",
            "results": [
                {
                    "symbol": "string",
                    "attempts": [
                        {
                            "settlement_date": "YYYY-MM-DD",
                            "quantity": "number|null",
                            "error": "string|null",
                        }
                    ],
                    "chosen_date": "YYYY-MM-DD",
                    "currentShortPositionQuantity": "number|null",
                    "totalWeeklyShareQuantity": "number|null",
                    "record": "object|null",
                }
            ],
            "errors": ["string"],
        },
        "fetch_short_interest": {
            "status": "ok|error",
            "symbol": "string",
            "requested_settlement_date": "YYYY-MM-DD",
            "dataset_name": "consolidatedShortInterest|weeklySummary",
            "chosen_date": "YYYY-MM-DD",
            "attempts": [
                {
                    "settlement_date": "YYYY-MM-DD",
                    "quantity": "number|null",
                    "error": "string|null",
                }
            ],
            "currentShortPositionQuantity": "number|null",
            "totalWeeklyShareQuantity": "number|null",
            "record": "object|null",
            "errors": ["string"],
        },
        "treasury_daily_aggregate": {
            "status": "ok|error",
            "tradeDate": "YYYY-MM-DD",
            "dataset_name": "treasuryDailyAggregates",
            "benchmark": "On-the-run|Off-the-run",
            "record": "object|null",
            "errors": ["string"],
        },
    }
    if is_treasury_delta:
        payload["response_format"]["treasury_daily_aggregate"].update(
            {
                "previous_trade_date": "YYYY-MM-DD",
                "best_years_to_maturity": "yearsToMaturity bucket string",
                "best_dealer_customer_volume_delta": "number",
                "record_current": "object|null",
                "record_previous": "object|null",
                "candidates_current": "array (rows considered)",
                "candidates_previous": "array (rows considered)",
                "attempts": "object (current/previous attempts)",
            }
        )
    elif is_treasury_max:
        payload["response_format"]["treasury_daily_aggregate"].update(
            {
                "best_years_to_maturity": "yearsToMaturity bucket string",
                "best_dealer_customer_volume": "number",
                "candidates": "array (rows considered)",
                "attempts": "array (date attempts)",
            }
        )
    else:
        payload["response_format"]["treasury_daily_aggregate"].update(
            {
                "yearsToMaturity": "<= 2 years",
                "dealerCustomerVolume": "number",
                "attempts": "array (date attempts)",
            }
        )
    return json.dumps(payload)


def _run_client_short(
    *,
    client_short_path: str,
    symbol: str,
    settlement_date: str,
    issue_name: str | None,
    dataset_group: str | None,
    dataset_name: str | None,
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
    if dataset_group:
        cmd.extend(["--dataset-group", dataset_group])
    if dataset_name:
        cmd.extend(["--dataset-name", dataset_name])

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


def _run_client_short_query(
    *,
    client_short_path: str,
    dataset_group: str | None,
    dataset_name: str | None,
    query_params: str | None,
    finra_client_id: str | None,
    finra_client_secret: str | None,
    timeout: int | None,
) -> Any:
    cmd = [sys.executable, client_short_path]
    if dataset_group:
        cmd.extend(["--dataset-group", dataset_group])
    if dataset_name:
        cmd.extend(["--dataset-name", dataset_name])
    if query_params:
        cmd.extend(["--query-params", query_params])
    if timeout:
        cmd.extend(["--timeout", str(timeout)])

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


def _tool_run_client_short(
    *,
    client_short_path: str,
    finra_client_id: str | None,
    finra_client_secret: str | None,
    timeout: int | None,
    dataset_group: str | None,
    dataset_name: str | None,
    symbols_allowed: set[str] | None,
    args: dict[str, Any],
) -> dict[str, Any]:
    symbol = str(args.get("symbol", "")).strip().upper()
    settlement_date = str(args.get("settlement_date", "")).strip()
    issue_name = str(args.get("issue_name", "")).strip() or None
    dataset_group_value = str(args.get("dataset_group") or args.get("datasetGroup") or "").strip() or None
    dataset_name_value = str(args.get("dataset_name") or args.get("datasetName") or "").strip() or None
    if not dataset_group_value:
        dataset_group_value = dataset_group
    if not dataset_name_value:
        dataset_name_value = dataset_name
    is_weekly = bool(dataset_name_value and "weeklysummary" in dataset_name_value.lower())
    if not symbol or not settlement_date:
        return {"error": "symbol and settlement_date are required"}
    if symbols_allowed is not None and symbol not in symbols_allowed:
        return {"error": f"symbol {symbol} is not in the provided symbol list"}

    mcp_command = os.environ.get("MCP_SERVER_COMMAND")
    if mcp_command:
        try:
            mcp_payload = _run_mcp_short_interest(
                command=mcp_command,
                client_short_path=client_short_path,
                finra_client_id=finra_client_id,
                finra_client_secret=finra_client_secret,
                timeout=timeout,
                dataset_group=dataset_group_value,
                dataset_name=dataset_name_value,
                symbol=symbol,
                settlement_date=settlement_date,
                issue_name=issue_name,
            )
            if isinstance(mcp_payload, dict) and mcp_payload.get("error"):
                return {
                    "symbol": symbol,
                    "settlement_date": settlement_date,
                    "error": str(mcp_payload.get("error")),
                }
            if isinstance(mcp_payload, dict):
                return mcp_payload
            return {
                "symbol": symbol,
                "settlement_date": settlement_date,
                "error": "Unexpected MCP response format",
            }
        except Exception as exc:
            return {"symbol": symbol, "settlement_date": settlement_date, "error": str(exc)}

    try:
        payload = _run_client_short(
            client_short_path=client_short_path,
            symbol=symbol,
            settlement_date=settlement_date,
            issue_name=issue_name,
            dataset_group=dataset_group_value,
            dataset_name=dataset_name_value,
            finra_client_id=finra_client_id,
            finra_client_secret=finra_client_secret,
            timeout=timeout,
        )
        if is_weekly:
            quantity, record = _extract_weekly_share(payload, symbol, settlement_date)
            return {
                "symbol": symbol,
                "settlement_date": settlement_date,
                "totalWeeklyShareQuantity": quantity,
                "record": record,
            }
        quantity, record = _extract_short_position(payload, symbol, settlement_date)
        return {
            "symbol": symbol,
            "settlement_date": settlement_date,
            "currentShortPositionQuantity": quantity,
            "record": record,
        }
    except Exception as exc:
        return {"symbol": symbol, "settlement_date": settlement_date, "error": str(exc)}


def _tool_run_treasury_daily(
    *,
    client_short_path: str,
    finra_client_id: str | None,
    finra_client_secret: str | None,
    timeout: int | None,
    dataset_group: str | None,
    dataset_name: str | None,
    args: dict[str, Any],
) -> dict[str, Any]:
    trade_date = str(
        args.get("trade_date")
        or args.get("tradeDate")
        or args.get("settlement_date")
        or ""
    ).strip()
    years_to_maturity = str(
        args.get("years_to_maturity")
        or args.get("yearsToMaturity")
        or args.get("expected_years_to_maturity")
        or args.get("expectedYearsToMaturity")
        or ""
    ).strip()
    benchmark = str(
        args.get("benchmark")
        or args.get("expected_benchmark")
        or args.get("expectedBenchmark")
        or ""
    ).strip()
    select_max = _coerce_bool(args.get("select_max") or args.get("selectMax"))
    bucket_explicit = _coerce_bool(
        args.get("bucket_explicit") or args.get("bucketExplicit")
    )
    dataset_group_value = str(
        args.get("dataset_group") or args.get("datasetGroup") or ""
    ).strip() or dataset_group
    dataset_name_value = str(
        args.get("dataset_name") or args.get("datasetName") or ""
    ).strip() or dataset_name
    if not trade_date:
        return {"error": "trade_date is required"}

    search_days = args.get("date_search_days") or args.get("dateSearchDays") or 7
    try:
        search_days = int(search_days)
    except (TypeError, ValueError):
        search_days = 7

    def _iter_search_dates(base: str, days: int) -> list[str]:
        parsed = _parse_date(base)
        if not parsed:
            return [base]
        offsets: list[int] = [0]
        for i in range(1, days + 1):
            offsets.append(-i)
            offsets.append(i)
        dates = []
        for offset in offsets:
            candidate = parsed + timedelta(days=offset)
            dates.append(candidate.strftime("%Y-%m-%d"))
        return dates

    def _fetch_payload(date_value: str) -> Any:
        query_params = f"tradeDate={date_value}&limit=200"
        mcp_command = os.environ.get("MCP_SERVER_COMMAND")
        if mcp_command:
            payload = _run_mcp_query_dataset(
                command=mcp_command,
                client_short_path=client_short_path,
                finra_client_id=finra_client_id,
                finra_client_secret=finra_client_secret,
                timeout=timeout,
                dataset_group=dataset_group_value,
                dataset_name=dataset_name_value,
                query_params=query_params,
            )
            if isinstance(payload, dict) and "data" in payload:
                payload = payload["data"]
            return payload
        return _run_client_short_query(
            client_short_path=client_short_path,
            dataset_group=dataset_group_value,
            dataset_name=dataset_name_value,
            query_params=query_params,
            finra_client_id=finra_client_id,
            finra_client_secret=finra_client_secret,
            timeout=timeout,
        )

    def _search_candidates(
        base_date: str,
        *,
        require_bucket: bool,
    ) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]]]:
        attempts: list[dict[str, Any]] = []
        for candidate_date in _iter_search_dates(base_date, search_days):
            payload = _fetch_payload(candidate_date)
            records = []
            for record in _normalize_records(payload):
                record_date = record.get("tradeDate")
                if not isinstance(record_date, str) or not record_date.startswith(
                    candidate_date
                ):
                    continue
                record_benchmark = str(record.get("benchmark") or "").strip()
                if record_benchmark.lower() != (benchmark or "On-the-run").lower():
                    continue
                record_years = str(record.get("yearsToMaturity") or "").strip()
                if require_bucket and years_to_maturity and record_years != years_to_maturity:
                    continue
                records.append(record)
            attempts.append({"tradeDate": candidate_date, "has_data": bool(records)})
            if records:
                return candidate_date, records, attempts
        return None, [], attempts

    try:
        if select_max and _coerce_bool(args.get("compare_previous")):
            previous_date = str(
                args.get("previous_trade_date")
                or args.get("previousTradeDate")
                or ""
            ).strip()
            if not previous_date:
                previous_date = _shift_year(trade_date, -1) or ""
            current_date, candidates_current, attempts_current = _search_candidates(
                trade_date, require_bucket=bucket_explicit
            )
            previous_date_resolved, candidates_previous, attempts_previous = (
                _search_candidates(previous_date, require_bucket=bucket_explicit)
            )
            if not current_date or not previous_date_resolved:
                return {
                    "tradeDate": trade_date,
                    "previous_trade_date": previous_date,
                    "benchmark": benchmark or "On-the-run",
                    "dataset_group": dataset_group_value,
                    "dataset_name": dataset_name_value,
                    "error": "No matching treasury records found",
                    "attempts": {
                        "current": attempts_current,
                        "previous": attempts_previous,
                    },
                }

            def _bucket_map(records: list[dict[str, Any]]) -> dict[str, float]:
                buckets: dict[str, float] = {}
                for record in records:
                    record_years = str(record.get("yearsToMaturity") or "").strip()
                    volume = _coerce_number(record.get("dealerCustomerVolume"))
                    if volume is None:
                        continue
                    current = buckets.get(record_years)
                    if current is None or volume > current:
                        buckets[record_years] = volume
                return buckets

            current_map = _bucket_map(candidates_current)
            previous_map = _bucket_map(candidates_previous)
            shared = [key for key in current_map if key in previous_map]
            if not shared:
                return {
                    "tradeDate": current_date,
                    "previous_trade_date": previous_date_resolved,
                    "benchmark": benchmark or "On-the-run",
                    "dataset_group": dataset_group_value,
                    "dataset_name": dataset_name_value,
                    "error": "No overlapping maturity buckets found",
                    "candidates_current": candidates_current,
                    "candidates_previous": candidates_previous,
                    "attempts": {
                        "current": attempts_current,
                        "previous": attempts_previous,
                    },
                }

            best_years = max(shared, key=lambda key: current_map[key] - previous_map[key])
            best_delta = current_map[best_years] - previous_map[best_years]
            record_current = next(
                (item for item in candidates_current if str(item.get("yearsToMaturity") or "").strip() == best_years),
                None,
            )
            record_previous = next(
                (item for item in candidates_previous if str(item.get("yearsToMaturity") or "").strip() == best_years),
                None,
            )
            return {
                "tradeDate": current_date,
                "previous_trade_date": previous_date_resolved,
                "benchmark": benchmark or "On-the-run",
                "best_years_to_maturity": best_years,
                "best_dealer_customer_volume_delta": best_delta,
                "record_current": record_current,
                "record_previous": record_previous,
                "candidates_current": candidates_current,
                "candidates_previous": candidates_previous,
                "attempts": {
                    "current": attempts_current,
                    "previous": attempts_previous,
                },
                "dataset_group": dataset_group_value,
                "dataset_name": dataset_name_value,
            }

        if select_max:
            resolved_date, candidates, attempts = _search_candidates(
                trade_date, require_bucket=bucket_explicit
            )
            if not resolved_date:
                return {
                    "tradeDate": trade_date,
                    "dataset_group": dataset_group_value,
                    "dataset_name": dataset_name_value,
                    "error": "No matching treasury records found",
                    "candidates": [],
                    "attempts": attempts,
                }
            best_record = max(
                candidates,
                key=lambda item: _coerce_number(item.get("dealerCustomerVolume"))
                or float("-inf"),
            )
            best_volume = _coerce_number(best_record.get("dealerCustomerVolume"))
            best_years = str(best_record.get("yearsToMaturity") or "").strip()
            if best_volume is None:
                return {
                    "tradeDate": resolved_date,
                    "benchmark": benchmark or "On-the-run",
                    "dataset_group": dataset_group_value,
                    "dataset_name": dataset_name_value,
                    "error": "No dealerCustomerVolume values found",
                    "candidates": candidates,
                    "attempts": attempts,
                }
            return {
                "tradeDate": resolved_date,
                "benchmark": benchmark or "On-the-run",
                "best_years_to_maturity": best_years,
                "best_dealer_customer_volume": best_volume,
                "record": best_record,
                "candidates": candidates,
                "attempts": attempts,
                "dataset_group": dataset_group_value,
                "dataset_name": dataset_name_value,
            }

        resolved_date, candidates, attempts = _search_candidates(
            trade_date, require_bucket=True
        )
        if not resolved_date or not candidates:
            return {
                "tradeDate": trade_date,
                "dataset_group": dataset_group_value,
                "dataset_name": dataset_name_value,
                "error": "No matching treasury record found",
                "record": None,
                "attempts": attempts,
            }
        record = candidates[0]
        volume = record.get("dealerCustomerVolume")
        return {
            "tradeDate": resolved_date,
            "dealerCustomerVolume": volume,
            "yearsToMaturity": years_to_maturity or "<= 2 years",
            "benchmark": benchmark or "On-the-run",
            "record": record,
            "attempts": attempts,
            "dataset_group": dataset_group_value,
            "dataset_name": dataset_name_value,
        }
    except Exception as exc:
        return {"tradeDate": trade_date, "error": str(exc)}


def _run_mcp_short_interest(
    *,
    command: str,
    client_short_path: str,
    finra_client_id: str | None,
    finra_client_secret: str | None,
    timeout: int | None,
    dataset_group: str | None,
    dataset_name: str | None,
    symbol: str,
    settlement_date: str,
    issue_name: str | None,
) -> dict[str, Any]:
    parts = shlex.split(command)
    if not parts:
        raise RuntimeError("MCP_SERVER_COMMAND is empty")
    params = StdioServerParameters(command=parts[0], args=parts[1:])
    tool_args = {
        "symbol": symbol,
        "settlement_date": settlement_date,
        "issue_name": issue_name,
        "client_short_path": client_short_path,
        "finra_client_id": finra_client_id,
        "finra_client_secret": finra_client_secret,
        "timeout": timeout,
        "dataset_group": dataset_group,
        "dataset_name": dataset_name,
    }
    return _run_async(_call_mcp_tool(params, tool_args))


def _run_mcp_query_dataset(
    *,
    command: str,
    client_short_path: str,
    finra_client_id: str | None,
    finra_client_secret: str | None,
    timeout: int | None,
    dataset_group: str | None,
    dataset_name: str | None,
    query_params: str,
) -> dict[str, Any]:
    parts = shlex.split(command)
    if not parts:
        raise RuntimeError("MCP_SERVER_COMMAND is empty")
    params = StdioServerParameters(command=parts[0], args=parts[1:])
    tool_args = {
        "client_short_path": client_short_path,
        "finra_client_id": finra_client_id,
        "finra_client_secret": finra_client_secret,
        "timeout": timeout,
        "dataset_group": dataset_group,
        "dataset_name": dataset_name,
        "query_params": query_params,
    }
    return _run_async(
        _call_mcp_tool(params, tool_args, tool_name="finra_query_dataset")
    )


async def _call_mcp_tool(
    params: StdioServerParameters,
    tool_args: dict[str, Any],
    tool_name: str = "finra_short_interest",
) -> dict[str, Any]:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, tool_args)
            return _coerce_mcp_result(result)


def _coerce_mcp_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        return {"data": result}
    content = getattr(result, "content", None)
    if isinstance(content, list) and content:
        first = content[0]
        text = getattr(first, "text", None)
        if isinstance(text, str) and text.strip():
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"error": f"Invalid JSON from MCP: {text}"}
    return {"error": "Unexpected MCP response"}


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_container: dict[str, Any] = {}
    error_container: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result_container["value"] = asyncio.run(coro)
        except BaseException as exc:
            error_container["error"] = exc

    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join()
    if error_container:
        raise error_container["error"]
    return result_container.get("value")


def _run_llm(
    *,
    task: str,
    symbol: str | None,
    symbols: list[str] | None,
    requested_date: str,
    min_attempts: int,
    question: str | None,
    client_short_path: str,
    dataset_group: str | None,
    dataset_name: str | None,
    finra_client_id: str | None,
    finra_client_secret: str | None,
    timeout: int | None,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_client_short",
                "description": "Run client_short.py for one symbol and settlement date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "settlement_date": {"type": "string"},
                        "issue_name": {"type": "string"},
                        "dataset_group": {"type": "string"},
                        "dataset_name": {"type": "string"},
                    },
                    "required": ["symbol", "settlement_date"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_treasury_daily",
                "description": "Query treasuryDailyAggregates and return dealerCustomerVolume.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "trade_date": {"type": "string"},
                        "dataset_group": {"type": "string"},
                        "dataset_name": {"type": "string"},
                        "years_to_maturity": {"type": "string"},
                        "benchmark": {"type": "string"},
                        "select_max": {"type": "boolean"},
                        "bucket_explicit": {"type": "boolean"},
                    },
                    "required": ["trade_date"],
                },
            },
        },
    ]

    system_prompt = _build_system_prompt(min_attempts)
    user_prompt = _build_user_prompt(
        task,
        symbol,
        symbols,
        requested_date,
        min_attempts,
        question,
        dataset_group,
        dataset_name,
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    symbols_allowed = {item.upper() for item in symbols} if symbols else None

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            output_text = message.content or ""
            if not output_text.strip():
                raise RuntimeError("LLM returned empty output")
            payload = json.loads(output_text)
            if isinstance(payload, dict):
                payload = _attach_dataset_info(payload, task, dataset_group, dataset_name)
            if task == "max_short_interest" and symbols:
                unwrapped = _unwrap_response(payload, task)
                if isinstance(unwrapped, dict):
                    missing = _missing_symbols(unwrapped, symbols, min_attempts)
                    if missing:
                        messages.append(message)
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "You are missing results for: "
                                    + ", ".join(missing)
                                    + f". Call the tool for each missing symbol at least {min_attempts} times "
                                    "near the requested date, then return the full JSON for all symbols."
                                ),
                            }
                        )
                        continue
            return payload

        messages.append(message)
        for call in tool_calls:
            arguments = call.function.arguments or "{}"
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                args = {}
            if call.function.name == "run_client_short":
                tool_payload = _tool_run_client_short(
                    client_short_path=client_short_path,
                    finra_client_id=finra_client_id,
                    finra_client_secret=finra_client_secret,
                    timeout=timeout,
                    dataset_group=dataset_group,
                    dataset_name=dataset_name,
                    symbols_allowed=symbols_allowed,
                    args=args,
                )
            elif call.function.name == "run_treasury_daily":
                is_max = _is_treasury_max_question(question)
                is_delta = _is_treasury_delta_question(question)
                bucket_explicit = _has_treasury_bucket(question)
                args.setdefault("select_max", is_max or is_delta)
                if is_delta:
                    args.setdefault("compare_previous", True)
                    args.setdefault("previous_trade_date", _shift_year(requested_date, -1) or "")
                args.setdefault("bucket_explicit", bucket_explicit)
                expected_years, expected_benchmark = _parse_treasury_bucket(question)
                if "benchmark" not in args:
                    args["benchmark"] = expected_benchmark
                if (not is_max and not is_delta or bucket_explicit) and (
                    "years_to_maturity" not in args and "yearsToMaturity" not in args
                ):
                    args["years_to_maturity"] = expected_years
                tool_payload = _tool_run_treasury_daily(
                    client_short_path=client_short_path,
                    finra_client_id=finra_client_id,
                    finra_client_secret=finra_client_secret,
                    timeout=timeout,
                    dataset_group=dataset_group,
                    dataset_name=dataset_name,
                    args=args,
                )
            else:
                tool_payload = {"error": f"Unsupported tool: {call.function.name}"}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.function.name,
                    "content": json.dumps(tool_payload),
                }
            )

    raise RuntimeError("LLM did not complete tool calls in time")


class Agent:
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        response: dict[str, Any] = {
            "status": "error",
            "error": "Unknown error",
        }

        try:
            request = PurpleRequest.model_validate_json(input_text)
        except ValidationError as exc:
            response["error"] = f"Invalid request: {exc}"
            await updater.add_artifact(
                parts=[Part(root=DataPart(data=response))], name="Result"
            )
            return

        args = request.args or {}
        symbol = str(args.get("symbol", "")).strip() or None
        settlement_date = str(args.get("settlement_date", "")).strip()
        trade_date = str(
            args.get("trade_date") or args.get("tradeDate") or settlement_date or ""
        ).strip()
        symbols_list = _normalize_symbols(args.get("symbols"))
        question = request.question or args.get("question")
        dataset_group = request.dataset_group or args.get("dataset_group") or args.get("datasetGroup")
        dataset_name = request.dataset_name or args.get("dataset_name") or args.get("datasetName")
        if not dataset_group or not dataset_name:
            inferred_group, inferred_name = _infer_dataset_from_question(question)
            if not dataset_group:
                dataset_group = inferred_group
            if not dataset_name:
                dataset_name = inferred_name
        is_treasury = bool(dataset_name and "treasurydailyaggregates" in dataset_name.lower())
        requested_date = request.requested_settlement_date or (
            trade_date if is_treasury else settlement_date
        )

        allowed_tasks = {None, "fetch_short_interest", "max_short_interest", "treasury_daily_aggregate"}
        if request.task not in allowed_tasks:
            response["error"] = f"Unsupported task: {request.task}"
            await updater.add_artifact(
                parts=[Part(root=DataPart(data=response))], name="Result"
            )
            return

        client_short_path = request.client_short_path or DEFAULT_CLIENT_SHORT_PATH
        if not os.path.exists(client_short_path):
            response["error"] = f"client_short.py not found: {client_short_path}"
            await updater.add_artifact(
                parts=[Part(root=DataPart(data=response))], name="Result"
            )
            return

        task_name = request.task or (
            "treasury_daily_aggregate" if is_treasury else ("max_short_interest" if symbols_list else "fetch_short_interest")
        )

        if task_name == "treasury_daily_aggregate":
            if not trade_date:
                response["error"] = "Missing required args: trade_date"
                await updater.add_artifact(
                    parts=[Part(root=DataPart(data=response))], name="Result"
                )
                return
        elif task_name == "max_short_interest":
            if not symbols_list or not settlement_date:
                response["error"] = "Missing required args: symbols and settlement_date"
                await updater.add_artifact(
                    parts=[Part(root=DataPart(data=response))], name="Result"
                )
                return
        else:
            if not symbol or not settlement_date:
                response["error"] = "Missing required args: symbol and settlement_date"
                await updater.add_artifact(
                    parts=[Part(root=DataPart(data=response))], name="Result"
                )
                return

        if task_name == "treasury_daily_aggregate":
            if not _parse_date(trade_date):
                response["error"] = "trade_date must be in YYYY-MM-DD format"
                await updater.add_artifact(
                    parts=[Part(root=DataPart(data=response))], name="Result"
                )
                return
        elif not _parse_date(settlement_date):
            response["error"] = "settlement_date must be in YYYY-MM-DD format"
            await updater.add_artifact(
                parts=[Part(root=DataPart(data=response))], name="Result"
            )
            return

        min_attempts = request.min_attempts or MIN_ATTEMPTS

        try:
            response = _run_llm(
                task=task_name,
                symbol=symbol,
                symbols=symbols_list,
                requested_date=requested_date,
                min_attempts=min_attempts,
                question=question,
                client_short_path=client_short_path,
                dataset_group=dataset_group,
                dataset_name=dataset_name,
                finra_client_id=request.finra_client_id,
                finra_client_secret=request.finra_client_secret,
                timeout=request.timeout,
            )
            if isinstance(response, dict):
                for key in ("max_short_interest", "fetch_short_interest"):
                    payload = response.get(key)
                    if isinstance(payload, dict):
                        response = payload
                        break
        except Exception as exc:
            response = {"status": "error", "error": str(exc)}

        await updater.add_artifact(
            parts=[Part(root=DataPart(data=response))], name="Result"
        )
