from typing import Any
from datetime import date, datetime
import json
import os
import subprocess
import sys

from openai import OpenAI

from pydantic import BaseModel, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, DataPart
from a2a.utils import get_message_text


DEFAULT_CLIENT_SHORT_PATH = "/home/wczubal1/projects/tau2/brokercheck/client_short.py"
MIN_ATTEMPTS = 3
MAX_TOOL_ROUNDS = 8


class PurpleRequest(BaseModel):
    task: str | None = None
    client_short_path: str | None = None
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


def _build_system_prompt(min_attempts: int) -> str:
    return (
        "You are a purple agent. You must use the provided tool to query FINRA short "
        "interest data. Data is reported twice per month (15th and month-end). "
        f"For each symbol, try at least {min_attempts} different settlement dates near the "
        "requested date to find the closest available date with data. "
        "Return JSON only (no markdown)."
    )


def _build_user_prompt(
    task: str,
    symbol: str | None,
    symbols: list[str] | None,
    requested_date: str,
    min_attempts: int,
) -> str:
    payload: dict[str, Any] = {
        "task": task,
        "requested_settlement_date": requested_date,
        "min_attempts": min_attempts,
    }
    if symbol:
        payload["symbol"] = symbol
    if symbols:
        payload["symbols"] = symbols
    payload["response_format"] = {
        "max_short_interest": {
            "status": "ok|error",
            "requested_settlement_date": "YYYY-MM-DD",
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
                    "record": "object|null",
                }
            ],
            "errors": ["string"],
        },
        "fetch_short_interest": {
            "status": "ok|error",
            "symbol": "string",
            "requested_settlement_date": "YYYY-MM-DD",
            "chosen_date": "YYYY-MM-DD",
            "attempts": [
                {
                    "settlement_date": "YYYY-MM-DD",
                    "quantity": "number|null",
                    "error": "string|null",
                }
            ],
            "currentShortPositionQuantity": "number|null",
            "record": "object|null",
            "errors": ["string"],
        },
    }
    return json.dumps(payload)


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


def _tool_run_client_short(
    *,
    client_short_path: str,
    finra_client_id: str | None,
    finra_client_secret: str | None,
    timeout: int | None,
    symbols_allowed: set[str] | None,
    args: dict[str, Any],
) -> dict[str, Any]:
    symbol = str(args.get("symbol", "")).strip().upper()
    settlement_date = str(args.get("settlement_date", "")).strip()
    issue_name = str(args.get("issue_name", "")).strip() or None
    if not symbol or not settlement_date:
        return {"error": "symbol and settlement_date are required"}
    if symbols_allowed is not None and symbol not in symbols_allowed:
        return {"error": f"symbol {symbol} is not in the provided symbol list"}

    try:
        payload = _run_client_short(
            client_short_path=client_short_path,
            symbol=symbol,
            settlement_date=settlement_date,
            issue_name=issue_name,
            finra_client_id=finra_client_id,
            finra_client_secret=finra_client_secret,
            timeout=timeout,
        )
        quantity, record = _extract_short_position(payload, symbol, settlement_date)
        return {
            "symbol": symbol,
            "settlement_date": settlement_date,
            "currentShortPositionQuantity": quantity,
            "record": record,
        }
    except Exception as exc:
        return {"symbol": symbol, "settlement_date": settlement_date, "error": str(exc)}


def _run_llm(
    *,
    task: str,
    symbol: str | None,
    symbols: list[str] | None,
    requested_date: str,
    min_attempts: int,
    client_short_path: str,
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
                    },
                    "required": ["symbol", "settlement_date"],
                },
            },
        }
    ]

    system_prompt = _build_system_prompt(min_attempts)
    user_prompt = _build_user_prompt(task, symbol, symbols, requested_date, min_attempts)
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
            return json.loads(output_text)

        messages.append(message)
        for call in tool_calls:
            if call.function.name != "run_client_short":
                tool_payload = {
                    "error": f"Unsupported tool: {call.function.name}"
                }
            else:
                arguments = call.function.arguments or "{}"
                try:
                    args = json.loads(arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_payload = _tool_run_client_short(
                    client_short_path=client_short_path,
                    finra_client_id=finra_client_id,
                    finra_client_secret=finra_client_secret,
                    timeout=timeout,
                    symbols_allowed=symbols_allowed,
                    args=args,
                )
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
        symbols_list = _normalize_symbols(args.get("symbols"))
        requested_date = request.requested_settlement_date or settlement_date

        allowed_tasks = {None, "fetch_short_interest", "max_short_interest"}
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

        task_name = request.task or ("max_short_interest" if symbols_list else "fetch_short_interest")

        if task_name == "max_short_interest":
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

        if not _parse_date(settlement_date):
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
                client_short_path=client_short_path,
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
