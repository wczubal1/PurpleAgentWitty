from typing import Any
from datetime import date, datetime
import asyncio
import json
import os
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


QUESTION_WEEKLY_KEYWORDS = ("weekly", "week", "weeklysummary", "weekly summary")
QUESTION_SHARE_KEYWORDS = ("share", "shares", "totalweeklysharequantity", "total weekly share")
QUESTION_SHORT_KEYWORDS = ("short interest", "short position", "current short")


def _infer_dataset_from_question(question: str | None) -> tuple[str | None, str | None]:
    if not question:
        return None, None
    lowered = question.lower()
    if any(key in lowered for key in QUESTION_WEEKLY_KEYWORDS) and any(
        key in lowered for key in QUESTION_SHARE_KEYWORDS
    ):
        return "otcmarket", "weeklySummary"
    if any(key in lowered for key in QUESTION_SHORT_KEYWORDS):
        return "otcmarket", "consolidatedShortInterest"
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
    target = payload
    nested = payload.get(task)
    if isinstance(nested, dict):
        target = nested
    if dataset_group and "dataset_group" not in target and "datasetGroup" not in target:
        target["dataset_group"] = dataset_group
    if dataset_name and "dataset_name" not in target and "datasetName" not in target:
        target["dataset_name"] = dataset_name
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
    }
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


async def _call_mcp_tool(
    params: StdioServerParameters,
    tool_args: dict[str, Any],
) -> dict[str, Any]:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("finra_short_interest", tool_args)
            return _coerce_mcp_result(result)


def _coerce_mcp_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
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
        }
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
                    dataset_group=dataset_group,
                    dataset_name=dataset_name,
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
        question = request.question or args.get("question")
        dataset_group = request.dataset_group or args.get("dataset_group") or args.get("datasetGroup")
        dataset_name = request.dataset_name or args.get("dataset_name") or args.get("datasetName")
        if not dataset_group or not dataset_name:
            inferred_group, inferred_name = _infer_dataset_from_question(question)
            if not dataset_group:
                dataset_group = inferred_group
            if not dataset_name:
                dataset_name = inferred_name

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
