import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the purple A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=9010, help="Port to bind")
    parser.add_argument("--card-url", type=str, help="URL to advertise in agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="short-interest-fetch",
        name="Short Interest Fetch",
        description=(
            "Calls client_short.py and returns short interest JSON. Requires "
            "OPENAI_API_KEY and FINRA credentials supplied at runtime."
        ),
        tags=["finra", "short-interest", "tool"],
        examples=[
            '{"task":"max_short_interest","client_short_path":"/home/wczubal1/projects/tau2/brokercheck/client_short.py","requested_settlement_date":"2025-05-15","min_attempts":3,"args":{"symbols":["TSLA","NVDA","AAPL"],"settlement_date":"2025-05-15"}}'
        ],
    )

    agent_card = AgentCard(
        name="Short Interest Purple Agent",
        description=(
            "Executes client_short.py and returns JSON artifacts. Requires "
            "OPENAI_API_KEY and FINRA credentials supplied at runtime."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
