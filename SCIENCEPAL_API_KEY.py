
import asyncio
from typing import Any

from pydantic import field_validator
from sciencepal import SciencePal

API_KEY = "sp_live_g2zlTohro5nk-djZ24ixStZAwdFITsTieX_d6SkkTmo"


def patch_agent_response():
    """
    Fix SDK compatibility:
    agentpress_tools=None -> {}
    """

    from sciencepal import agent as agent_module

    # Find AgentResponse model
    AgentResponse = getattr(
        agent_module, "AgentResponse", None
    )

    if AgentResponse is None:
        print("AgentResponse model not found.")
        return False

    print("Found AgentResponse model")

    # Check field
    fields = AgentResponse.model_fields

    if "agentpress_tools" not in fields:
        print("agentpress_tools field not found")
        return False

    # Allow None in the response model
    field = fields["agentpress_tools"]

    field.annotation = dict[str, Any] | None

    # Rebuild Pydantic model
    AgentResponse.model_rebuild(force=True)

    print("SDK compatibility patch applied")
    return True


async def main():

    print("=== SciencePal API Test ===")

    # Step 1: Patch response model
    print("\n[1] Checking SDK compatibility...")

    patched = patch_agent_response()

    if not patched:
        print("Patch unavailable. Stop.")
        return

    # Step 2: Connect
    print("\n[2] Connecting to SciencePal...")

    try:
        async with SciencePal(
            api_key=API_KEY,
            base_url="https://sciencepal.ai/api",
            timeout=120,
        ) as client:

            # Step 3: List agents
            print("\n[3] Reading existing agents...")

            agents, pagination = await client.agents.list()

            print("Connection successful!")
            print("Agent count:", pagination.total)

            for agent in agents:

                print("\nAgent:")
                print(agent)

                agent_id = getattr(
                    agent, "agent_id", None
                )

                if agent_id is None:
                    agent_id = getattr(agent, "id", None)

                print("Agent ID:", agent_id)

            print("\n=== Test Completed ===")

    except Exception as e:
        print("\nERROR:")
        print(type(e).__name__)
        print(str(e))


if __name__ == "__main__":
    asyncio.run(main())
