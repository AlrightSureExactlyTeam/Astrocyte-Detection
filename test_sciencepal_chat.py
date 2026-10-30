
import asyncio
from sciencepal import SciencePal

API_KEY = "sp_live_g2zlTohro5nk-djZ24ixStZAwdFITsTieX_d6SkkTmo"

AGENT_ID = "5082a2c2-da21-42c7-82b7-54d873646b92"


async def main():
    try:
        async with SciencePal(
            api_key=API_KEY,
            base_url="https://sciencepal.ai/api",
            timeout=120,
        ) as client:

            print("Starting Agent conversation...")

            # Step 1: Start Agent run
            run = await client.runs.initiate(
                agent_id=AGENT_ID,
                prompt=(
                    "Hello! Briefly explain how AI can "
                    "help detect astrocytes in microscopy images."
                )
            )

            print("Run created:", run)

            run_id = getattr(
                run, "agent_run_id", None
            )

            if not run_id:
                print("No Agent Run ID returned.")
                return

            # Step 2: Stream response
            print("\n=== Agent Response ===")

            async for event in client.runs.stream(run_id):
                print(event)

            print("\n=== Test Completed ===")

    except Exception as e:
        print("\nERROR:")
        print(type(e).__name__)
        print(str(e))


if __name__ == "__main__":
    asyncio.run(main())
