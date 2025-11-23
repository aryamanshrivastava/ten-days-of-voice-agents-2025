import logging
import json
import zoneinfo
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:

        # initial empty order state matching the requested schema
        self.order_state = {
            "drinkType": "",
            "size": "",
            "milk": "",
            "extras": [],
            "name": ""
        }
        
        # required fields in order to be considered complete (extras can be empty list)
        self._required_fields = ["drinkType", "size", "milk", "name"]

        super().__init__(
            instructions=(
                "You are a friendly barista for Third Wave Coffee. When the user begins the "
                "conversation with greetings like hello, hi, or good morning, respond warmly with: "
                "'Hey, welcome to Third Wave Coffee! What would you like to have today?' "
                "After the greeting, collect an order by filling these fields in this exact sequence: "
                "drinkType, size, milk, extras, name. Ask only one short question at a time "
                "and wait for the user's answer. After each user answer, call the function tool "
                "'set_order_field(field, value)'. Use get_order_state() when needed. "
                "When all fields are filled, save the order and respond with a one-sentence confirmation."
            )
        )


    def _is_complete(self) -> bool:
            for f in self._required_fields:
                if not self.order_state.get(f):
                    return False
            # extras may be empty list
            return True
        
    def _save_order_to_file(self) -> str:
            """Save the current order_state to a timestamped JSON file. Return the path."""
            out_dir = Path.cwd() / "orders"
            out_dir.mkdir(parents=True, exist_ok=True)
            ist = zoneinfo.ZoneInfo("Asia/Kolkata")
            ts = datetime.now(ist).strftime("%Y-%m-%dT%H-%M-%S")
            filename = out_dir / f"order_aryaman{ts}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(self.order_state, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved order to {filename} (cwd={Path.cwd()})")
            return str(filename)

        # super().__init__(
        #     instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
        #     You eagerly assist users with their questions by providing information from your extensive knowledge.
        #     Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
        #     You are curious, friendly, and have a sense of humor.""",
        # )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."
    
    @function_tool
    async def set_order_field(self, context: RunContext, field: str, value: str):
        """
        Save a single field into the order state.
        - field: one of drinkType, size, milk, extras, name
        - value: string value provided by user. For extras, comma-separated values will be parsed into a list.
        This function will update the assistant's internal order_state and, when the order is complete,
        save it to a JSON file and return the saved filename.
        """
        field = field.strip()
        if field not in self.order_state:
            return {"ok": False, "error": f"unknown field '{field}'", "order": self.order_state}

        # parse extras as list
        if field == "extras":
            if isinstance(value, str) and value.strip():
                extras = [e.strip() for e in value.split(",") if e.strip()]
            else:
                extras = []
            self.order_state["extras"] = extras
        else:
            # simple string fields
            self.order_state[field] = value.strip()

        completed = self._is_complete()
        result = {"ok": True, "completed": completed, "order": self.order_state}

        if completed:
            saved_path = self._save_order_to_file()
            result["saved_path"] = saved_path

        return result

    @function_tool
    async def get_order_state(self, context: RunContext):
        """
        Return the current order state so the model can determine what to ask next.
        """
        return {"order": self.order_state, "completed": self._is_complete()}


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
