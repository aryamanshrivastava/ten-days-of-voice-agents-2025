import logging
import asyncio
import json
import os
import zoneinfo
from datetime import datetime
from typing import Any
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

# Path for JSON persistence
WELLNESS_LOG_PATH = os.path.join(os.path.dirname(__file__), "wellness_log.json")


def ensure_log_file():
    if not os.path.exists(WELLNESS_LOG_PATH):
        with open(WELLNESS_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)

ensure_log_file()

# Module-level lock to guard concurrent coroutines in this process
_file_lock = asyncio.Lock()


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert common non-serializable types to serializable ones."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat() + "Z"
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    # Fallback: convert unknown objects to their string representation
    return str(obj)


class Assistant(Agent):
    def __init__(self) -> None:
         super().__init__(
            instructions=(
                """
                You are a concise, friendly daily wellness voice companion. Each session is a short check-in (1-3 minutes).
                On session start:
                - Call `load_last_checkin`. If a previous check-in exists, reference it in one brief, natural sentence.
                Flow of the check-in:
                1. Begin with a warm, simple greeting that includes the user's name: “Hi Aryaman” or a natural variation.
                2. Ask about the user's mood and energy using open, supportive, phrasing. 
                Use natural variations such as asking how they're feeling, how their energy is today, or whether anything is on their mind.
                - Do NOT offer medical diagnoses or medical claims.
                3. Ask for simple objectives they want to accomplish today.
                - Also ask whether they want to do anything for themselves (rest, movement, hobbies, or something enjoyable).
                4. After they answer, offer one short piece of advice that is:
                - Small, realistic, and actionable
                - Non-medical and non-diagnostic
                - Grounded.
                5. End with a brief recap:
                - A plain-language summary of the user's mood
                - The main objectives for the day
                - Ask, “Does this sound right?”
                When you have the user's final responses:
                - Call the `save_checkin` tool with a JSON object containing:
                - A timestamp
                - Reported mood and energy (numeric only if the user explicitly gives numbers)
                - Objectives list
                - A short summary sentence
                Keep your language spoken-friendly: no complex formatting, no emojis, and no code blocks in the conversation.
                """
            )
        )
         
    @function_tool
    async def save_checkin(context: RunContext, data: dict):
        """
        Persist a check-in entry to wellness_log.json.
        Expected `data` keys (minimum):
        - 'mood' : str
        - 'objectives' : list[str]
        - 'energy' : optional str or numeric
        - 'summary' : optional str
        Adds 'timestamp' if missing and returns {"status": "ok", "entry": entry}.
        """
        # Basic validation
        if not isinstance(data, dict):
            return {"status": "error", "error": "save_checkin expects a dict in `data`"}

        # Normalize and sanitize entry
        entry = dict(data)  # shallow copy
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.utcnow()

        # Convert problematic types (datetimes, objects) into JSON-friendly values
        entry = _make_json_serializable(entry)

        # Acquire lock so multiple coroutines won't clobber the file
        async with _file_lock:
            try:
                ensure_log_file()  # ensures file exists
                # Read existing content
                try:
                    with open(WELLNESS_LOG_PATH, "r", encoding="utf-8") as f:
                        arr = json.load(f)
                        if not isinstance(arr, list):
                            # unexpected schema — recover by treating as empty list
                            arr = []
                except json.JSONDecodeError:
                    # The JSON is corrupted. Back it up and start fresh to avoid crashing.
                    ist = zoneinfo.ZoneInfo("Asia/Kolkata")
                    corrupt_path = WELLNESS_LOG_PATH + ".corrupt." + datetime.now(ist).strftime("%Y-%m-%dT%H-%M-%S")
                    try:
                        os.replace(WELLNESS_LOG_PATH, corrupt_path)
                        logger.warning(f"wellness_log.json was invalid JSON — moved to {corrupt_path}")
                    except Exception:
                        logger.exception("Failed to move corrupted wellness_log.json; proceeding with fresh file")
                    arr = []

                arr.append(entry)

                # Atomic write: write to tmp file then replace
                tmp_path = WELLNESS_LOG_PATH + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(arr, f, indent=2, ensure_ascii=False)

                os.replace(tmp_path, WELLNESS_LOG_PATH)

                logger.info("Saved check-in to wellness_log.json")
                return {"status": "ok", "entry": entry}
            except Exception as e:
                logger.exception("Failed to save check-in")
                return {"status": "error", "error": str(e)}

    

    @function_tool
    async def load_last_checkin(context: RunContext):
        """
        Return the most recent check-in entry, or None if none exist.
        """
        try:
            ensure_log_file()
            with open(WELLNESS_LOG_PATH, "r", encoding="utf-8") as f:
                arr = json.load(f)
                if isinstance(arr, list) and arr:
                    return arr[-1]
                else:
                    return None
        except Exception as e:
            logger.exception("Failed to load last check-in")
            return None

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
