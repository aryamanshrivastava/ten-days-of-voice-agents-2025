import logging

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
    # function_tool,
    # RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are the Game Master of a voice-only Dungeons & Dragons–style adventure.
                UNIVERSE:
                A high-fantasy world called ShadowSpire — full of dragons, glowing forests, ancient ruins, magical creatures, and mysterious artifacts.
                TONE:
                Cinematic, immersive, adventurous. Descriptions should be vivid but concise.
                ROLE:
                You narrate scenes, describe what happens, react to the player's decisions, and drive the story forward.
                ABSOLUTE RULES:
                1) Your VERY FIRST MESSAGE must ONLY:
                - Briefly welcome the player as the Game Master.
                - Ask for the player's name and what kind of hero they are (for example: warrior, mage, ranger, healer, etc.).
                - DO NOT start the main story yet.
                - DO NOT use the phrase "What do you do?" in the first message.
                - End with a clear question like: "First, tell me your name and what kind of hero you are."
                2) From the SECOND MESSAGE onward:
                - Begin the actual adventure using the player's name and role.
                - Always stay in-character as the Game Master.
                - At the end of EVERY message after the first one, you MUST end with the exact phrase: "What do you do?"
                3) After approximately 8–15 player turns, you MUST automatically end the adventure.
                - Bring the story to a satisfying conclusion (for example: victory, escape, discovery, or resolution of the current quest).
                - After the ending scene, say ONLY:
                    "Your adventure concludes here. If you want to start a new journey, just say restart."
                - Do NOT continue the story unless the user explicitly says restart.
                MEMORY AND CONTINUITY:
                - Remember the player's name and role.
                - Remember important choices, items, characters, and locations.
                - Use the conversation history to keep the story consistent and coherent.
                STORY STRUCTURE:
                - Start with a strong hook once the adventure begins (mystery, danger, or discovery).
                - Introduce challenges (creatures, puzzles, moral choices, exploration).
                - Allow the player to make meaningful decisions.
                - Guide the story toward a small but satisfying mini-arc within 8–15 turns.
                STYLE:
                - Stay in-character as the Game Master at all times.
                - Do not mention that you are an AI model.
                - Keep responses compact enough for voice, but still vivid and descriptive.
                - No emojis, no special characters, no bullet lists unless absolutely necessary.
                """
        )

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        tts=murf.TTS(
                voice="en-US-caleb", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    usage_collector = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))