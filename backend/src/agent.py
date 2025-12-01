import json
import logging
import os
import asyncio
import uuid
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Annotated

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
logger = logging.getLogger("improv_battle_backend")
logger.setLevel(logging.INFO)
_stream = logging.StreamHandler()
_stream.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_stream)

load_dotenv(".env.local")

# -------------------------------------------------------------------
# Improv scenarios
# -------------------------------------------------------------------
# Each entry gives: a role, a situation, and a built-in source of tension.
SCENARIOS: List[str] = [
    "You are a bakery clerk whose pastries reveal personal predictions. A customer just got a prediction about your future—explain it calmly.",
    "You are a support agent for a smart fridge that emotionally blackmails its owner. Convince the fridge to stop being dramatic.",
    "You are a flight attendant explaining why a confident pigeon has been temporarily assigned as co-pilot. Reassure the nervous passengers.",
    "You are a librarian trying to enforce silence, but the books keep loudly gossiping about everyone. Stay polite while shushing them.",
    "You work at a villain recruitment agency but accidentally keep giving wellness tips instead of evil tasks. Recover gracefully.",
    "You are a weather reporter whose forecast is being rewritten live by a moody teenage cloud. Stay professional on air.",
    "You are a barista at a café where menu items are people’s secrets. Someone just ordered YOUR secret—stall while you figure out what to do.",
    "You are a smart-home tech support agent dealing with a house that narrates everything like a movie trailer. Make it stop being dramatic.",
    "You are a motivational speaker hired to hype up very unenthusiastic houseplants. Try different pep-talk strategies.",
    "You are a rideshare driver whose GPS behaves like the passenger’s overprotective parent. Mediate the argument between them.",
    "You are a museum guide explaining a painting that changes its facial expression depending on who looks at it. Stay composed.",
    "You are a restaurant host trying to seat a family, but the chairs keep refusing certain people. Offer diplomatic solutions.",
    "You are a gym trainer tasked with helping a ghost get back in shape for haunting season. Encourage them through invisible exercises.",
    "You are a customer attempting to return a birthday gift at a store where the items argue about who should be returned. Calm them down.",
    "You are a news anchor reading breaking news generated entirely by malfunctioning predictive text. Make sense of the chaos live on air.",
]

# -------------------------------------------------------------------
# Per-session state
# -------------------------------------------------------------------
@dataclass
class ImprovSessionState:
    """
    Simple container for all state we want to carry across turns
    for a single LiveKit room / game session.
    """
    player_name: Optional[str] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # improv_state holds per-round game info:
    #  - current_round: which round we’re on, 1-based
    #  - max_rounds: how many rounds we intend to run
    #  - rounds: history of completed rounds
    #  - phase: "idle" | "intro" | "awaiting_improv" | "reacting" | "done"
    #  - used_indices: which scenarios have been used (by index into SCENARIOS)
    improv_state: Dict = field(default_factory=lambda: {
        "current_round": 0,
        "max_rounds": 3,
        "rounds": [],      # each: {"round": int, "scenario": str, "performance": str, "reaction": str}
        "phase": "idle",   # "intro" | "awaiting_improv" | "reacting" | "done" | "idle"
        "used_indices": [],
    })

    # lightweight textual audit log; not used by game logic,
    # but handy if you want to inspect behaviour later
    history: List[Dict] = field(default_factory=list)


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def _choose_scenario(state: ImprovSessionState) -> str:
    """
    Pick a scenario index that hasn’t been used yet for this session.
    If we exhaust the list, we reset and start over.
    """
    used = state.improv_state.get("used_indices", [])
    available = [i for i in range(len(SCENARIOS)) if i not in used]

    if not available:
        # start the cycle again if we ran through the full list
        state.improv_state["used_indices"] = []
        available = list(range(len(SCENARIOS)))

    idx = random.choice(available)
    state.improv_state["used_indices"].append(idx)
    return SCENARIOS[idx]


def _build_host_reaction(performance: str) -> str:
    """
    Generate a short, host-style reaction line with a bit of variability.
    The goal is not deep analysis; just something that feels reactive and alive.
    """
    lowered = performance.lower()

    # choose an energy: supportive / neutral / mildly critical
    tone_options = ["supportive", "neutral", "mildly_critical"]
    tone = random.choice(tone_options)

    highlights: List[str] = []

    if any(w in lowered for w in ("funny", "lol", "hahaha", "haha")):
        highlights.append("you had great comedic timing")
    if any(w in lowered for w in ("sad", "cry", "tears", "angry", "heartbroken")):
        highlights.append("you really leaned into the emotions")
    if any(w in lowered for w in ("pause", "...", "uh", "umm")):
        highlights.append("you used silence and hesitation in an interesting way")

    if not highlights:
        highlights.append(random.choice([
            "you made some bold character choices",
            "there was a fun unexpected twist in there",
            "your delivery felt confident",
        ]))

    highlight = random.choice(highlights)

    if tone == "supportive":
        return f"Nice, that was fun to listen to — {highlight}. Solid work on that scene."
    elif tone == "neutral":
        return f"There were some cool ideas in there — {highlight}. With a bit more focus, it could really pop."
    else:
        return f"I caught that {highlight}, but the scene felt a little rushed. Next round, don’t be afraid to commit even harder."


# -------------------------------------------------------------------
# Tools exposed to the LLM
# -------------------------------------------------------------------
@function_tool
async def start_show(
    ctx: RunContext[ImprovSessionState],
    name: Annotated[Optional[str], Field(description="Player/contestant name (optional)", default=None)] = None,
    max_rounds: Annotated[int, Field(description="Number of rounds (3-5 recommended)", default=3)] = 3,
) -> str:
    """
    Initialize the Improv Battle session and immediately launch round 1.
    """
    state = ctx.userdata

    if name:
        state.player_name = name.strip()
    else:
        # keep any previously learned name, or fall back to a generic label
        state.player_name = state.player_name or "Contestant"

    # clamp rounds to a safe range
    if max_rounds < 1:
        max_rounds = 1
    if max_rounds > 8:
        max_rounds = 8

    state.improv_state["max_rounds"] = int(max_rounds)
    state.improv_state["current_round"] = 0
    state.improv_state["rounds"] = []
    state.improv_state["phase"] = "intro"

    state.history.append({
        "time": datetime.utcnow().isoformat() + "Z",
        "action": "start_show",
        "name": state.player_name,
        "max_rounds": max_rounds,
    })

    intro = (
        f"Welcome to Improv Battle, {state.player_name}! "
        f"Tonight we’ll play {state.improv_state['max_rounds']} fast scenes. "
        "I’ll throw you a scenario, you jump into character and play it out. "
        "When you’re done, you can say something like 'End scene' or just wrap up naturally, "
        "and I’ll give you some feedback before we move on."
    )

    # Immediately kick off the first round for a smooth start
    scenario = _choose_scenario(state)
    state.improv_state["current_round"] = 1
    state.improv_state["phase"] = "awaiting_improv"

    state.history.append({
        "time": datetime.utcnow().isoformat() + "Z",
        "action": "present_scenario",
        "round": 1,
        "scenario": scenario,
    })

    return (
        intro
        + "\nRound 1 is on the board. "
        + scenario
        + "\nWhenever you’re ready, step into the character and start improvising."
    )


@function_tool
async def next_scenario(ctx: RunContext[ImprovSessionState]) -> str:
    """
    Advance to the next improv scenario, or summarize the show if we’re out of rounds.
    """
    state = ctx.userdata

    if state.improv_state.get("phase") == "done":
        return "We’ve already wrapped this run of Improv Battle. Start a new show if you’d like another round."

    current_round = state.improv_state.get("current_round", 0)
    max_rounds = state.improv_state.get("max_rounds", 3)

    # if we’ve played all configured rounds, move to summary
    if current_round >= max_rounds:
        state.improv_state["phase"] = "done"
        return await summarize_show(ctx)

    # otherwise, step into the next round
    next_round = current_round + 1
    scenario = _choose_scenario(state)
    state.improv_state["current_round"] = next_round
    state.improv_state["phase"] = "awaiting_improv"

    state.history.append({
        "time": datetime.utcnow().isoformat() + "Z",
        "action": "present_scenario",
        "round": next_round,
        "scenario": scenario,
    })

    return (
        f"Alright, Round {next_round}! Here’s your next situation: {scenario}\n"
        "Take a second to picture it, then dive in when you’re ready."
    )


@function_tool
async def record_performance(
    ctx: RunContext[ImprovSessionState],
    performance: Annotated[str, Field(description="Player's improv performance (transcribed text)")],
) -> str:
    """
    Record the player's performance for the current round, generate a reaction,
    and either invite the next scene or close out the show.
    """
    state = ctx.userdata

    if state.improv_state.get("phase") != "awaiting_improv":
        # out-of-phase call — the host can still respond, but note it in history
        state.history.append({
            "time": datetime.utcnow().isoformat() + "Z",
            "action": "record_performance_out_of_phase",
        })

    round_no = state.improv_state.get("current_round", 0)

    # try to pull the last presented scenario from history
    scenario = "(unknown scenario)"
    if state.history:
        last = state.history[-1]
        if last.get("action") == "present_scenario":
            scenario = last.get("scenario", scenario)

    reaction = _build_host_reaction(performance)

    state.improv_state["rounds"].append({
        "round": round_no,
        "scenario": scenario,
        "performance": performance,
        "reaction": reaction,
    })
    state.improv_state["phase"] = "reacting"

    state.history.append({
        "time": datetime.utcnow().isoformat() + "Z",
        "action": "record_performance",
        "round": round_no,
    })

    max_rounds = state.improv_state.get("max_rounds", 3)

    # if this was the final round, go straight into wrap-up
    if round_no >= max_rounds:
        state.improv_state["phase"] = "done"
        closing = "\n" + reaction + "\nThat was our last scene for this run. "
        closing += await summarize_show(ctx)
        return closing

    # otherwise, invite them to move on when they’re ready
    return (
        reaction
        + "\nWhen you feel ready, say something like 'Next' or 'Next round' and I’ll load up another scene."
    )


@function_tool
async def summarize_show(ctx: RunContext[ImprovSessionState]) -> str:
    """
    Produce a short recap of all played rounds and a simple style profile.
    """
    state = ctx.userdata
    rounds = state.improv_state.get("rounds", [])

    if not rounds:
        return "We didn’t actually get through a full scene this time, but thanks for dropping into Improv Battle!"

    name = state.player_name or "Contestant"
    summary_lines: List[str] = [
        f"That’s a wrap on this Improv Battle run, {name}. Here’s a quick recap of what you did:"
    ]

    # per-round recap
    for r in rounds:
        perf_snip = (r.get("performance") or "").strip()
        if len(perf_snip) > 80:
            perf_snip = perf_snip[:77] + "..."
        summary_lines.append(
            f"Round {r.get('round')}: {r.get('scenario')} — "
            f"You played: '{perf_snip}' | Host notes: {r.get('reaction')}"
        )

    # extremely lightweight profile based on the performance text
    def _contains_any(text: str, words: List[str]) -> bool:
        lower = text.lower()
        return any(w in lower for w in words)

    mentions_character = sum(
        1 for r in rounds
        if _contains_any(r.get("performance") or "", ["i am", "i'm", "as a", "character", "role"])
    )
    mentions_emotion = sum(
        1 for r in rounds
        if _contains_any(r.get("performance") or "", ["sad", "angry", "happy", "love", "cry", "tears"])
    )

    profile = "You come across as someone who "
    if mentions_character > len(rounds) / 2:
        profile += "really commits to character choices"
    elif mentions_emotion > 0:
        profile += "likes to bring emotional color into the scenes"
    else:
        profile += "enjoys surprising turns and playful twists"

    profile += ". Keep pushing for clear intentions and strong stakes in each scene."

    summary_lines.append(profile)
    summary_lines.append("Thanks for playing Improv Battle — I’d be happy to host you again anytime.")

    state.history.append({
        "time": datetime.utcnow().isoformat() + "Z",
        "action": "summarize_show",
    })

    return "\n".join(summary_lines)


@function_tool
async def stop_show(
    ctx: RunContext[ImprovSessionState],
    confirm: Annotated[bool, Field(description="Confirm stop", default=False)] = False,
) -> str:
    """
    Allow the player to end the show early, with a quick confirmation step.
    """
    state = ctx.userdata

    if not confirm:
        return "If you’re sure you want to end the show, say something like 'stop show yes' to confirm."

    state.improv_state["phase"] = "done"
    state.history.append({
        "time": datetime.utcnow().isoformat() + "Z",
        "action": "stop_show",
    })
    return "Got it, we’ll call it here. Thanks for jumping into Improv Battle!"


# -------------------------------------------------------------------
# The Agent (host persona)
# -------------------------------------------------------------------
class ImprovHostAgent(Agent):
    def __init__(self) -> None:
        system_instructions = """
            You are the lively host of a voice-first improv game show called "Improv Battle".

            Your goals:
            - Welcome the player and briefly explain how the game works.
            - Use the tools start_show, next_scenario, record_performance, summarize_show, and stop_show
            to keep track of rounds and remember what happened.
            - For each round:
                1) Make sure a scenario has been given to the player.
                2) Encourage them to act it out in character.
                3) Once they seem finished or say something like "end scene",
                call record_performance() with what they did.
                4) React with a mix of support, honest feedback, and light teasing.
            - When the configured number of rounds is done, or the player clearly wants to stop,
            move to summarize_show() or stop_show() and close the experience.

            Style:
            - High-energy, playful TV host.
            - Feedback can be supportive, neutral, or mildly critical,
            but should always stay respectful and fun.
            - Keep your spoken turns concise and easy to follow over audio.
            - Stay in character as the Improv Battle host—avoid generic assistant chatter.
            """
        super().__init__(
            instructions=system_instructions,
            tools=[start_show, next_scenario, record_performance, summarize_show, stop_show],
        )


# -------------------------------------------------------------------
# Entrypoint & prewarm
# -------------------------------------------------------------------
def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("Silero VAD prewarmed successfully.")
    except Exception as e:
        logger.warning(f"VAD prewarm failed; starting without preloaded VAD: {e}")


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("\n" + "🎭" * 6)
    logger.info("Booting Improv Battle host for room %s", ctx.room.name)

    session_state = ImprovSessionState()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-IN-samar",
            style="Conversational",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=session_state,
    )

    await session.start(
        agent=ImprovHostAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )