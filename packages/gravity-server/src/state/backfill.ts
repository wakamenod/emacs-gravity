// backfill.ts — Rebuild a session's turn tree from its Claude Code transcript.
//
// Session state lives only in this process's memory. A server restart — a
// plugin auto-update, `make restart-server`, a reboot — therefore erases
// every session it knew about, and `claude --resume` then reattaches to a
// conversation gravity cannot render even though Claude Code itself replays
// it from disk. This module closes that gap by reading the same file.
//
// The replay deliberately goes through the ordinary mutation API
// (`addPrompt` / `addTool` / `completeTool` / `closeTurn`) rather than
// assembling a Session literal. Step boundaries, the tool index, per-turn
// counts and the emitted patches are all non-trivial invariants; routing
// backfilled state through the same code as live state is what keeps the
// two indistinguishable to a terminal.

import type { ParsedTranscript, ParsedTurn, Patch, Session, Tool } from "@gravity/shared";
import { addPrompt, addTool, closeTurn, completeTool, updateMeta } from "./session.js";

/** Claude Code model ids are long; terminals show the family name. */
function shortModelName(modelId: string): string {
  const m = modelId.toLowerCase();
  if (m.includes("opus")) return "opus";
  if (m.includes("sonnet")) return "sonnet";
  if (m.includes("haiku")) return "haiku";
  if (m.includes("fable")) return "fable";
  return modelId;
}

function toolFrom(
  parsed: ParsedTurn["steps"][number]["tools"][number],
  turnNumber: number,
): Tool {
  return {
    toolUseId: parsed.toolUseId,
    name: parsed.name,
    input: parsed.input,
    // addTool records the call; completeTool below applies the outcome, so
    // start every tool in the state the live PreToolUse path starts it in.
    status: "running",
    result: null,
    partial: null,
    timestamp: parsed.timestamp,
    duration: null,
    turn: turnNumber,
    assistantText: parsed.assistantText,
    assistantThinking: parsed.assistantThinking,
    // The transcript records what followed a tool as the next step's prose,
    // so there is nothing left to attribute as post-tool context.
    postText: null,
    postThinking: null,
    parentAgentId: null,
    ambiguous: false,
    candidateAgentIds: null,
    agentId: null,
  };
}

export interface BackfillResult {
  patches: Patch[];
  turnsRestored: number;
  toolsRestored: number;
}

/**
 * Replay PARSED into SESSION. Intended for a session with no history yet;
 * `addTool` dedups by tool_use_id, so a partial replay over existing state
 * is safe but will not reorder what is already there.
 *
 * Returns the patches produced. Callers that have already sent a snapshot
 * should re-send one instead of forwarding these — the patch stream assumes
 * a terminal is following along from the same starting point.
 */
export function backfillSession(session: Session, parsed: ParsedTranscript): BackfillResult {
  const patches: Patch[] = [];
  let turnsRestored = 0;
  let toolsRestored = 0;

  if (parsed.branch || parsed.model) {
    patches.push(
      ...updateMeta(session, {
        branch: parsed.branch ?? undefined,
        modelName: parsed.model ? shortModelName(parsed.model) : undefined,
      }),
    );
  }

  for (const [index, turn] of parsed.turns.entries()) {
    // A leading promptless turn is pre-prompt activity, which belongs on the
    // turn 0 every session already starts with — opening another would push
    // an empty turn ahead of the conversation.
    if (turn.prompt) {
      patches.push(
        ...addPrompt(session, {
          type: "user",
          text: turn.prompt.text,
          submitted: turn.prompt.submitted,
          elapsed: null,
          toolUseId: null,
          answer: null,
        }),
      );
      turnsRestored++;
    } else if (index > 0) {
      // A promptless turn anywhere else cannot be reconstructed as a
      // boundary; fold its steps into whatever turn is current.
    }

    const turnNumber = session.currentTurn;

    for (const step of turn.steps) {
      for (const parsedTool of step.tools) {
        const tool = toolFrom(parsedTool, turnNumber);
        const added = addTool(session, tool);
        if (added.length === 0) continue; // duplicate tool_use_id
        patches.push(...added);
        toolsRestored++;

        if (parsedTool.status !== "running") {
          patches.push(
            ...completeTool(
              session,
              parsedTool.toolUseId,
              parsedTool.result,
              parsedTool.status,
              undefined,
              undefined,
              parsedTool.duration ?? undefined,
            ),
          );
        }
      }
    }

    // Stamp elapsed from the transcript's own clock. closeTurn would
    // otherwise derive it from `Date.now()`, which for a conversation
    // resumed the next day reads as a turn that took twenty hours.
    const current = session.turns[session.turns.length - 1];
    if (current?.prompt && turn.lastEventTime != null) {
      current.prompt.elapsed = Math.max(0, (turn.lastEventTime - current.prompt.submitted) / 1000);
    }

    // Only a turn the model actually finished gets closed. The final turn of
    // a transcript being resumed mid-flight has no trailing prose, and
    // freezing it would misreport it as complete.
    if (turn.stopText || turn.stopThinking) {
      patches.push(
        ...closeTurn(session, {
          stopText: turn.stopText ?? undefined,
          stopThinking: turn.stopThinking ?? undefined,
        }),
      );
    }
  }

  const lastEvent = parsed.turns[parsed.turns.length - 1]?.lastEventTime;
  if (lastEvent != null) session.lastEventTime = lastEvent;
  const firstEvent = parsed.turns[0]?.prompt?.submitted;
  if (firstEvent != null) session.startTime = firstEvent;

  return { patches, turnsRestored, toolsRestored };
}
