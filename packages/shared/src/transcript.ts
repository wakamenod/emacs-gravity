// transcript.ts — Parse a Claude Code transcript (.jsonl) into a neutral
// intermediate representation of the turn tree.
//
// Why this exists: gravity-server holds session state in memory only. When
// the server restarts, everything it knew about a session is gone, so
// `claude --resume` reattaches to a conversation the server cannot render.
// Claude Code itself has no such gap — it reads the transcript back. This
// module gives the server the same ability.
//
// The output is deliberately NOT a `Session`: this package cannot import
// gravity-server's state mutators, and keeping the IR neutral makes the
// parse independently testable. `state/backfill.ts` on the server side
// replays the IR through the normal mutation API so backfilled state goes
// through exactly the same invariants (step boundaries, tool index, counts)
// as live state.
//
// ── Transcript shape (verified against Claude Code 2.1.x) ──────────────
//
// Newline-delimited JSON, one record per line, in wall-clock order.
//
//   {"type":"user","message":{"content":"..."},"isMeta":false,...}
//     A typed user prompt when `content` is a string and `isMeta` is unset.
//     `isMeta:true` marks injected system text (skill instructions,
//     "Continue from where you left off") — not a turn boundary.
//
//   {"type":"user","message":{"content":[{"type":"tool_result",...}]},
//    "toolUseResult":{...}}
//     Tool completion. `toolUseResult` is the structured payload the live
//     path receives as PostToolUse's `tool_response`; the `tool_result`
//     block is the model-facing rendering of the same thing.
//
//   {"type":"assistant","message":{"id":"msg_…","content":[<one block>]}}
//     ONE content block per line — thinking, text, or tool_use. Consecutive
//     lines sharing `message.id` are one logical assistant message.
//
//   {"isSidechain":true,...}
//     Subagent traffic. Skipped here; agent backfill is future work.
//
// Every other `type` (attachment, system, file-history-snapshot, …) is
// bookkeeping and ignored.

/** A tool call recovered from the transcript. */
export interface ParsedTool {
  toolUseId: string;
  name: string;
  input: Record<string, unknown>;
  /** "running" when the transcript ends before the tool_result arrives. */
  status: "running" | "done" | "error";
  result: unknown;
  /** Epoch ms of the assistant message that issued the call. */
  timestamp: number;
  /** Seconds between the call and its result; null while running. */
  duration: number | null;
  assistantText: string | null;
  assistantThinking: string | null;
}

/** One response step: an assistant message's prose plus the tools it called. */
export interface ParsedStep {
  thinking: string | null;
  text: string | null;
  tools: ParsedTool[];
}

/** One turn: a user prompt and everything the assistant did in response. */
export interface ParsedTurn {
  prompt: { text: string; submitted: number } | null;
  steps: ParsedStep[];
  /** Trailing assistant prose with no tool call after it. */
  stopText: string | null;
  stopThinking: string | null;
  /** Epoch ms of the last record belonging to this turn. */
  lastEventTime: number | null;
}

export interface ParsedTranscript {
  sessionId: string | null;
  cwd: string | null;
  branch: string | null;
  /** Model id from the most recent assistant message, e.g. "claude-opus-5". */
  model: string | null;
  /**
   * Turn 0 carries any activity that precedes the first user prompt, mirroring
   * the live turn tree. Later entries are one per typed prompt.
   */
  turns: ParsedTurn[];
  /** True when `maxTurns` dropped older turns. */
  truncated: boolean;
  /** Turns dropped from the front by `maxTurns`. */
  droppedTurns: number;
  /** Records that failed to parse as JSON; a healthy transcript yields 0. */
  malformedLines: number;
}

export interface ParseTranscriptOptions {
  /**
   * Keep at most this many of the most recent turns (turn 0 always counts).
   * Older turns are dropped and `truncated` is set. Omit for no limit.
   */
  maxTurns?: number;
}

// ── Record shapes (structurally validated, never trusted) ─────────────

interface ContentBlock {
  type?: string;
  text?: string;
  thinking?: string;
  // tool_use
  id?: string;
  name?: string;
  input?: unknown;
  // tool_result
  tool_use_id?: string;
  is_error?: boolean;
  content?: unknown;
}

interface TranscriptRecord {
  type?: string;
  isMeta?: boolean;
  isSidechain?: boolean;
  timestamp?: string;
  cwd?: string;
  gitBranch?: string;
  sessionId?: string;
  toolUseResult?: unknown;
  message?: {
    id?: string;
    model?: string;
    content?: unknown;
  };
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function blocksOf(message: TranscriptRecord["message"]): ContentBlock[] {
  const content = message?.content;
  if (Array.isArray(content)) return content as ContentBlock[];
  return [];
}

function epochOf(record: TranscriptRecord): number | null {
  if (typeof record.timestamp !== "string") return null;
  const ms = Date.parse(record.timestamp);
  return Number.isNaN(ms) ? null : ms;
}

function joined(parts: string[]): string | null {
  const text = parts.join("\n\n").trim();
  return text.length > 0 ? text : null;
}

/**
 * A user record is a turn boundary when a human typed it. Injected system
 * text (`isMeta`) reuses the same record type and must not open a turn.
 */
function promptTextOf(record: TranscriptRecord): string | null {
  if (record.isMeta === true) return null;
  const content = record.message?.content;
  if (typeof content === "string") {
    return content.trim().length > 0 ? content : null;
  }
  const blocks = blocksOf(record.message);
  if (blocks.length === 0) return null;
  // An array-shaped prompt is only a prompt when no block is a tool result.
  if (blocks.some((b) => b.type === "tool_result")) return null;
  const texts = blocks
    .filter((b) => b.type === "text" && typeof b.text === "string")
    .map((b) => b.text as string);
  return joined(texts);
}

/** Accumulates the blocks of one logical assistant message (one `message.id`). */
interface PendingMessage {
  id: string;
  thinking: string[];
  text: string[];
  tools: ParsedTool[];
}

class TurnBuilder {
  private turns: ParsedTurn[] = [];
  /** Prose seen since the last tool call — becomes the next step's context,
   * or the turn's trailing text if no tool follows. */
  private carriedThinking: string[] = [];
  private carriedText: string[] = [];

  constructor() {
    this.turns.push(TurnBuilder.emptyTurn());
  }

  private static emptyTurn(): ParsedTurn {
    return { prompt: null, steps: [], stopText: null, stopThinking: null, lastEventTime: null };
  }

  private current(): ParsedTurn {
    return this.turns[this.turns.length - 1]!;
  }

  touch(at: number | null): void {
    if (at != null) this.current().lastEventTime = at;
  }

  /** Flush carried prose onto the turn as its trailing text. */
  private settle(): void {
    const turn = this.current();
    const thinking = joined(this.carriedThinking);
    const text = joined(this.carriedText);
    if (thinking) turn.stopThinking = thinking;
    if (text) turn.stopText = text;
    this.carriedThinking = [];
    this.carriedText = [];
  }

  openTurn(text: string, submitted: number | null): void {
    this.settle();
    // Turn 0 exists to hold pre-prompt activity. When the transcript opens
    // with a prompt (the common case) it is still empty, so reuse it rather
    // than emitting a vacant turn ahead of the conversation.
    const turn = this.current();
    const reuseTurn0 =
      this.turns.length === 1 && turn.prompt === null && turn.steps.length === 0 &&
      turn.stopText === null && turn.stopThinking === null;
    const target = reuseTurn0 ? turn : (this.turns.push(TurnBuilder.emptyTurn()), this.current());
    target.prompt = { text, submitted: submitted ?? Date.now() };
    if (submitted != null) target.lastEventTime = submitted;
  }

  addMessage(pending: PendingMessage): void {
    this.carriedThinking.push(...pending.thinking);
    this.carriedText.push(...pending.text);
    if (pending.tools.length === 0) return;

    // The prose accumulated up to this point is this step's context — the
    // same attribution the live path gets from the bridge's
    // `assistant_text` / `assistant_thinking`.
    const thinking = joined(this.carriedThinking);
    const text = joined(this.carriedText);
    this.carriedThinking = [];
    this.carriedText = [];

    for (const tool of pending.tools) {
      tool.assistantThinking = thinking;
      tool.assistantText = text;
    }
    this.current().steps.push({ thinking, text, tools: pending.tools });
  }

  finish(): ParsedTurn[] {
    this.settle();
    return this.turns;
  }
}

/**
 * Parse transcript NDJSON into the turn IR.
 *
 * Pure and total: malformed lines are counted, never thrown. An empty or
 * entirely unparseable input yields a single empty turn 0.
 */
export function parseTranscript(
  ndjson: string,
  opts: ParseTranscriptOptions = {},
): ParsedTranscript {
  const builder = new TurnBuilder();
  const toolsById = new Map<string, ParsedTool>();

  let sessionId: string | null = null;
  let cwd: string | null = null;
  let branch: string | null = null;
  let model: string | null = null;
  let malformedLines = 0;
  let pending: PendingMessage | null = null;

  const flushPending = (): void => {
    if (pending) {
      builder.addMessage(pending);
      pending = null;
    }
  };

  for (const line of ndjson.split("\n")) {
    if (line.trim().length === 0) continue;

    let record: TranscriptRecord;
    try {
      record = JSON.parse(line) as TranscriptRecord;
    } catch {
      malformedLines++;
      continue;
    }
    // Subagent traffic lives in the same file under `isSidechain`. Rendering
    // it as root-level turns would be wrong, and attributing it to the
    // spawning Task tool needs agent reconstruction — future work.
    if (record.isSidechain === true) continue;

    if (typeof record.sessionId === "string" && !sessionId) sessionId = record.sessionId;
    if (typeof record.cwd === "string") cwd = record.cwd;
    if (typeof record.gitBranch === "string" && record.gitBranch.length > 0) {
      branch = record.gitBranch;
    }

    const at = epochOf(record);

    if (record.type === "user") {
      const prompt = promptTextOf(record);
      if (prompt !== null) {
        flushPending();
        builder.openTurn(prompt, at);
        continue;
      }
      // Tool results close out calls issued by an earlier assistant message.
      const blocks = blocksOf(record.message).filter((b) => b.type === "tool_result");
      if (blocks.length === 0) continue;
      flushPending();
      for (const block of blocks) {
        const id = typeof block.tool_use_id === "string" ? block.tool_use_id : null;
        if (!id) continue;
        const tool = toolsById.get(id);
        if (!tool) continue;
        // `toolUseResult` is the structured payload; the block's `content` is
        // the model-facing rendering. Prefer the former to match what the
        // live PostToolUse path stores, and fall back when it is absent.
        tool.result = record.toolUseResult !== undefined ? record.toolUseResult : block.content;
        tool.status = block.is_error === true ? "error" : "done";
        if (at != null) tool.duration = Math.max(0, (at - tool.timestamp) / 1000);
      }
      builder.touch(at);
      continue;
    }

    if (record.type === "assistant") {
      if (typeof record.message?.model === "string") model = record.message.model;
      const id = typeof record.message?.id === "string" ? record.message.id : "";
      if (pending && pending.id !== id) flushPending();
      if (!pending) pending = { id, thinking: [], text: [], tools: [] };

      for (const block of blocksOf(record.message)) {
        if (block.type === "thinking" && typeof block.thinking === "string") {
          if (block.thinking.trim().length > 0) pending.thinking.push(block.thinking);
        } else if (block.type === "text" && typeof block.text === "string") {
          if (block.text.trim().length > 0) pending.text.push(block.text);
        } else if (block.type === "tool_use" && typeof block.id === "string") {
          const tool: ParsedTool = {
            toolUseId: block.id,
            name: typeof block.name === "string" ? block.name : "Unknown",
            input: asRecord(block.input),
            status: "running",
            result: null,
            timestamp: at ?? Date.now(),
            duration: null,
            assistantText: null,
            assistantThinking: null,
          };
          pending.tools.push(tool);
          toolsById.set(tool.toolUseId, tool);
        }
      }
      builder.touch(at);
      continue;
    }
  }

  flushPending();
  let turns = builder.finish();

  // Drop leading turn 0 when nothing happened before the first prompt.
  if (
    turns.length > 1 && turns[0]!.prompt === null && turns[0]!.steps.length === 0 &&
    turns[0]!.stopText === null && turns[0]!.stopThinking === null
  ) {
    turns = turns.slice(1);
  }

  let truncated = false;
  let droppedTurns = 0;
  if (opts.maxTurns != null && opts.maxTurns > 0 && turns.length > opts.maxTurns) {
    droppedTurns = turns.length - opts.maxTurns;
    turns = turns.slice(droppedTurns);
    truncated = true;
  }

  return { sessionId, cwd, branch, model, turns, truncated, droppedTurns, malformedLines };
}
