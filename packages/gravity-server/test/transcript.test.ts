import { describe, it, expect } from "vitest";
import { parseTranscript } from "@gravity/shared";

// ── Fixture builders ─────────────────────────────────────────────────
//
// Claude Code writes ONE content block per assistant line; consecutive
// lines sharing `message.id` form one logical message. The builders below
// mirror that so the fixtures stay faithful to the real file shape.

const T0 = Date.parse("2026-09-03T00:00:00.000Z");
const at = (offsetSeconds: number) => new Date(T0 + offsetSeconds * 1000).toISOString();

const line = (record: unknown) => JSON.stringify(record);

const userPrompt = (text: string, offset: number) =>
  line({
    type: "user",
    message: { role: "user", content: text },
    promptSource: "typed",
    origin: { kind: "human" },
    isSidechain: false,
    cwd: "/proj",
    gitBranch: "master",
    sessionId: "sess-1",
    timestamp: at(offset),
  });

const metaUser = (text: string, offset: number) =>
  line({
    type: "user",
    message: { role: "user", content: [{ type: "text", text }] },
    isMeta: true,
    isSidechain: false,
    timestamp: at(offset),
  });

const assistantBlock = (id: string, block: unknown, offset: number, extra: object = {}) =>
  line({
    type: "assistant",
    message: { id, model: "claude-opus-5", role: "assistant", content: [block] },
    isSidechain: false,
    timestamp: at(offset),
    ...extra,
  });

const thinking = (id: string, text: string, offset: number) =>
  assistantBlock(id, { type: "thinking", thinking: text }, offset);

const say = (id: string, text: string, offset: number) =>
  assistantBlock(id, { type: "text", text }, offset);

const toolUse = (id: string, toolUseId: string, name: string, input: object, offset: number) =>
  assistantBlock(id, { type: "tool_use", id: toolUseId, name, input }, offset);

const toolResult = (
  toolUseId: string,
  offset: number,
  opts: { result?: unknown; isError?: boolean } = {},
) =>
  line({
    type: "user",
    message: {
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: toolUseId,
          is_error: opts.isError === true,
          content: "rendered for the model",
        },
      ],
    },
    toolUseResult: opts.result ?? { stdout: "ok", stderr: "" },
    isSidechain: false,
    timestamp: at(offset),
  });

describe("parseTranscript", () => {
  it("builds one turn per typed prompt", () => {
    const t = parseTranscript(
      [
        userPrompt("first question", 0),
        say("msg_1", "first answer", 1),
        userPrompt("second question", 10),
        say("msg_2", "second answer", 11),
      ].join("\n"),
    );

    expect(t.turns).toHaveLength(2);
    expect(t.turns[0]!.prompt?.text).toBe("first question");
    expect(t.turns[0]!.stopText).toBe("first answer");
    expect(t.turns[1]!.prompt?.text).toBe("second question");
    expect(t.turns[1]!.stopText).toBe("second answer");
    expect(t.malformedLines).toBe(0);
  });

  it("carries session metadata off the records", () => {
    const t = parseTranscript([userPrompt("hi", 0), say("msg_1", "hello", 1)].join("\n"));

    expect(t.sessionId).toBe("sess-1");
    expect(t.cwd).toBe("/proj");
    expect(t.branch).toBe("master");
    expect(t.model).toBe("claude-opus-5");
  });

  it("groups consecutive assistant lines sharing a message id into one step", () => {
    const t = parseTranscript(
      [
        userPrompt("do it", 0),
        thinking("msg_1", "I should read the file", 1),
        say("msg_1", "Reading the file now.", 2),
        toolUse("msg_1", "tu_1", "Read", { file_path: "/a.ts" }, 3),
        toolUse("msg_1", "tu_2", "Read", { file_path: "/b.ts" }, 3),
        toolResult("tu_1", 4),
        toolResult("tu_2", 5),
      ].join("\n"),
    );

    expect(t.turns).toHaveLength(1);
    const steps = t.turns[0]!.steps;
    expect(steps).toHaveLength(1);
    expect(steps[0]!.thinking).toBe("I should read the file");
    expect(steps[0]!.text).toBe("Reading the file now.");
    expect(steps[0]!.tools.map((x) => x.toolUseId)).toEqual(["tu_1", "tu_2"]);
    expect(steps[0]!.tools[0]!.name).toBe("Read");
    expect(steps[0]!.tools[0]!.input).toEqual({ file_path: "/a.ts" });
  });

  it("starts a new step for each logical assistant message", () => {
    const t = parseTranscript(
      [
        userPrompt("do it", 0),
        toolUse("msg_1", "tu_1", "Read", {}, 1),
        toolResult("tu_1", 2),
        say("msg_2", "Now editing.", 3),
        toolUse("msg_2", "tu_2", "Edit", {}, 4),
        toolResult("tu_2", 5),
      ].join("\n"),
    );

    const steps = t.turns[0]!.steps;
    expect(steps).toHaveLength(2);
    expect(steps[0]!.text).toBeNull();
    expect(steps[1]!.text).toBe("Now editing.");
  });

  it("attributes prose from a text-only message to the step that follows", () => {
    const t = parseTranscript(
      [
        userPrompt("do it", 0),
        say("msg_1", "Let me look around.", 1),
        toolUse("msg_2", "tu_1", "Glob", {}, 2),
        toolResult("tu_1", 3),
      ].join("\n"),
    );

    const steps = t.turns[0]!.steps;
    expect(steps).toHaveLength(1);
    expect(steps[0]!.text).toBe("Let me look around.");
    expect(steps[0]!.tools[0]!.assistantText).toBe("Let me look around.");
    // Consumed by the step, so it is not also the turn's trailing text.
    expect(t.turns[0]!.stopText).toBeNull();
  });

  it("completes tools from tool_result, preferring toolUseResult", () => {
    const t = parseTranscript(
      [
        userPrompt("run it", 0),
        toolUse("msg_1", "tu_1", "Bash", { command: "ls" }, 1),
        toolResult("tu_1", 4, { result: { stdout: "a\nb", stderr: "" } }),
      ].join("\n"),
    );

    const tool = t.turns[0]!.steps[0]!.tools[0]!;
    expect(tool.status).toBe("done");
    expect(tool.result).toEqual({ stdout: "a\nb", stderr: "" });
    expect(tool.duration).toBe(3);
  });

  it("marks a failed tool_result as an error", () => {
    const t = parseTranscript(
      [
        userPrompt("run it", 0),
        toolUse("msg_1", "tu_1", "Bash", { command: "false" }, 1),
        toolResult("tu_1", 2, { isError: true }),
      ].join("\n"),
    );

    expect(t.turns[0]!.steps[0]!.tools[0]!.status).toBe("error");
  });

  it("leaves a tool running when the transcript ends before its result", () => {
    const t = parseTranscript(
      [userPrompt("run it", 0), toolUse("msg_1", "tu_1", "Bash", {}, 1)].join("\n"),
    );

    const tool = t.turns[0]!.steps[0]!.tools[0]!;
    expect(tool.status).toBe("running");
    expect(tool.result).toBeNull();
    expect(tool.duration).toBeNull();
  });

  it("keeps trailing prose as the turn's stop text", () => {
    const t = parseTranscript(
      [
        userPrompt("do it", 0),
        toolUse("msg_1", "tu_1", "Read", {}, 1),
        toolResult("tu_1", 2),
        thinking("msg_2", "That covers it.", 3),
        say("msg_2", "Done — the file was already correct.", 4),
      ].join("\n"),
    );

    expect(t.turns[0]!.stopThinking).toBe("That covers it.");
    expect(t.turns[0]!.stopText).toBe("Done — the file was already correct.");
    expect(t.turns[0]!.steps).toHaveLength(1);
  });

  it("ignores injected system text so it does not open a turn", () => {
    const t = parseTranscript(
      [
        userPrompt("real question", 0),
        metaUser("Continue from where you left off.", 1),
        say("msg_1", "answer", 2),
      ].join("\n"),
    );

    expect(t.turns).toHaveLength(1);
    expect(t.turns[0]!.prompt?.text).toBe("real question");
  });

  it("skips subagent sidechain records", () => {
    const sidechain = line({
      type: "assistant",
      message: { id: "msg_side", content: [{ type: "tool_use", id: "tu_side", name: "Grep", input: {} }] },
      isSidechain: true,
      timestamp: at(2),
    });
    const t = parseTranscript(
      [
        userPrompt("do it", 0),
        toolUse("msg_1", "tu_1", "Task", {}, 1),
        sidechain,
        toolResult("tu_1", 3),
      ].join("\n"),
    );

    const ids = t.turns[0]!.steps.flatMap((s) => s.tools.map((x) => x.toolUseId));
    expect(ids).toEqual(["tu_1"]);
  });

  it("records activity that precedes the first prompt as turn 0", () => {
    const t = parseTranscript(
      [
        toolUse("msg_0", "tu_0", "Read", {}, 0),
        toolResult("tu_0", 1),
        userPrompt("now do it", 2),
        say("msg_1", "ok", 3),
      ].join("\n"),
    );

    expect(t.turns).toHaveLength(2);
    expect(t.turns[0]!.prompt).toBeNull();
    expect(t.turns[0]!.steps[0]!.tools[0]!.toolUseId).toBe("tu_0");
    expect(t.turns[1]!.prompt?.text).toBe("now do it");
  });

  it("drops the empty turn 0 when the transcript opens with a prompt", () => {
    const t = parseTranscript([userPrompt("first", 0), say("msg_1", "hi", 1)].join("\n"));
    expect(t.turns).toHaveLength(1);
    expect(t.turns[0]!.prompt?.text).toBe("first");
  });

  it("keeps only the newest turns under maxTurns", () => {
    const lines: string[] = [];
    for (let i = 0; i < 5; i++) {
      lines.push(userPrompt(`prompt ${i}`, i * 10), say(`msg_${i}`, `answer ${i}`, i * 10 + 1));
    }
    const t = parseTranscript(lines.join("\n"), { maxTurns: 2 });

    expect(t.turns).toHaveLength(2);
    expect(t.truncated).toBe(true);
    expect(t.droppedTurns).toBe(3);
    expect(t.turns[0]!.prompt?.text).toBe("prompt 3");
    expect(t.turns[1]!.prompt?.text).toBe("prompt 4");
  });

  it("counts malformed lines instead of throwing", () => {
    const t = parseTranscript(
      [userPrompt("hi", 0), "{not json", "", say("msg_1", "hello", 1)].join("\n"),
    );

    expect(t.malformedLines).toBe(1);
    expect(t.turns).toHaveLength(1);
    expect(t.turns[0]!.stopText).toBe("hello");
  });

  it("returns a single empty turn for empty input", () => {
    const t = parseTranscript("");
    expect(t.turns).toHaveLength(1);
    expect(t.turns[0]!.prompt).toBeNull();
    expect(t.turns[0]!.steps).toHaveLength(0);
  });

  it("ignores bookkeeping record types", () => {
    const t = parseTranscript(
      [
        line({ type: "attachment", timestamp: at(0) }),
        line({ type: "system", timestamp: at(0) }),
        line({ type: "file-history-snapshot", timestamp: at(0) }),
        userPrompt("hi", 1),
        say("msg_1", "hello", 2),
      ].join("\n"),
    );

    expect(t.turns).toHaveLength(1);
    expect(t.turns[0]!.prompt?.text).toBe("hi");
  });
});
