import { describe, it, expect, vi } from "vitest";
import { streamChat } from "../../lib/chat";
import type { DoneMeta } from "../../lib/types";

function makeStream(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
}

function mockFetch(chunks: string[]) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({ ok: true, body: makeStream(chunks) })
  );
}

describe("streamChat — token events", () => {
  it("calls onToken for each token event", async () => {
    mockFetch([
      `event: token\ndata: {"chunk":"Hello"}\n\n`,
      `event: token\ndata: {"chunk":" world"}\n\n`,
      `event: done\ndata: {"session_paused":false,"concept":"math","concepts_needing_review":[],"conversation_history":[]}\n\n`,
    ]);
    const tokens: string[] = [];
    await streamChat("thread-1", "hi", (c) => tokens.push(c), vi.fn(), vi.fn());
    expect(tokens).toEqual(["Hello", " world"]);
  });
});

describe("streamChat — done event", () => {
  it("calls onDone with the done payload", async () => {
    const doneMeta: DoneMeta = {
      session_paused: true,
      concept: "fractions",
      concepts_needing_review: ["fractions"],
      conversation_history: [],
    };
    mockFetch([
      `event: done\ndata: ${JSON.stringify(doneMeta)}\n\n`,
    ]);
    const onDone = vi.fn();
    await streamChat("thread-1", "hi", vi.fn(), onDone, vi.fn());
    expect(onDone).toHaveBeenCalledWith(doneMeta);
  });
});

describe("streamChat — error event", () => {
  it("calls onError with the error message", async () => {
    mockFetch([
      `event: error\ndata: {"message":"Something went wrong."}\n\n`,
    ]);
    const onError = vi.fn();
    await streamChat("thread-1", "hi", vi.fn(), vi.fn(), onError);
    expect(onError).toHaveBeenCalledWith("Something went wrong.");
  });
});

describe("streamChat — chunked SSE delivery", () => {
  it("handles an SSE event split across multiple chunks", async () => {
    // Simulate network delivering partial data
    mockFetch([
      `event: token\ndata: {"ch`,
      `unk":"Hi"}\n\n`,
      `event: done\ndata: {"session_paused":false,"concept":"","concepts_needing_review":[],"conversation_history":[]}\n\n`,
    ]);
    const tokens: string[] = [];
    await streamChat("thread-1", "hi", (c) => tokens.push(c), vi.fn(), vi.fn());
    expect(tokens).toEqual(["Hi"]);
  });
});

describe("streamChat — network error", () => {
  it("calls onError when fetch throws", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("Network down")));
    const onError = vi.fn();
    await streamChat("thread-1", "hi", vi.fn(), vi.fn(), onError);
    expect(onError).toHaveBeenCalledWith("Something went wrong. Please try again.");
  });
});
