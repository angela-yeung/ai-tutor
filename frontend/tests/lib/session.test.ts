import { describe, it, expect, beforeEach, vi } from "vitest";

// localStorage is available in jsdom; reset between tests
beforeEach(() => {
  if (typeof window !== "undefined" && window.localStorage) {
    const keys = Object.keys(window.localStorage);
    for (const key of keys) {
      window.localStorage.removeItem(key);
    }
  }
});

// Delay import so localStorage is fresh each test
async function importSession() {
  vi.resetModules();
  return import("../../lib/session");
}

describe("getThreadId", () => {
  it("generates a UUID on first call", async () => {
    const { getThreadId } = await importSession();
    const id = getThreadId();
    expect(id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
    );
  });

  it("returns the same UUID on second call", async () => {
    const { getThreadId } = await importSession();
    const first = getThreadId();
    const second = getThreadId();
    expect(first).toBe(second);
  });

  it("persists across re-imports (same localStorage)", async () => {
    const { getThreadId: get1 } = await importSession();
    const id = get1();
    const { getThreadId: get2 } = await importSession();
    expect(get2()).toBe(id);
  });
});

describe("newSession", () => {
  it("returns a new UUID different from the previous one", async () => {
    const { getThreadId, newSession } = await importSession();
    const first = getThreadId();
    const second = newSession();
    expect(second).not.toBe(first);
  });

  it("stores the new UUID so subsequent getThreadId returns it", async () => {
    const { getThreadId, newSession } = await importSession();
    const newId = newSession();
    expect(getThreadId()).toBe(newId);
  });
});
