import type { DoneMeta } from "./types";

export async function streamChat(
  threadId: string,
  message: string,
  onToken: (chunk: string) => void,
  onDone: (meta: DoneMeta) => void,
  onError: (msg: string) => void
): Promise<void> {
  try {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL;
    if (!apiUrl) {
      onError("Something went wrong. Please try again.");
      return;
    }

    const res = await fetch(`${apiUrl}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: threadId, message }),
    });

    if (!res.ok || !res.body) {
      onError("Something went wrong. Please try again.");
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop()!;
      for (const part of parts) {
        const eventLine = part.match(/^event: (\w+)/m)?.[1];
        const dataLine = part.match(/^data: (.+)/m)?.[1];
        if (!eventLine || !dataLine) continue;
        let payload: Record<string, unknown>;
        try {
          payload = JSON.parse(dataLine);
        } catch {
          continue;
        }
        if (eventLine === "token") onToken(payload.chunk as string);
        else if (eventLine === "done") onDone(payload as DoneMeta);
        else if (eventLine === "error") { onError(payload.message as string); return; }
      }
    }
  } catch {
    onError("Something went wrong. Please try again.");
  }
}
