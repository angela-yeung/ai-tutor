import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import MessageBubble from "../../components/MessageBubble";
import type { Message } from "../../lib/types";

const userMessage: Message = {
  id: "1",
  role: "user",
  content: "What is 2 + 2?",
};

const assistantMessage: Message = {
  id: "2",
  role: "assistant",
  content: "2 + 2 equals 4!",
};

const streamingMessage: Message = {
  id: "3",
  role: "assistant",
  content: "Thinking...",
  streaming: true,
};

describe("MessageBubble", () => {
  it("renders user message content", () => {
    render(<MessageBubble message={userMessage} />);
    expect(screen.getByText("What is 2 + 2?")).toBeInTheDocument();
  });

  it("renders assistant message content", () => {
    render(<MessageBubble message={assistantMessage} />);
    expect(screen.getByText("2 + 2 equals 4!")).toBeInTheDocument();
  });

  it("user bubble has blue-100 background class", () => {
    const { container } = render(<MessageBubble message={userMessage} />);
    expect(container.firstChild).toHaveClass("bg-blue-100");
  });

  it("assistant bubble has white background class", () => {
    const { container } = render(<MessageBubble message={assistantMessage} />);
    expect(container.firstChild).toHaveClass("bg-white");
  });

  it("user bubble is right-aligned", () => {
    const { container } = render(<MessageBubble message={userMessage} />);
    expect(container.firstChild).toHaveClass("self-end");
  });

  it("assistant bubble is left-aligned", () => {
    const { container } = render(<MessageBubble message={assistantMessage} />);
    expect(container.firstChild).toHaveClass("self-start");
  });

  it("streaming message shows a blinking cursor", () => {
    const { container } = render(<MessageBubble message={streamingMessage} />);
    // Cursor is a span with animate-pulse
    const cursor = container.querySelector(".animate-pulse");
    expect(cursor).toBeInTheDocument();
  });

  it("non-streaming message has no blinking cursor", () => {
    const { container } = render(<MessageBubble message={assistantMessage} />);
    expect(container.querySelector(".animate-pulse")).not.toBeInTheDocument();
  });
});
