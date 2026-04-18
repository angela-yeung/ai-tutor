import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import ChatWindow from "../../components/ChatWindow";
import type { Message } from "../../lib/types";

const messages: Message[] = [
  { id: "1", role: "user", content: "Hello tutor" },
  { id: "2", role: "assistant", content: "Hello! How can I help?" },
];

describe("ChatWindow", () => {
  it("renders all messages", () => {
    render(<ChatWindow messages={messages} />);
    expect(screen.getByText("Hello tutor")).toBeInTheDocument();
    expect(screen.getByText("Hello! How can I help?")).toBeInTheDocument();
  });

  it("renders empty state without crashing", () => {
    const { container } = render(<ChatWindow messages={[]} />);
    expect(container.firstChild).toBeInTheDocument();
  });

  it("renders the correct number of message bubbles", () => {
    render(<ChatWindow messages={messages} />);
    // Each MessageBubble renders a div with text-lg
    const bubbles = screen.getAllByText(/Hello/);
    expect(bubbles).toHaveLength(2);
  });
});
