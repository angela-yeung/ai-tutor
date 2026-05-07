import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import InputBar from "../../components/InputBar";

describe("InputBar", () => {
  it("renders an input and a send button", () => {
    render(<InputBar onSend={vi.fn()} isStreaming={false} isPaused={false} />);
    expect(screen.getByPlaceholderText(/ask pip/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /send/i })).toBeInTheDocument();
  });

  it("calls onSend with the typed message on button click", async () => {
    const onSend = vi.fn();
    render(<InputBar onSend={onSend} isStreaming={false} isPaused={false} />);
    await userEvent.type(screen.getByRole("textbox"), "What is 3 + 3?");
    await userEvent.click(screen.getByRole("button", { name: /send/i }));
    expect(onSend).toHaveBeenCalledWith("What is 3 + 3?");
  });

  it("clears the input after sending", async () => {
    render(<InputBar onSend={vi.fn()} isStreaming={false} isPaused={false} />);
    const input = screen.getByRole("textbox");
    await userEvent.type(input, "Hello");
    await userEvent.click(screen.getByRole("button", { name: /send/i }));
    expect(input).toHaveValue("");
  });

  it("calls onSend on Enter key", async () => {
    const onSend = vi.fn();
    render(<InputBar onSend={onSend} isStreaming={false} isPaused={false} />);
    await userEvent.type(screen.getByRole("textbox"), "Hello{Enter}");
    expect(onSend).toHaveBeenCalledWith("Hello");
  });

  it("disables input and button when isStreaming=true", () => {
    render(<InputBar onSend={vi.fn()} isStreaming={true} isPaused={false} />);
    expect(screen.getByRole("textbox")).toBeDisabled();
    expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
  });

  it("disables input and button when isPaused=true", () => {
    render(<InputBar onSend={vi.fn()} isStreaming={false} isPaused={true} />);
    expect(screen.getByRole("textbox")).toBeDisabled();
    expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
  });

  it("does not call onSend for empty input", async () => {
    const onSend = vi.fn();
    render(<InputBar onSend={onSend} isStreaming={false} isPaused={false} />);
    await userEvent.click(screen.getByRole("button", { name: /send/i }));
    expect(onSend).not.toHaveBeenCalled();
  });
});
