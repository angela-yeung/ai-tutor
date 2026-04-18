"use client";

import { useState, KeyboardEvent } from "react";

interface Props {
  onSend: (message: string) => void;
  isStreaming: boolean;
  isPaused: boolean;
}

export default function InputBar({ onSend, isStreaming, isPaused }: Props) {
  const [value, setValue] = useState("");
  const disabled = isStreaming || isPaused;

  function handleSend() {
    const trimmed = value.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setValue("");
  }

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="flex gap-3 px-4 py-3 border-t border-slate-200 bg-white">
      <input
        type="text"
        className="flex-1 rounded-xl border border-slate-300 px-4 py-2 text-lg focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:opacity-50"
        placeholder="Type your question..."
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
      />
      <button
        onClick={handleSend}
        disabled={disabled}
        className="rounded-xl bg-blue-500 hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed text-white px-5 py-2 text-lg font-medium transition-colors"
      >
        Send →
      </button>
    </div>
  );
}
