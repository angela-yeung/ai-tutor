"use client";

import { useState, KeyboardEvent } from "react";

const TRY_ASKING = ["Help me with fractions", "Read with me", "Photosynthesis?", "Spell 'rhythm'"];

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
    <div className="flex-none bg-white border-t border-stone-200">
      <div className="flex items-center gap-2 px-4 py-3">
        <input
          type="text"
          className="flex-1 rounded-xl border border-stone-200 bg-stone-50 px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-400 disabled:opacity-50"
          placeholder="Ask Pip anything..."
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
        />
        <button
          onClick={handleSend}
          disabled={disabled}
          aria-label="Send"
          className="p-2.5 rounded-xl bg-emerald-500 hover:bg-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors flex items-center justify-center"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
            <path d="M3.105 2.289a.75.75 0 00-.826.95l1.903 6.557H13.5a.75.75 0 010 1.5H4.182l-1.903 6.557a.75.75 0 00.826.95 28.896 28.896 0 0015.293-7.154.75.75 0 000-1.115A28.897 28.897 0 003.105 2.289z" />
          </svg>
        </button>
      </div>
      <div className="flex items-center gap-2 px-4 pb-3 flex-wrap">
        <span className="text-xs text-stone-400 flex-none">Try asking:</span>
        {TRY_ASKING.map((chip) => (
          <button
            key={chip}
            onClick={() => onSend(chip)}
            disabled={disabled}
            className="px-2.5 py-1 rounded-full bg-stone-100 text-xs text-stone-600 hover:bg-stone-200 transition-colors border border-stone-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {chip}
          </button>
        ))}
      </div>
    </div>
  );
}
