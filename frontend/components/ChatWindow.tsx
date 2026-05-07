"use client";

import { useEffect, useRef } from "react";
import MessageBubble from "./MessageBubble";
import type { Message } from "@/lib/types";

const WELCOME_CHIPS = ["Fractions warm-up", "Read with me", "Something in science"];

interface Props {
  messages: Message[];
  onChipSend?: (text: string) => void;
}

export default function ChatWindow({ messages, onChipSend }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (bottomRef.current && typeof bottomRef.current.scrollIntoView === "function") {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6">
      <div className="flex flex-col gap-4">
        {messages.map((msg) =>
          msg.role === "assistant" ? (
            <div key={msg.id} className="flex items-end gap-2">
              <div className="w-7 h-7 rounded-full bg-emerald-500 flex items-center justify-center text-white text-xs font-semibold flex-none mb-0.5">
                P
              </div>
              <MessageBubble message={msg} />
            </div>
          ) : (
            <MessageBubble key={msg.id} message={msg} />
          )
        )}

        {messages.length === 1 && onChipSend && (
          <div className="flex flex-wrap gap-2 pl-9 mt-1">
            {WELCOME_CHIPS.map((chip) => (
              <button
                key={chip}
                onClick={() => onChipSend(chip)}
                className="px-3 py-1.5 rounded-full bg-stone-100 text-stone-700 text-sm hover:bg-stone-200 transition-colors border border-stone-200"
              >
                {chip}
              </button>
            ))}
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
