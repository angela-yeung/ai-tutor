"use client";

import { useState, useRef } from "react";
import ChatWindow from "@/components/ChatWindow";
import InputBar from "@/components/InputBar";
import PausedBanner from "@/components/PausedBanner";
import ReviewPanel from "@/components/ReviewPanel";
import Sidebar from "@/components/Sidebar";
import { streamChat } from "@/lib/chat";
import { getThreadId, newSession } from "@/lib/session";
import type { Message, DoneMeta } from "@/lib/types";

const GREETING: Message = {
  id: "greeting",
  role: "assistant",
  content: "Hi Mia! I'm Pip. Want to warm up with a quick fractions puzzle, or jump into something else?",
};

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([GREETING]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [conceptsNeedingReview, setConceptsNeedingReview] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const threadIdRef = useRef<string>(
    typeof window !== "undefined" ? getThreadId() : ""
  );

  function addUserMessage(content: string): string {
    const id = crypto.randomUUID();
    setMessages((prev) => [...prev, { id, role: "user", content }]);
    return id;
  }

  function addAssistantPlaceholder(): string {
    const id = crypto.randomUUID();
    setMessages((prev) => [
      ...prev,
      { id, role: "assistant", content: "", streaming: true },
    ]);
    return id;
  }

  function appendToken(assistantId: string, chunk: string) {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === assistantId
          ? { ...m, content: m.content + chunk }
          : m
      )
    );
  }

  function finaliseAssistant(assistantId: string, meta: DoneMeta) {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === assistantId
          ? { ...m, content: meta.current_response || m.content, streaming: false }
          : m
      )
    );
    setIsPaused(meta.session_paused);
    setConceptsNeedingReview(meta.concepts_needing_review);
  }

  async function handleSend(message: string) {
    if (isStreaming) return;
    setError(null);
    addUserMessage(message);
    const assistantId = addAssistantPlaceholder();
    setIsStreaming(true);

    await streamChat(
      threadIdRef.current,
      message,
      (chunk) => appendToken(assistantId, chunk),
      (meta) => finaliseAssistant(assistantId, meta),
      (msg) => {
        setMessages((prev) => prev.filter((m) => m.id !== assistantId));
        setError(msg);
      }
    );

    setIsStreaming(false);
  }

  function handleResume() {
    handleSend("I'm back, let's continue.");
  }

  function handleNewChat() {
    threadIdRef.current = newSession();
    setMessages([GREETING]);
    setIsPaused(false);
    setConceptsNeedingReview([]);
    setError(null);
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />

      {/* Right panel */}
      <div className="flex-1 flex flex-col min-w-0 bg-stone-50">
        {/* Header */}
        <header className="flex items-center justify-between px-5 py-3 bg-white border-b border-stone-200 flex-none">
          <div className="flex flex-col">
            <span className="text-lg font-semibold text-stone-800">Pip</span>
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-emerald-500" />
              <span className="text-xs text-stone-500">Ready to help · Mia&apos;s tutor</span>
            </div>
          </div>
          <button
            onClick={handleNewChat}
            aria-label="New chat"
            className="p-1.5 rounded-lg text-stone-400 hover:text-stone-600 hover:bg-stone-100 transition-colors"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
              <path fillRule="evenodd" d="M15.312 11.424a5.5 5.5 0 01-9.201 2.466l-.312-.311h2.433a.75.75 0 000-1.5H3.989a.75.75 0 00-.75.75v4.242a.75.75 0 001.5 0v-2.43l.31.31a7 7 0 0011.712-3.138.75.75 0 00-1.449-.39zm1.23-3.723a.75.75 0 00.219-.53V2.929a.75.75 0 00-1.5 0V5.36l-.31-.31A7 7 0 003.239 8.188a.75.75 0 101.448.389A5.5 5.5 0 0113.89 6.11l.311.31h-2.432a.75.75 0 000 1.5h4.243a.75.75 0 00.53-.219z" clipRule="evenodd" />
            </svg>
          </button>
        </header>

        {/* Paused banner */}
        <PausedBanner isPaused={isPaused} onResume={handleResume} />

        {/* Chat window */}
        <ChatWindow messages={messages} onChipSend={handleSend} />

        {/* Error message */}
        {error && (
          <p className="px-4 py-2 text-red-600 text-base">{error}</p>
        )}

        {/* Input bar */}
        <InputBar
          onSend={handleSend}
          isStreaming={isStreaming}
          isPaused={isPaused}
        />

        {/* Review panel */}
        <ReviewPanel concepts={conceptsNeedingReview} />
      </div>
    </div>
  );
}
