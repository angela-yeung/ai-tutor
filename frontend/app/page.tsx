"use client";

import { useState, useRef } from "react";
import ChatWindow from "@/components/ChatWindow";
import InputBar from "@/components/InputBar";
import PausedBanner from "@/components/PausedBanner";
import ReviewPanel from "@/components/ReviewPanel";
import { streamChat } from "@/lib/chat";
import { getThreadId, newSession } from "@/lib/session";
import type { Message, DoneMeta } from "@/lib/types";

const GREETING: Message = {
  id: "greeting",
  role: "assistant",
  content: "Hi! I'm your tutor. What would you like to learn today?",
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
    <div className="flex flex-col h-screen max-w-[680px] mx-auto">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 bg-white border-b border-slate-200">
        <span className="text-xl font-semibold text-slate-800">AI Tutor</span>
        <button
          onClick={handleNewChat}
          className="text-sm text-slate-500 hover:text-slate-800 transition-colors"
        >
          New Chat
        </button>
      </header>

      {/* Paused banner */}
      <PausedBanner isPaused={isPaused} onResume={handleResume} />

      {/* Chat window */}
      <ChatWindow messages={messages} />

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
  );
}
