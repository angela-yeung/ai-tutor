import type { Message } from "@/lib/types";

interface Props {
  message: Message;
}

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";
  return (
    <div
      className={[
        "max-w-[80%] rounded-2xl px-4 py-3 text-lg leading-relaxed",
        isUser
          ? "self-end bg-blue-100 border border-blue-200 text-slate-800"
          : "self-start bg-white border border-slate-200 text-slate-800",
      ].join(" ")}
    >
      {message.content}
      {message.streaming && (
        <span className="inline-block w-2 h-4 ml-0.5 bg-slate-400 animate-pulse rounded-sm" />
      )}
    </div>
  );
}
