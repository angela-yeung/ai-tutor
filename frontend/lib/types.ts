export interface DoneMeta {
  session_paused: boolean;
  concept: string;
  concepts_needing_review: string[];
  conversation_history: { role: string; content: string }[];
  current_response: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
}
