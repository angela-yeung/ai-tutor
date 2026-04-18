# Frontend Design Spec — Sub-project 3: Next.js Chat UI

**Date:** 2026-04-18
**Scope:** React/Next.js frontend only. No auth (anonymous sessions). Supabase auth is Sub-project 2 (future).

---

## Context

The AI tutor currently has a FastAPI backend (Sub-project 1) that exposes `POST /chat` as an SSE stream and `GET /health`. This frontend is the student-facing web UI — a calm, minimal chat interface for Grade 1 students that streams the tutor's responses token-by-token. It lives in `frontend/` inside the existing repo and deploys as a separate Railway service.

---

## Architecture

**Framework:** Next.js 14 App Router with TypeScript  
**Styling:** Tailwind CSS  
**State:** React `useState` + `useRef` — no external state library  
**Session persistence:** `thread_id` stored in `localStorage` (survives page reloads)  
**API:** Direct `fetch` to FastAPI (no Next.js proxy route) via `NEXT_PUBLIC_API_URL` env var

```
frontend/
├── app/
│   ├── layout.tsx          # Root layout, Inter font, metadata
│   ├── page.tsx            # Single chat page — composes all components
│   └── globals.css         # Tailwind base + CSS custom properties
├── components/
│   ├── ChatWindow.tsx      # Scrollable message list, auto-scrolls to bottom
│   ├── MessageBubble.tsx   # Individual message bubble (user vs assistant)
│   ├── InputBar.tsx        # Text input + Send button (disabled when paused/streaming)
│   ├── PausedBanner.tsx    # Amber banner + Resume button (shown when session_paused=true)
│   └── ReviewPanel.tsx     # Collapsible "Topics to revisit" list with count badge
├── lib/
│   ├── chat.ts             # streamChat() — fetch POST + SSE ReadableStream parsing
│   └── session.ts          # getThreadId(): read/write thread_id in localStorage
├── package.json
├── tailwind.config.ts
├── next.config.ts
└── tsconfig.json
```

---

## Endpoints Consumed

| Method | URL | Usage |
|--------|-----|-------|
| `POST /chat` | `${NEXT_PUBLIC_API_URL}/chat` | Send student message, receive SSE stream |
| `GET /health` | `${NEXT_PUBLIC_API_URL}/health` | (Optional) startup liveness check |

**Request body:**
```json
{ "thread_id": "uuid-v4", "message": "What is 2 + 2?" }
```

**SSE events received:**
- `event: token` → `{"chunk": "string"}` — append to streaming bubble
- `event: done` → `{"session_paused": bool, "concept": string, "concepts_needing_review": string[], "conversation_history": [...]}` — apply final state
- `event: error` → `{"message": string}` — show inline error

---

## UI Layout

```
┌─────────────────────────────────────┐
│  🎓 AI Tutor              [New Chat]│  ← header (slate-800 text, white bg)
├─────────────────────────────────────┤
│  ⚠ A grown-up needs to help.        │  ← PausedBanner (amber-50 bg, amber-400
│    [Resume]                         │    border) — hidden unless session_paused
├─────────────────────────────────────┤
│                                     │
│   ChatWindow (flex-col, scrollable) │
│   ┌─────────────────────────┐       │
│   │ Student bubble (blue-100│       │  ← right-aligned
│   └─────────────────────────┘       │
│   ┌──────────────────────┐          │
│   │ Tutor bubble (white) │          │  ← left-aligned, streams in real-time
│   └──────────────────────┘          │
│                                     │
├─────────────────────────────────────┤
│  [Type your question...  ] [Send →] │  ← InputBar — disabled while streaming/paused
├─────────────────────────────────────┤
│  [▼ Topics to revisit (2)]          │  ← ReviewPanel toggle (emerald-600 text)
│  • fractions                        │  ← collapsible, hidden by default
│  • multiplication                   │
└─────────────────────────────────────┘
```

Single column, max-width 680px, centred on desktop. Full-height on mobile.

---

## Colour Palette (calm & focused, soft pastels)

| Element | Tailwind class |
|---------|---------------|
| Page background | `bg-slate-50` |
| Header | `bg-white border-b border-slate-200` |
| Student bubble | `bg-blue-100 border border-blue-200` |
| Tutor bubble | `bg-white border border-slate-200` |
| Send button | `bg-blue-500 hover:bg-blue-600 text-white` |
| Input field | `border border-slate-300 focus:ring-blue-400` |
| Paused banner | `bg-amber-50 border border-amber-400 text-amber-800` |
| Resume button | `bg-amber-500 hover:bg-amber-600 text-white` |
| Review panel | `bg-emerald-50 border border-emerald-200` |
| Review toggle | `text-emerald-700` |
| Font | Inter (Next.js `next/font/google`) |
| Base text size | `text-lg` (18px) — large for Grade 1 readability |

---

## Session Management

**`lib/session.ts`:**
```typescript
export function getThreadId(): string {
  let id = localStorage.getItem("thread_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("thread_id", id);
  }
  return id;
}

export function newSession(): string {
  const id = crypto.randomUUID();
  localStorage.setItem("thread_id", id);
  return id;
}
```

- `getThreadId()` called on component mount (in `useEffect`)
- `newSession()` called when student clicks "New Chat" → clears messages, resets local state, generates new UUID
- **No history restored on reload** — blank chat, tutor retains full context via Redis

---

## Streaming Implementation

**`lib/chat.ts`:**

```typescript
export async function streamChat(
  threadId: string,
  message: string,
  onToken: (chunk: string) => void,
  onDone: (meta: DoneMeta) => void,
  onError: (msg: string) => void
): Promise<void> {
  const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ thread_id: threadId, message }),
  });

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop()!;
    for (const part of parts) {
      const eventLine = part.match(/^event: (\w+)/m)?.[1];
      const dataLine = part.match(/^data: (.+)/m)?.[1];
      if (!eventLine || !dataLine) continue;
      const payload = JSON.parse(dataLine);
      if (eventLine === "token") onToken(payload.chunk);
      else if (eventLine === "done") onDone(payload);
      else if (eventLine === "error") onError(payload.message);
    }
  }
}

export interface DoneMeta {
  session_paused: boolean;
  concept: string;
  concepts_needing_review: string[];
  conversation_history: { role: string; content: string }[];
}
```

---

## Session Resume Flow

When `session_paused=true`:
1. `PausedBanner` appears, input is disabled
2. Parent reads the tutor's message and assists the student
3. Parent/student clicks **"Resume"** button in the banner
4. Frontend calls `streamChat(threadId, "I'm back, let's continue.", ...)` 
5. Backend `entry_router` detects `session_paused=true` → routes to `resume_session` node → welcome-back message, clears pause
6. On `event: done` with `session_paused=false`: banner hides, input re-enables

---

## New Session Flow

"New Chat" button in header:
1. Calls `newSession()` → new UUID saved to localStorage
2. Clears `messages` array in local state
3. Resets `sessionPaused=false`, `conceptsNeedingReview=[]`
4. Old session data remains in Redis (orphaned but harmless)

---

## Page Reload Behaviour

- `thread_id` persists in localStorage → same session resumes
- Chat screen starts blank (no history restored)
- Tutor has full context from Redis — first message gets a contextually aware response
- If `session_paused` was true before reload: the banner will NOT show on reload (frontend state is reset). The student/parent must send a message; the `done` event will set `session_paused=true` again if the backend state still has it set.

---

## Error Handling

| Scenario | UI behaviour |
|----------|-------------|
| `event: error` from backend | Inline error below input: "Something went wrong. Please try again." |
| Network failure (`fetch` throws) | Same inline error |
| Stream ends without `done` event | Treat as complete; no banner |

---

## Environment Variables

| Variable | Required | Default (local dev) | Purpose |
|----------|----------|---------------------|---------|
| `NEXT_PUBLIC_API_URL` | Yes | `http://localhost:8000` | FastAPI backend base URL |

Set in Railway dashboard to `https://api-prod.up.railway.app` for production.

---

## Local Development

```bash
cd frontend
npm install
npm run dev      # http://localhost:3000
```

FastAPI backend must be running on `localhost:8000` (via `uvicorn api.main:app --reload` or `docker compose up`).

---

## Railway Deployment

1. In Railway project, add a new service → "Deploy from repo" → set **Root Directory** to `frontend`
2. Build command: `npm run build`
3. Start command: `npm start`
4. Set env var: `NEXT_PUBLIC_API_URL=https://<api-service>.railway.app`
5. FastAPI service: add `CORS_ORIGINS=https://<frontend-service>.railway.app`

---

## Testing

- Component tests: Vitest + React Testing Library
  - `session.ts`: `getThreadId` generates UUID on first call, returns same on second
  - `chat.ts`: parses SSE chunks correctly from a mock ReadableStream
  - `ReviewPanel`: toggles open/closed, shows correct count badge
  - `PausedBanner`: visible when `sessionPaused=true`, hidden otherwise
  - `InputBar`: disabled when `isStreaming=true` or `sessionPaused=true`
- E2E: Playwright (student sends message → streaming tokens appear → done state applied)

---

## What This Spec Does NOT Cover

- Supabase login/signup (Sub-project 2)
- User profiles / persistent student names
- Conversation history restore on reload (future)
- Stop/interrupt mid-stream (future)
