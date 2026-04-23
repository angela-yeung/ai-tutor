"use client";

import { useState } from "react";

interface Props {
  concepts: string[];
}

export default function ReviewPanel({ concepts }: Props) {
  const [open, setOpen] = useState(false);

  if (concepts.length === 0) return null;

  const unique = Array.from(new Set(concepts));

  return (
    <div className="border-t border-emerald-200 bg-emerald-50">
      <button
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
        className="w-full flex items-center gap-2 px-4 py-3 text-emerald-700 text-base font-medium"
      >
        <span>{open ? "▲" : "▼"} Topics to revisit</span>
        <span className="ml-auto flex items-center justify-center w-6 h-6 rounded-full bg-emerald-600 text-white text-sm font-bold">
          {unique.length}
        </span>
      </button>
      {open && (
        <ul className="px-6 pb-4 space-y-1">
          {unique.map((concept) => (
            <li key={concept} className="text-emerald-800 text-base list-disc list-inside">
              {concept}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
