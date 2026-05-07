const LESSONS = [
  { icon: "🍕", bg: "bg-orange-100", color: "text-orange-600", title: "Fraction pizza party", meta: "5 min · Math" },
  { icon: "📖", bg: "bg-blue-100",   color: "text-blue-600",   title: "Read along: The Lost Key", meta: "10 min · Reading" },
  { icon: "🌿", bg: "bg-green-100",  color: "text-green-600",  title: "Why leaves are green", meta: "Science" },
];

const TABS = ["Review", "Fix-ups", "What's next"];

export default function Sidebar() {
  return (
    <aside className="w-60 flex-none flex flex-col bg-white border-r border-stone-200 overflow-y-auto">
      {/* Student profile */}
      <div className="p-4 border-b border-stone-100">
        <div className="w-10 h-10 rounded-full bg-emerald-500 flex items-center justify-center text-white font-semibold text-sm">
          KI
        </div>
        <p className="text-base font-semibold text-stone-800 mt-2">Hi, Mia!</p>
        <p className="text-xs text-stone-500">Grade 4 · Tuesday afternoon</p>
      </div>

      {/* Nav tabs */}
      <div className="flex border-b border-stone-100">
        {TABS.map((tab) => {
          const active = tab === "What's next";
          return (
            <button
              key={tab}
              className={[
                "flex-1 py-2 text-center text-xs font-medium transition-colors",
                active
                  ? "text-emerald-600 border-b-2 border-emerald-500"
                  : "text-stone-500 hover:text-stone-700",
              ].join(" ")}
            >
              {tab}
            </button>
          );
        })}
      </div>

      {/* Suggested next */}
      <p className="px-4 pt-4 pb-2 text-xs font-semibold text-stone-500 uppercase tracking-wide">
        Suggested next
      </p>
      <div className="flex flex-col">
        {LESSONS.map((lesson) => (
          <div
            key={lesson.title}
            className="flex items-center gap-3 px-4 py-3 hover:bg-stone-50 transition-colors cursor-default"
          >
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center text-sm flex-none ${lesson.bg} ${lesson.color}`}>
              {lesson.icon}
            </div>
            <div className="min-w-0">
              <p className="text-sm font-medium text-stone-800 truncate">{lesson.title}</p>
              <p className="text-xs text-stone-500">{lesson.meta}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Progress */}
      <div className="mt-auto p-4 border-t border-stone-100">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-stone-600">Today&apos;s plan</span>
          <span className="text-xs text-stone-500">3 / 5</span>
        </div>
        <div className="h-1.5 rounded-full bg-stone-200">
          <div className="h-1.5 rounded-full bg-emerald-500 w-[60%]" />
        </div>
      </div>
    </aside>
  );
}
