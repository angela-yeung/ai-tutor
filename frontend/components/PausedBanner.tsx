interface Props {
  isPaused: boolean;
  onResume: () => void;
}

export default function PausedBanner({ isPaused, onResume }: Props) {
  if (!isPaused) return null;

  return (
    <div className="flex items-center justify-between px-4 py-3 bg-amber-50 border border-amber-400 text-amber-800">
      <span className="text-lg">⚠ A grown-up needs to help.</span>
      <button
        onClick={onResume}
        className="rounded-lg bg-amber-500 hover:bg-amber-600 text-white px-4 py-1.5 text-base font-medium transition-colors"
      >
        Resume
      </button>
    </div>
  );
}
