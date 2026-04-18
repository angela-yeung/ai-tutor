import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import PausedBanner from "../../components/PausedBanner";

describe("PausedBanner", () => {
  it("is not rendered when isPaused=false", () => {
    const { container } = render(
      <PausedBanner isPaused={false} onResume={vi.fn()} />
    );
    expect(container.firstChild).toBeNull();
  });

  it("is rendered when isPaused=true", () => {
    render(<PausedBanner isPaused={true} onResume={vi.fn()} />);
    expect(screen.getByText(/grown-up/i)).toBeInTheDocument();
  });

  it("shows a Resume button when paused", () => {
    render(<PausedBanner isPaused={true} onResume={vi.fn()} />);
    expect(screen.getByRole("button", { name: /resume/i })).toBeInTheDocument();
  });

  it("calls onResume when Resume is clicked", async () => {
    const onResume = vi.fn();
    render(<PausedBanner isPaused={true} onResume={onResume} />);
    await userEvent.click(screen.getByRole("button", { name: /resume/i }));
    expect(onResume).toHaveBeenCalledOnce();
  });

  it("has amber styling when paused", () => {
    const { container } = render(
      <PausedBanner isPaused={true} onResume={vi.fn()} />
    );
    expect(container.firstChild).toHaveClass("bg-amber-50");
  });
});
