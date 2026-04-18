import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ReviewPanel from "../../components/ReviewPanel";

describe("ReviewPanel", () => {
  it("renders nothing when concepts list is empty", () => {
    const { container } = render(<ReviewPanel concepts={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("shows the count badge with number of concepts", () => {
    render(<ReviewPanel concepts={["fractions", "multiplication"]} />);
    expect(screen.getByText("2")).toBeInTheDocument();
  });

  it("starts collapsed — concept items not visible", () => {
    render(<ReviewPanel concepts={["fractions", "multiplication"]} />);
    expect(screen.queryByText("fractions")).not.toBeInTheDocument();
  });

  it("shows concepts when toggle is clicked", async () => {
    render(<ReviewPanel concepts={["fractions", "multiplication"]} />);
    await userEvent.click(screen.getByRole("button"));
    expect(screen.getByText("fractions")).toBeInTheDocument();
    expect(screen.getByText("multiplication")).toBeInTheDocument();
  });

  it("hides concepts when toggled a second time", async () => {
    render(<ReviewPanel concepts={["fractions"]} />);
    await userEvent.click(screen.getByRole("button"));
    expect(screen.getByText("fractions")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button"));
    expect(screen.queryByText("fractions")).not.toBeInTheDocument();
  });

  it("shows 'Topics to revisit' label in the toggle", () => {
    render(<ReviewPanel concepts={["fractions"]} />);
    expect(screen.getByText(/topics to revisit/i)).toBeInTheDocument();
  });
});
