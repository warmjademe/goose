import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SettingsModal } from "../SettingsModal";

vi.mock("../AppearanceSettings", () => ({
  AppearanceSettings: () => <div>Appearance settings</div>,
}));

vi.mock("../ProvidersSettings", () => ({
  ProvidersSettings: () => <div>Provider settings</div>,
}));

vi.mock("../VoiceInputSettings", () => ({
  VoiceInputSettings: () => <div>Voice settings</div>,
}));

vi.mock("../GeneralSettings", () => ({
  GeneralSettings: () => <div>General settings</div>,
}));

vi.mock("../CompactionSettings", () => ({
  CompactionSettings: () => <div>Compaction settings</div>,
}));

vi.mock("../ProjectsSettings", () => ({
  ProjectsSettings: () => <div>Projects settings</div>,
}));

vi.mock("../ChatsSettings", () => ({
  ChatsSettings: () => <div>Chats settings</div>,
}));

vi.mock("../DoctorSettings", () => ({
  DoctorSettings: () => <div>Doctor settings</div>,
}));

vi.mock("../AboutSettings", () => ({
  AboutSettings: () => <div>About settings</div>,
}));

describe("SettingsModal", () => {
  it("closes on a backdrop click", () => {
    const onClose = vi.fn();

    render(<SettingsModal onClose={onClose} />);

    const backdrop = screen.getByTestId("settings-backdrop");
    fireEvent.pointerDown(backdrop, { clientX: 20, clientY: 20 });
    fireEvent.click(backdrop, { clientX: 20, clientY: 20 });

    expect(onClose).toHaveBeenCalledOnce();
  });

  it("keeps settings open when the backdrop pointer moves like a window drag", () => {
    const onClose = vi.fn();

    render(<SettingsModal onClose={onClose} />);

    const backdrop = screen.getByTestId("settings-backdrop");
    fireEvent.pointerDown(backdrop, { clientX: 20, clientY: 20 });
    fireEvent.click(backdrop, { clientX: 44, clientY: 20 });

    expect(onClose).not.toHaveBeenCalled();
  });
});
