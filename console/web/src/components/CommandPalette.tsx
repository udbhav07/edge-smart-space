import { Command } from "cmdk";
import {
  CalendarDays,
  CircleAlert,
  Gauge,
  MessageSquare,
  RefreshCw,
  RotateCcw,
  Thermometer,
  Unplug,
  Wrench,
} from "lucide-react";
import type { PanelTab } from "./ContextPanel";

export interface PaletteActions {
  ask: (text: string) => void;
  focusComposer: () => void;
  setTarget: (value: number) => void;
  openTab: (tab: PanelTab) => void;
  refreshCalendar: () => void;
  inject: (subject: string, fault: string, magnitude?: number) => void;
  reset: () => void;
}

const Item = ({ icon: Icon, label, hint, onSelect }: { icon: typeof Gauge; label: string; hint?: string; onSelect: () => void }) => (
  <Command.Item onSelect={onSelect} className="flex cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-[14px] text-fg">
    <Icon className="h-4 w-4 text-muted" />
    {label}
    {hint && <span className="ml-auto text-[12px] text-muted">{hint}</span>}
  </Command.Item>
);

export function CommandPalette({ open, onOpenChange, actions, operator }: { open: boolean; onOpenChange: (open: boolean) => void; actions: PaletteActions; operator: boolean }) {
  const run = (action: () => void) => () => {
    onOpenChange(false);
    action();
  };
  return (
    <Command.Dialog
      open={open}
      onOpenChange={onOpenChange}
      label="Commands"
      overlayClassName="fixed inset-0 z-40 bg-black/40 backdrop-blur-[2px]"
      contentClassName="fixed left-1/2 top-[18vh] z-50 w-[min(640px,calc(100vw-32px))] -translate-x-1/2 overflow-hidden rounded-2xl border hairline bg-surface shadow-[0_30px_80px_-20px_rgb(0_0_0/0.7)]"
    >
      <Command.Input
        placeholder="Type a command, or something to ask…"
        className="w-full border-b hairline bg-transparent px-4 py-3.5 text-[15px] text-fg placeholder:text-muted focus:outline-none"
      />
      <Command.List className="scroll-thin max-h-[52vh] overflow-y-auto p-2">
        <Command.Empty className="px-3 py-6 text-center text-[13px] text-muted">Nothing matches. Press Enter in the message box to ask Jarvis instead.</Command.Empty>
        <Command.Group heading="Ask">
          <Item icon={MessageSquare} label="Write a message" hint="/" onSelect={run(actions.focusComposer)} />
          <Item icon={CalendarDays} label="What's on my calendar today?" onSelect={run(() => actions.ask("What's on my calendar today?"))} />
          <Item icon={Gauge} label="Is anything wrong with the room?" onSelect={run(() => actions.ask("Is anything wrong with the room?"))} />
        </Command.Group>
        <Command.Group heading="Comfort">
          {[22, 23, 24, 25].map((value) => (
            <Item key={value} icon={Thermometer} label={`Set the room to ${value}°`} onSelect={run(() => actions.setTarget(value))} />
          ))}
        </Command.Group>
        <Command.Group heading="View">
          <Item icon={Gauge} label="Room" onSelect={run(() => actions.openTab("room"))} />
          <Item icon={Wrench} label="Agent trace" onSelect={run(() => actions.openTab("agent"))} />
          <Item icon={CalendarDays} label="Calendar" onSelect={run(() => actions.openTab("calendar"))} />
          <Item icon={CircleAlert} label="System and faults" onSelect={run(() => actions.openTab("system"))} />
          <Item icon={RefreshCw} label="Refresh the calendar" onSelect={run(actions.refreshCalendar)} />
        </Command.Group>
        {operator && (
          <Command.Group heading="Operator">
            <Item icon={Unplug} label="Inject: thermometer drops out" onSelect={run(() => actions.inject("temp_01", "DROPOUT"))} />
            <Item icon={Unplug} label="Inject: thermometer stuck at 27°" onSelect={run(() => actions.inject("temp_01", "STUCK_AT", 27))} />
            <Item icon={Unplug} label="Inject: impossible reading (999°)" onSelect={run(() => actions.inject("temp_01", "OUT_OF_RANGE", 999))} />
            <Item icon={Unplug} label="Inject: air conditioner dead" onSelect={run(() => actions.inject("ac", "NO_RESPONSE"))} />
            <Item icon={RefreshCw} label="Clear the thermometer" onSelect={run(() => actions.inject("temp_01", "NONE"))} />
            <Item icon={RefreshCw} label="Clear the air conditioner" onSelect={run(() => actions.inject("ac", "NONE"))} />
            <Item icon={RotateCcw} label="Reset after checking" onSelect={run(actions.reset)} />
          </Command.Group>
        )}
      </Command.List>
    </Command.Dialog>
  );
}
