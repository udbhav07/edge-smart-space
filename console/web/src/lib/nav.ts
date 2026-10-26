import { BookOpen, CalendarDays, Cpu, Gauge, House, LayoutDashboard, Settings, Library, Mail, MessageSquare, Network, Plane, ShieldAlert, type LucideIcon } from "lucide-react";

export interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
  end?: boolean;
}

/** Every page, grouped as the sidebar shows them. */
export const NAV: { group: string; items: NavItem[] }[] = [
  {
    group: "Room",
    items: [{ to: "/", label: "Home", icon: House, end: true }],
  },
  {
    group: "Monitor",
    items: [
      { to: "/overview", label: "Overview", icon: LayoutDashboard },
      { to: "/services", label: "Services", icon: Cpu },
      { to: "/estimator", label: "Thermal model", icon: Gauge },
      { to: "/mqtt", label: "MQTT board", icon: Network },
      { to: "/faults", label: "Faults & control", icon: ShieldAlert },
    ],
  },
  {
    group: "Assistant",
    items: [
      { to: "/chat", label: "Chat", icon: MessageSquare },
      { to: "/calendar", label: "Calendar", icon: CalendarDays },
      { to: "/bookings", label: "Bookings", icon: Plane },
      { to: "/mail", label: "Mail", icon: Mail },
    ],
  },
  {
    group: "Learn",
    items: [
      { to: "/architecture", label: "How it works", icon: BookOpen },
      { to: "/topics", label: "Topic guide", icon: Library },
    ],
  },
  {
    group: "You",
    items: [{ to: "/settings", label: "Settings", icon: Settings }],
  },
];
