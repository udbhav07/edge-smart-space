import { Suspense, lazy } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Shell } from "./Shell";

// One route per page; the gateway serves index.html for any path, so a page
// can be bookmarked or reloaded directly. Each page is its own chunk, so the
// chart library loads with the pages that draw charts, not before.
const page = <K extends string>(load: () => Promise<Record<K, React.ComponentType>>, name: K) =>
  lazy(() => load().then((module) => ({ default: module[name] })));

const Home = page(() => import("./pages/Home"), "Home");
const Settings = page(() => import("./pages/Settings"), "Settings");
const Overview = page(() => import("./pages/Overview"), "Overview");
const Services = page(() => import("./pages/Services"), "Services");
const Estimator = page(() => import("./pages/Estimator"), "Estimator");
const Mqtt = page(() => import("./pages/Mqtt"), "Mqtt");
const Faults = page(() => import("./pages/Faults"), "Faults");
const Chat = page(() => import("./pages/Chat"), "Chat");
const Calendar = page(() => import("./pages/Calendar"), "Calendar");
const Bookings = page(() => import("./pages/Bookings"), "Bookings");
const Mail = page(() => import("./pages/Mail"), "Mail");
const Architecture = page(() => import("./pages/Architecture"), "Architecture");
const Topics = page(() => import("./pages/Topics"), "Topics");
const TopicGuide = page(() => import("./pages/Topics"), "TopicGuide");

function Loading() {
  return <div className="px-8 py-10 text-[13px] text-muted">Loading…</div>;
}

export default function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route element={<Shell />}>
          <Route index element={<Home />} />
          <Route path="overview" element={<Overview />} />
          <Route path="settings" element={<Settings />} />
          <Route path="services" element={<Services />} />
          <Route path="estimator" element={<Estimator />} />
          <Route path="mqtt" element={<Mqtt />} />
          <Route path="faults" element={<Faults />} />
          <Route path="chat" element={<Chat />} />
          <Route path="calendar" element={<Calendar />} />
          <Route path="bookings" element={<Bookings />} />
          <Route path="mail" element={<Mail />} />
          <Route path="architecture" element={<Architecture />} />
          <Route path="topics" element={<Topics />} />
          <Route path="topics/:slug" element={<TopicGuide />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
