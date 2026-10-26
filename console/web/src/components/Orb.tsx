import { motion } from "framer-motion";
import { cn } from "@/lib/format";

export type Presence = "offline" | "idle" | "listening" | "transcribing" | "thinking" | "speaking";

const palette: Record<Presence, { core: string; halo: string }> = {
  offline: { core: "rgb(var(--muted) / 0.55)", halo: "rgb(var(--muted) / 0.08)" },
  idle: { core: "rgb(var(--glacier))", halo: "rgb(var(--glacier) / 0.16)" },
  listening: { core: "rgb(var(--glacier))", halo: "rgb(var(--glacier) / 0.34)" },
  transcribing: { core: "rgb(var(--glacier))", halo: "rgb(var(--glacier) / 0.26)" },
  thinking: { core: "rgb(var(--glacier))", halo: "rgb(var(--glacier) / 0.24)" },
  speaking: { core: "rgb(var(--glacier))", halo: "rgb(var(--glacier) / 0.4)" },
};

/** The assistant's presence: one shape whose motion says what it is doing. */
export function Orb({ presence, size = 40, className }: { presence: Presence; size?: number; className?: string }) {
  const colours = palette[presence];
  const active = presence !== "offline" && presence !== "idle";
  return (
    <div className={cn("relative shrink-0", className)} style={{ width: size, height: size }} aria-hidden>
      <motion.div
        className="absolute inset-0 rounded-full"
        style={{ background: `radial-gradient(circle, ${colours.halo} 0%, transparent 70%)` }}
        animate={active ? { scale: [1, 1.55, 1], opacity: [0.9, 0.3, 0.9] } : { scale: 1.25, opacity: 0.8 }}
        transition={{ duration: presence === "listening" ? 1.2 : 2.6, repeat: Infinity, ease: "easeInOut" }}
      />
      {presence === "listening" &&
        [0, 1].map((ring) => (
          <motion.div
            key={ring}
            className="absolute inset-0 rounded-full border"
            style={{ borderColor: "rgb(var(--glacier) / 0.5)" }}
            initial={{ scale: 0.9, opacity: 0.7 }}
            animate={{ scale: 1.9, opacity: 0 }}
            transition={{ duration: 1.6, repeat: Infinity, delay: ring * 0.8, ease: "easeOut" }}
          />
        ))}
      <motion.div
        className="absolute inset-[14%] rounded-full"
        style={{
          background: `radial-gradient(circle at 34% 28%, rgb(255 255 255 / 0.95) 0%, ${colours.core} 38%, rgb(28 111 138) 100%)`,
          boxShadow: presence === "offline" ? "none" : `0 0 ${size / 2}px ${colours.halo}`,
          filter: presence === "offline" ? "grayscale(1)" : "none",
        }}
        animate={
          presence === "thinking" || presence === "transcribing"
            ? { rotate: 360, scale: [1, 0.94, 1] }
            : presence === "speaking"
              ? { scale: [1, 1.08, 1] }
              : { scale: [1, 1.03, 1] }
        }
        transition={
          presence === "thinking" || presence === "transcribing"
            ? { rotate: { duration: 3, repeat: Infinity, ease: "linear" }, scale: { duration: 1.4, repeat: Infinity } }
            : { duration: presence === "speaking" ? 0.9 : 5, repeat: Infinity, ease: "easeInOut" }
        }
      />
      {(presence === "thinking" || presence === "transcribing") && (
        <motion.div
          className="absolute inset-0"
          animate={{ rotate: -360 }}
          transition={{ duration: 2.2, repeat: Infinity, ease: "linear" }}
        >
          <span
            className="absolute left-1/2 top-0 -translate-x-1/2 rounded-full"
            style={{ width: Math.max(3, size / 12), height: Math.max(3, size / 12), background: "rgb(var(--glacier))" }}
          />
        </motion.div>
      )}
    </div>
  );
}
