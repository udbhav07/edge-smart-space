/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["-apple-system", "BlinkMacSystemFont", '"SF Pro Text"', '"Segoe UI Variable Text"', '"Segoe UI"', '"Geist Variable"', "system-ui", "sans-serif"],
        mono: ['"SF Mono"', "ui-monospace", '"Cascadia Mono"', '"Geist Mono Variable"', "Consolas", "monospace"],
      },
      colors: {
        ink: "rgb(var(--ink) / <alpha-value>)",
        surface: "rgb(var(--surface) / <alpha-value>)",
        raised: "rgb(var(--raised) / <alpha-value>)",
        line: "rgb(var(--line) / <alpha-value>)",
        fg: "rgb(var(--fg) / <alpha-value>)",
        muted: "rgb(var(--muted) / <alpha-value>)",
        glacier: "rgb(var(--glacier) / <alpha-value>)",
        ember: "rgb(var(--ember) / <alpha-value>)",
        alert: "rgb(var(--alert) / <alpha-value>)",
        good: "rgb(var(--good) / <alpha-value>)",
      },
      keyframes: {
        breathe: { "0%,100%": { transform: "scale(1)" }, "50%": { transform: "scale(1.04)" } },
        shimmer: { "0%": { backgroundPosition: "-200% 0" }, "100%": { backgroundPosition: "200% 0" } },
        rise: { "0%": { transform: "translateY(6px)" }, "100%": { transform: "translateY(0)" } },
      },
      animation: {
        breathe: "breathe 5s ease-in-out infinite",
        shimmer: "shimmer 2.4s linear infinite",
        rise: "rise .35s ease-out both",
      },
    },
  },
  plugins: [],
};
