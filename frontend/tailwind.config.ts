import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        risk: {
          high: "#ef4444",
          medium: "#f97316",
          low: "#22c55e",
        },
      },
    },
  },
  plugins: [],
};

export default config;
