import { defineConfig } from "@playwright/test";

const ui = "http://127.0.0.1:3100";
const agent = "http://127.0.0.1:3201";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  use: { baseURL: ui, trace: "retain-on-failure" },
  webServer: [
    {
      command:
        "uv --directory ../fact-checker-agent run --locked python tests/serve_smoke.py",
      url: `${agent}/ok`,
      timeout: 120_000,
      gracefulShutdown: { signal: "SIGTERM", timeout: 5000 },
    },
    {
      command: "pnpm dev --hostname 127.0.0.1 --port 3100",
      url: ui,
      timeout: 120_000,
      env: {
        NEXT_PUBLIC_API_URL: agent,
        NEXT_PUBLIC_ASSISTANT_ID: "fact_checker",
      },
    },
  ],
});
