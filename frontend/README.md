# RAGBench

RAGBench is a frontend workbench for evaluating retrieval strategies in RAG systems.

## Local setup

```bash
pnpm install
pnpm dev
```

Open `http://localhost:3000`.

## Demo mode

The app ships with clearly labeled illustrative fixtures so the UI can be explored without a backend. Configure `NEXT_PUBLIC_API_URL` in the Settings view when connecting a Python/FastAPI evaluation service. API calls should be routed through `lib/api-client.ts`.
