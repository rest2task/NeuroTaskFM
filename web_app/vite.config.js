import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  root: resolve(import.meta.dirname, "frontend"),
  build: {
    outDir: resolve(import.meta.dirname, "dist"),
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
