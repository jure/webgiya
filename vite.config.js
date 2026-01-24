// vite.config.js
import { defineConfig } from 'vite';
import mkcert from 'vite-plugin-mkcert'

export default defineConfig(({ command }) => ({
  base: command === 'serve' ? '/' : '/webgiya/',
  server: {
    watch: {
      usePolling: true, // Enable polling
      interval: 100, // Optional: adjust the polling interval in milliseconds (default is often 100ms)
    },
    https: true,
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  plugins: [mkcert()]
}));
