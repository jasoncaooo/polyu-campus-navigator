import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The build lands in the folder the Python server already serves, so running the
// app still needs nothing but Python. `base` is /static/ because web.py maps
// /static/* onto that folder and serves its index.html at "/".
export default defineConfig({
  plugins: [react()],
  base: '/static/',
  build: {
    outDir: '../campus_nav/static',
    emptyOutDir: true,
    assetsDir: 'assets',
  },
  server: {
    // npm run dev, against the Python engine on its own port
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});
