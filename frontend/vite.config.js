import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Change 'ccel-daily-news' to your actual GitHub repo name
  base: '/CCEL_NEWS/',
  build: {
    outDir: 'dist',
  },
})
