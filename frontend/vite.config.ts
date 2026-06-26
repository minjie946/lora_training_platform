import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 端口由 start.sh 通过环境变量注入，保证前端端口与代理目标和实际后端一致
const frontendPort = Number(process.env.FRONTEND_PORT) || 5173
const backendPort = Number(process.env.BACKEND_PORT) || 8000

export default defineConfig({
  plugins: [react()],
  server: {
    port: frontendPort,
    strictPort: true,
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${backendPort}`,
        changeOrigin: true,
      },
    },
  },
})
