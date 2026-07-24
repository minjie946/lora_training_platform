import { useState } from 'react'
import { NavLink, Route, Routes } from 'react-router-dom'
import {
  LayoutGrid, Database, ListTodo, Library, Mic, Server, Images as ImagesIcon,
  Box, ChevronLeft, ChevronRight,
} from 'lucide-react'
import Dashboard from './pages/Dashboard/Dashboard'
import Datasets from './pages/Datasets/Datasets'
import DatasetDetail from './pages/DatasetDetail/DatasetDetail'
import NewTraining from './pages/NewTraining/NewTraining'
import Jobs from './pages/Jobs/Jobs'
import JobDetail from './pages/JobDetail/JobDetail'
import Models from './pages/Models/Models'
import Remotes from './pages/Remotes/Remotes'
import Voice from './pages/Voice/Voice'
import Images from './pages/Images/Images'
import './App.css'

const navItems = [
  { to: '/', end: true, icon: LayoutGrid, label: '概览' },
  { to: '/images', icon: ImagesIcon, label: '图片管理' },
  { to: '/datasets', icon: Database, label: '数据集' },
  { to: '/jobs', icon: ListTodo, label: '训练任务' },
  { to: '/models', icon: Library, label: '模型库' },
  { section: '声音克隆' },
  { to: '/voice', icon: Mic, label: '声音 / SVC' },
  { section: '系统' },
  { to: '/remotes', icon: Server, label: '算力 / 远程' },
] as const

export default function App() {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <div className="app">
      <aside className={`sidebar${collapsed ? ' collapsed' : ''}`}>
        <div className="brand">
          <Box className="brand-logo" size={24} />
          {!collapsed && <span className="brand-text">LoRA</span>}
        </div>
        <nav className="nav-list">
          {navItems.map((item, i) => {
            if ('section' in item) {
              return collapsed
                ? <div key={i} className="nav-divider" />
                : <div key={i} className="nav-section">{item.section}</div>
            }
            const Icon = item.icon
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={'end' in item ? item.end : undefined}
                className="nav-link"
                title={collapsed ? item.label : undefined}
              >
                <Icon className="nav-icon" size={18} />
                {!collapsed && <span>{item.label}</span>}
              </NavLink>
            )
          })}
        </nav>
        <div className="sidebar-foot">
          <button
            className="collapse-btn"
            onClick={() => setCollapsed((c) => !c)}
            title={collapsed ? '展开菜单' : '收起菜单'}
          >
            {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
          </button>
        </div>
      </aside>
      <main className="main">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/datasets/:id" element={<DatasetDetail />} />
          <Route path="/jobs/new" element={<NewTraining />} />
          <Route path="/jobs/:id/edit" element={<NewTraining />} />
          <Route path="/jobs" element={<Jobs />} />
          <Route path="/jobs/:id" element={<JobDetail />} />
          <Route path="/models" element={<Models />} />
          <Route path="/images" element={<Images />} />
          <Route path="/voice" element={<Voice />} />
          <Route path="/remotes" element={<Remotes />} />
        </Routes>
      </main>
    </div>
  )
}
