import { NavLink, Route, Routes } from 'react-router-dom'
import { LayoutGrid, Database, ListTodo, Library, Mic, Server } from 'lucide-react'
import Dashboard from './pages/Dashboard/Dashboard'
import Datasets from './pages/Datasets/Datasets'
import DatasetDetail from './pages/DatasetDetail/DatasetDetail'
import NewTraining from './pages/NewTraining/NewTraining'
import Jobs from './pages/Jobs/Jobs'
import JobDetail from './pages/JobDetail/JobDetail'
import Models from './pages/Models/Models'
import Remotes from './pages/Remotes/Remotes'
import Voice from './pages/Voice/Voice'
import './App.css'

export default function App() {
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <img src="/icon.svg" alt="LoRA" className="brand-logo" />
          <span className="brand-text">LoRA</span>
        </div>
        <nav className="nav-list">
          <NavLink to="/" end className="nav-link">
            <LayoutGrid className="nav-icon" size={18} /><span>概览</span>
          </NavLink>
          <NavLink to="/datasets" className="nav-link">
            <Database className="nav-icon" size={18} /><span>数据集</span>
          </NavLink>
          <NavLink to="/jobs" className="nav-link">
            <ListTodo className="nav-icon" size={18} /><span>训练任务</span>
          </NavLink>
          <NavLink to="/models" className="nav-link">
            <Library className="nav-icon" size={18} /><span>模型库</span>
          </NavLink>

          <div className="nav-section">声音克隆</div>
          <NavLink to="/voice" className="nav-link">
            <Mic className="nav-icon" size={18} /><span>声音 / SVC</span>
          </NavLink>

          <div className="nav-section">系统</div>
          <NavLink to="/remotes" className="nav-link">
            <Server className="nav-icon" size={18} /><span>算力 / 远程</span>
          </NavLink>
        </nav>
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
          <Route path="/voice" element={<Voice />} />
          <Route path="/remotes" element={<Remotes />} />
        </Routes>
      </main>
    </div>
  )
}
