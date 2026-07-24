import { ReactNode } from 'react'
import './PageHeader.css'

interface PageHeaderProps {
  title: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
}

// 统一的吸顶页头：左侧标题 + 竖线分隔的副标题，右侧操作区。
// 对齐 Prototype v2.2 的 PageHeader 布局。
export default function PageHeader({ title, subtitle, actions }: PageHeaderProps) {
  return (
    <div className="page-header">
      <div className="page-header-titles">
        <div className="page-header-title">{title}</div>
        {subtitle != null && subtitle !== '' && (
          <>
            <span className="page-header-divider" />
            <div className="page-header-subtitle">{subtitle}</div>
          </>
        )}
      </div>
      {actions && <div className="page-header-actions">{actions}</div>}
    </div>
  )
}
