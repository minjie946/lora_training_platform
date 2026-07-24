import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import './Select.css'

export interface SelectOption {
    value: string | number
    label: string
    // 可选说明：展开的下拉面板里显示在 label 下方第二行；选中态（trigger）只展示 label。
    desc?: string
}

interface Props {
    value: string | number
    options: SelectOption[]
    onChange: (value: string | number) => void
    placeholder?: string
    disabled?: boolean
}

// 自定义下拉框：选项面板贴在选择框下方，空间不足时翻到上方，避免原生 select 的居中遮挡问题。
export default function Select({ value, options, onChange, placeholder = '请选择', disabled }: Props) {
    const [open, setOpen] = useState(false)
    const [dropUp, setDropUp] = useState(false)
    const rootRef = useRef<HTMLDivElement>(null)
    const panelRef = useRef<HTMLDivElement>(null)

    const selected = options.find((o) => o.value === value)

    // 点击外部 / 按下 Esc 时关闭
    useEffect(() => {
        if (!open) return
        const onDocClick = (e: MouseEvent) => {
            if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false)
        }
        const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
        document.addEventListener('mousedown', onDocClick)
        document.addEventListener('keydown', onKey)
        return () => {
            document.removeEventListener('mousedown', onDocClick)
            document.removeEventListener('keydown', onKey)
        }
    }, [open])

    // 打开时根据剩余空间决定向下还是向上展开
    useLayoutEffect(() => {
        if (!open || !rootRef.current) return
        const rect = rootRef.current.getBoundingClientRect()
        const panelH = panelRef.current?.offsetHeight ?? 220
        const spaceBelow = window.innerHeight - rect.bottom
        setDropUp(spaceBelow < panelH + 12 && rect.top > spaceBelow)
    }, [open])

    const pick = (v: string | number) => { onChange(v); setOpen(false) }

    return (
        <div className={`select ${disabled ? 'disabled' : ''}`} ref={rootRef}>
            <button
                type="button"
                className="select-trigger"
                disabled={disabled}
                onClick={() => setOpen((o) => !o)}
            >
                <span className={selected ? '' : 'muted'}>{selected ? selected.label : placeholder}</span>
                <span className={`select-caret ${open ? 'up' : ''}`}>▾</span>
            </button>
            {open && (
                <div className={`select-panel ${dropUp ? 'up' : 'down'}`} ref={panelRef} role="listbox">
                    {options.map((o) => (
                        <div
                            key={o.value}
                            role="option"
                            aria-selected={o.value === value}
                            className={`select-option ${o.value === value ? 'active' : ''}`}
                            onClick={() => pick(o.value)}
                        >
                            <span className="select-option-label">{o.label}</span>
                            {o.desc && <span className="select-option-desc">{o.desc}</span>}
                        </div>
                    ))}
                </div>
            )}
        </div>
    )
}
