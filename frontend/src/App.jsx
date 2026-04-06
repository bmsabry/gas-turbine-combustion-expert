import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeRaw from 'rehype-raw'
import 'katex/dist/katex.min.css'
import AdminPanel from './AdminPanel'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [selectedSource, setSelectedSource] = useState(null)
  const [showAdmin, setShowAdmin] = useState(false)
  const chatEndRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = async (e) => {
    e?.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input, history: messages })
      })
      
      const data = await response.json()
      
      const assistantMessage = {
        role: 'assistant',
        content: data.response,
        sources: data.sources || [],
        conflicts: data.conflicts || [],
        singleStudy: data.single_study_notes || []
      }
      
      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Error connecting to the server. Please try again.',
        error: true
      }])
    }
    setLoading(false)
  }

  const renderMessage = (msg, idx) => {
    if (msg.role === 'user') {
      return (
        <div key={idx} className="message user">
          {msg.content}
        </div>
      )
    }

    return (
      <div key={idx} className="message assistant">
        <ReactMarkdown
          remarkPlugins={[remarkMath, remarkGfm]}
          rehypePlugins={[rehypeKatex, rehypeRaw]}
          components={{
            table: ({node, ...props}) => (
              <div className="table-wrapper">
                <table {...props} />
              </div>
            ),
            th: ({node, ...props}) => <th {...props} />,
            td: ({node, ...props}) => <td {...props} />,
            tr: ({node, ...props}) => <tr {...props} />,
            h1: ({node, ...props}) => <h1 {...props} />,
            h2: ({node, ...props}) => <h2 {...props} />,
            h3: ({node, ...props}) => <h3 {...props} />,
            h4: ({node, ...props}) => <h4 {...props} />,
            p:  ({node, ...props}) => <p {...props} />,
            ul: ({node, ...props}) => <ul {...props} />,
            ol: ({node, ...props}) => <ol {...props} />,
            li: ({node, ...props}) => <li {...props} />,
            strong: ({node, ...props}) => <strong {...props} />,
            em: ({node, ...props}) => <em {...props} />,
            blockquote: ({node, ...props}) => <blockquote {...props} />,
            code: ({node, inline, className, children, ...props}) => (
              inline
                ? <code className={className} {...props}>{children}</code>
                : <pre><code className={className} {...props}>{children}</code></pre>
            )
          }}
        >
          {msg.content}
        </ReactMarkdown>

        {msg.conflicts?.map((conflict, i) => (
          <div key={i} className="conflict-warning">
            {conflict}
          </div>
        ))}

        {msg.singleStudy?.map((note, i) => (
          <div key={i} className="single-study-note">
            {note}
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="app-container">
      {showAdmin && <AdminPanel onClose={() => setShowAdmin(false)} />}
      
      {sidebarOpen && (
        <div className="sidebar">
          <h2>🔬 Knowledge Graph</h2>
          <p style={{color: '#64748b', fontSize: '14px', marginBottom: '16px'}}>
            155 entities • 1,014 relations • 87 conflicts
          </p>
          
          {selectedSource && (
            <div style={{marginTop: '16px', padding: '12px', background: '#f8fafc', borderRadius: '8px'}}>
              <h4 style={{marginBottom: '8px'}}>Selected Source</h4>
              <p style={{fontSize: '14px'}}><strong>{selectedSource.title}</strong></p>
              <p style={{fontSize: '12px', color: '#64748b'}}>
                {selectedSource.authors?.join(', ')} • {selectedSource.year}
              </p>
              <button 
                style={{marginTop: '8px', fontSize: '12px', padding: '6px 12px'}}
                onClick={() => setSelectedSource(null)}
              >
                Close
              </button>
            </div>
          )}
          
          <div style={{marginTop: '24px'}}>
            <h4 style={{marginBottom: '12px'}}>Top Entities</h4>
            <span className="entity-tag quantity">NOx (316)</span>
            <span className="entity-tag quantity">CO (316)</span>
            <span className="entity-tag method">LES (315)</span>
            <span className="entity-tag method">RANS (302)</span>
            <span className="entity-tag component">Swirl (289)</span>
            <span className="entity-tag phenomenon">Flashback (156)</span>
          </div>
          
          <div style={{marginTop: '24px'}}>
            <button 
                onClick={() => setShowAdmin(true)}
                style={{width: '100%', padding: '12px', background: '#f59e0b', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: '500'}}
              >
                ⚙️ Admin Panel
              </button>
          </div>
        </div>
      )}
      
      <div className="main-content">
        <div className="header">
          <button 
            style={{background: 'transparent', color: '#64748b', padding: '8px'}}
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            ☰
          </button>
          <h1>🔥 Gas Turbine Combustion Expert</h1>
          <span className="badge">RAG System</span>
        </div>
        
        <div className="chat-container">
          {messages.length === 0 && (
            <div style={{textAlign: 'center', padding: '60px 20px', color: '#64748b'}}>
              <h2 style={{marginBottom: '16px'}}>Welcome to the Gas Turbine Combustion Expert</h2>
              <p>Ask me anything about gas turbine combustion, NOx emissions, swirl flames, or combustion instabilities.</p>
              <p style={{marginTop: '16px', fontSize: '14px'}}>Based on 386 scientific research papers, textbooks and technical documents with knowledge graph analysis.</p>
            </div>
          )}
          
          {messages.map((msg, idx) => renderMessage(msg, idx))}
          
          {loading && (
            <div className="message assistant">
              <div className="loading">
                <span></span><span></span><span></span>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>
        
        <div className="input-container">
          <form onSubmit={sendMessage} className="input-wrapper">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about combustion, emissions, swirl dynamics..."
              disabled={loading}
            />
            <button type="submit" disabled={loading || !input.trim()}>
              {loading ? 'Thinking...' : 'Send'}
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

export default App
