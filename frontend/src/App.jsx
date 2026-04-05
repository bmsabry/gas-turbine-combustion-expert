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
              <div style={{overflowX: 'auto', margin: '12px 0'}}>
                <table style={{
                  borderCollapse: 'collapse',
                  width: '100%',
                  fontSize: '14px'
                }} {...props} />
              </div>
            ),
            th: ({node, ...props}) => (
              <th style={{
                background: '#1e3a5f',
                color: 'white',
                padding: '8px 12px',
                textAlign: 'left',
                fontWeight: '600',
                borderBottom: '2px solid #2d5a8e'
              }} {...props} />
            ),
            td: ({node, ...props}) => (
              <td style={{
                padding: '8px 12px',
                borderBottom: '1px solid #e2e8f0',
                verticalAlign: 'top'
              }} {...props} />
            ),
            tr: ({node, ...props}) => (
              <tr style={{
                background: 'inherit'
              }} {...props} />
            ),
            h3: ({node, ...props}) => (
              <h3 style={{
                color: '#1e3a5f',
                marginTop: '20px',
                marginBottom: '8px',
                fontSize: '16px',
                fontWeight: '700',
                borderBottom: '1px solid #e2e8f0',
                paddingBottom: '4px'
              }} {...props} />
            ),
            h4: ({node, ...props}) => (
              <h4 style={{
                color: '#2d5a8e',
                marginTop: '16px',
                marginBottom: '6px',
                fontSize: '14px',
                fontWeight: '600'
              }} {...props} />
            ),
            h2: ({node, ...props}) => (
              <h2 style={{
                color: '#1e3a5f',
                marginTop: '24px',
                marginBottom: '10px',
                fontSize: '18px',
                fontWeight: '700'
              }} {...props} />
            ),
            p: ({node, ...props}) => (
              <p style={{
                marginBottom: '10px',
                lineHeight: '1.7'
              }} {...props} />
            ),
            ul: ({node, ...props}) => (
              <ul style={{
                paddingLeft: '20px',
                marginBottom: '10px'
              }} {...props} />
            ),
            ol: ({node, ...props}) => (
              <ol style={{
                paddingLeft: '20px',
                marginBottom: '10px'
              }} {...props} />
            ),
            li: ({node, ...props}) => (
              <li style={{
                marginBottom: '4px',
                lineHeight: '1.6'
              }} {...props} />
            ),
            code: ({node, inline, ...props}) => (
              inline
                ? <code style={{
                    background: '#f1f5f9',
                    padding: '1px 4px',
                    borderRadius: '3px',
                    fontFamily: 'monospace',
                    fontSize: '13px',
                    color: '#1e3a5f'
                  }} {...props} />
                : <code style={{
                    display: 'block',
                    background: '#f1f5f9',
                    padding: '12px',
                    borderRadius: '6px',
                    fontFamily: 'monospace',
                    fontSize: '13px',
                    overflowX: 'auto',
                    margin: '8px 0'
                  }} {...props} />
            ),
            blockquote: ({node, ...props}) => (
              <blockquote style={{
                borderLeft: '4px solid #2d5a8e',
                paddingLeft: '12px',
                margin: '10px 0',
                color: '#475569',
                fontStyle: 'italic'
              }} {...props} />
            ),
            strong: ({node, ...props}) => (
              <strong style={{ color: '#1e3a5f' }} {...props} />
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
