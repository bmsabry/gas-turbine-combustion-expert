import React, { useState, useEffect } from 'react'

function AdminPanel({ onClose }) {
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [loginForm, setLoginForm] = useState({ username: '', password: '' })
  const [settings, setSettings] = useState(null)
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [token, setToken] = useState(localStorage.getItem('admin_token') || '')
  const [passwordForm, setPasswordForm] = useState({ current: '', new: '' })

  useEffect(() => {
    if (token) {
      setIsLoggedIn(true)
      fetchSettings()
      fetchStats()
    }
  }, [token])

  const login = async (e) => {
    e?.preventDefault()
    setLoading(true)
    setError('')
    
    try {
      const res = await fetch('/api/admin/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(loginForm)
      })
      
      if (res.ok) {
        const data = await res.json()
        setToken(data.token)
        localStorage.setItem('admin_token', data.token)
        setIsLoggedIn(true)
        fetchSettings()
        fetchStats()
      } else {
        const data = await res.json()
        setError(data.detail || 'Login failed')
      }
    } catch (err) {
      setError('Connection error')
    }
    setLoading(false)
  }

  const fetchSettings = async () => {
    try {
      const res = await fetch('/api/admin/settings', {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      if (res.ok) {
        const data = await res.json()
        // Merge with existing settings to preserve llm_api_key
        setSettings(prev => ({
          ...data,
          llm_api_key: prev?.llm_api_key || ''
        }))
      }
    } catch (err) {
      console.error('Failed to fetch settings')
    }
  }

  const fetchStats = async () => {
    try {
      const res = await fetch('/api/admin/stats', {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      if (res.ok) {
        const data = await res.json()
        setStats(data)
      }
    } catch (err) {
      console.error('Failed to fetch stats')
    }
  }

  const saveSettings = async (e) => {
    e?.preventDefault()
    setLoading(true)
    
    try {
      const res = await fetch('/api/admin/settings', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(settings)
      })
      
      if (res.ok) {
        alert('Settings saved successfully!')
      } else {
        setError('Failed to save settings')
      }
    } catch (err) {
      setError('Connection error')
    }
    setLoading(false)
  }

  const changePassword = async (e) => {
    e?.preventDefault()
    setLoading(true)
    
    try {
      const res = await fetch('/api/admin/change-password', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          current_password: passwordForm.current,
          new_password: passwordForm.new
        })
      })
      
      if (res.ok) {
        alert('Password changed successfully!')
        setPasswordForm({ current: '', new: '' })
      } else {
        const data = await res.json()
        setError(data.detail || 'Failed to change password')
      }
    } catch (err) {
      setError('Connection error')
    }
    setLoading(false)
  }

  const logout = () => {
    localStorage.removeItem('admin_token')
    setToken('')
    setIsLoggedIn(false)
    setSettings(null)
    setStats(null)
  }

  if (!isLoggedIn) {
    return (
      <div style={{
        position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
        background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000
      }}>
        <div style={{
          background: 'white', padding: '32px', borderRadius: '16px', width: '400px', maxWidth: '90%'
        }}>
          <h2 style={{marginBottom: '24px'}}>🔐 Admin Login</h2>
          {error && <div style={{color: 'red', marginBottom: '16px'}}>{error}</div>}
          <form onSubmit={login}>
            <div style={{marginBottom: '16px'}}>
              <label style={{display: 'block', marginBottom: '4px'}}>Username</label>
              <input
                type="text"
                value={loginForm.username}
                onChange={(e) => setLoginForm({...loginForm, username: e.target.value})}
                style={{width: '100%', padding: '10px', border: '1px solid #ddd', borderRadius: '6px'}}
              />
            </div>
            <div style={{marginBottom: '16px'}}>
              <label style={{display: 'block', marginBottom: '4px'}}>Password</label>
              <input
                type="password"
                value={loginForm.password}
                onChange={(e) => setLoginForm({...loginForm, password: e.target.value})}
                style={{width: '100%', padding: '10px', border: '1px solid #ddd', borderRadius: '6px'}}
              />
            </div>
            <div style={{display: 'flex', gap: '12px'}}>
              <button type="submit" disabled={loading} style={{flex: 1, padding: '12px', background: '#2563eb', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer'}}>
                {loading ? 'Logging in...' : 'Login'}
              </button>
              <button type="button" onClick={onClose} style={{padding: '12px 24px', background: '#ef4444', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer'}}>
                Cancel
              </button>
            </div>
          </form>
          <p style={{marginTop: '16px', fontSize: '12px', color: '#666'}}>
            Default credentials: admin / admin123
          </p>
        </div>
      </div>
    )
  }

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: '#f8fafc', zIndex: 1000, overflow: 'auto'
    }}>
      <div style={{maxWidth: '1200px', margin: '0 auto', padding: '24px'}}>
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px'}}>
          <h1>⚙️ Admin Dashboard</h1>
          <div style={{display: 'flex', gap: '12px'}}>
            <button onClick={logout} style={{padding: '10px 20px', background: '#ef4444', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer'}}>
              Logout
            </button>
            <button onClick={onClose} style={{padding: '10px 20px', background: '#64748b', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer'}}>
              Close
            </button>
          </div>
        </div>

        {error && <div style={{background: '#fef2f2', color: '#ef4444', padding: '12px', borderRadius: '8px', marginBottom: '16px'}}>{error}</div>}

        {/* Stats Section */}
        {stats && (
          <div style={{background: 'white', padding: '24px', borderRadius: '12px', marginBottom: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)'}}>
            <h2 style={{marginBottom: '16px'}}>📊 System Statistics</h2>
            <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '16px'}}>
              <div style={{textAlign: 'center', padding: '16px', background: '#f8fafc', borderRadius: '8px'}}>
                <div style={{fontSize: '32px', fontWeight: 'bold', color: '#2563eb'}}>{stats.papers_processed}</div>
                <div style={{color: '#64748b'}}>Papers</div>
              </div>
              <div style={{textAlign: 'center', padding: '16px', background: '#f8fafc', borderRadius: '8px'}}>
                <div style={{fontSize: '32px', fontWeight: 'bold', color: '#10b981'}}>{stats.chunks}</div>
                <div style={{color: '#64748b'}}>Chunks</div>
              </div>
              <div style={{textAlign: 'center', padding: '16px', background: '#f8fafc', borderRadius: '8px'}}>
                <div style={{fontSize: '32px', fontWeight: 'bold', color: '#f59e0b'}}>{stats.entities}</div>
                <div style={{color: '#64748b'}}>Entities</div>
              </div>
              <div style={{textAlign: 'center', padding: '16px', background: '#f8fafc', borderRadius: '8px'}}>
                <div style={{fontSize: '32px', fontWeight: 'bold', color: '#ef4444'}}>{stats.relationships}</div>
                <div style={{color: '#64748b'}}>Relationships</div>
              </div>
              <div style={{textAlign: 'center', padding: '16px', background: '#f8fafc', borderRadius: '8px'}}>
                <div style={{fontSize: '32px', fontWeight: 'bold', color: '#8b5cf6'}}>{stats.contradictions}</div>
                <div style={{color: '#64748b'}}>Conflicts</div>
              </div>
            </div>
          </div>
        )}

        {/* LLM Settings Section */}
        {settings && (
          <div style={{background: 'white', padding: '24px', borderRadius: '12px', marginBottom: '24px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)'}}>
            <h2 style={{marginBottom: '16px'}}>🤖 LLM Configuration</h2>
            <form onSubmit={saveSettings}>
              <div style={{display: 'grid', gap: '16px'}}>
                <div>
                  <label style={{display: 'block', marginBottom: '4px', fontWeight: '500'}}>Provider</label>
                  <select
                    value={settings.llm_provider}
                    onChange={(e) => setSettings({...settings, llm_provider: e.target.value})}
                    style={{width: '100%', padding: '10px', border: '1px solid #ddd', borderRadius: '6px'}}
                  >
                    <option value="anthropic">Anthropic (Claude)</option>
                    <option value="openai">OpenAI (GPT)</option>
                    <option value="openrouter">OpenRouter (Multi)</option>
                  </select>
                </div>
                
                <div>
                  <label style={{display: 'block', marginBottom: '4px', fontWeight: '500'}}>API URL</label>
                  <input
                    type="text"
                    value={settings.llm_api_url}
                    onChange={(e) => setSettings({...settings, llm_api_url: e.target.value})}
                    style={{width: '100%', padding: '10px', border: '1px solid #ddd', borderRadius: '6px'}}
                    placeholder="https://api.anthropic.com"
                  />
                </div>
                
                <div>
                  <label style={{display: 'block', marginBottom: '4px', fontWeight: '500'}}>API Key</label>
                  <input
                    type="password"
                    value={settings.llm_api_key}
                    onChange={(e) => setSettings({...settings, llm_api_key: e.target.value})}
                    style={{width: '100%', padding: '10px', border: '1px solid #ddd', borderRadius: '6px'}}
                    placeholder="Enter your API key"
                  />
                </div>
                
                <div>
                  <label style={{display: 'block', marginBottom: '4px', fontWeight: '500'}}>Model</label>
                  <input
                    type="text"
                    value={settings.llm_model}
                    onChange={(e) => setSettings({...settings, llm_model: e.target.value})}
                    style={{width: '100%', padding: '10px', border: '1px solid #ddd', borderRadius: '6px'}}
                    placeholder="claude-sonnet-4-6"
                  />
                </div>
              </div>
              
              <button type="submit" disabled={loading} style={{marginTop: '16px', padding: '12px 24px', background: '#10b981', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer'}}>
                {loading ? 'Saving...' : '💾 Save Settings'}
              </button>
            </form>
          </div>
        )}

        {/* Change Password Section */}
        <div style={{background: 'white', padding: '24px', borderRadius: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)'}}>
          <h2 style={{marginBottom: '16px'}}>🔑 Change Password</h2>
          <form onSubmit={changePassword}>
            <div style={{display: 'grid', gap: '16px'}}>
              <div>
                <label style={{display: 'block', marginBottom: '4px', fontWeight: '500'}}>Current Password</label>
                <input
                  type="password"
                  value={passwordForm.current}
                  onChange={(e) => setPasswordForm({...passwordForm, current: e.target.value})}
                  style={{width: '100%', padding: '10px', border: '1px solid #ddd', borderRadius: '6px'}}
                />
              </div>
              <div>
                <label style={{display: 'block', marginBottom: '4px', fontWeight: '500'}}>New Password</label>
                <input
                  type="password"
                  value={passwordForm.new}
                  onChange={(e) => setPasswordForm({...passwordForm, new: e.target.value})}
                  style={{width: '100%', padding: '10px', border: '1px solid #ddd', borderRadius: '6px'}}
                />
              </div>
            </div>
            <button type="submit" disabled={loading} style={{marginTop: '16px', padding: '12px 24px', background: '#f59e0b', color: 'white', border: 'none', borderRadius: '6px', cursor: 'pointer'}}>
              {loading ? 'Updating...' : '🔄 Change Password'}
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

export default AdminPanel
