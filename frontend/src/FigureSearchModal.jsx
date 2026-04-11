
import React, { useState, useEffect } from 'react'

function FigureSearchModal({ onClose }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [selectedFigure, setSelectedFigure] = useState(null)
  const [error, setError] = useState(null)

  const searchFigures = async (searchQuery) => {
    if (!searchQuery.trim()) return
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch('/api/search_figures', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery, limit: 20 })
      })
      
      if (!response.ok) {
        throw new Error('Search failed')
      }
      
      const data = await response.json()
      setResults(data.results || [])
    } catch (err) {
      setError('Failed to search figures. Please try again.')
      console.error(err)
    }
    setLoading(false)
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    searchFigures(query)
  }

  const handleFigureClick = (figure) => {
    setSelectedFigure(figure)
  }

  // Quick search suggestions
  const quickSearches = [
    'combustor schematic',
    'NOx emissions graph',
    'swirl stabilized flame',
    'velocity flow field',
    'temperature distribution',
    'fuel injector design',
    'combustion instability'
  ]

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content figure-search-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>📊 Figure Search</h2>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
        
        <div className="modal-body">
          <form onSubmit={handleSubmit} className="search-form">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Describe the figure you're looking for..."
              className="search-input"
              autoFocus
            />
            <button type="submit" disabled={loading} className="search-btn">
              {loading ? 'Searching...' : 'Search'}
            </button>
          </form>

          {error && <div className="error-message">{error}</div>}

          <div className="quick-searches">
            <span className="quick-label">Quick search:</span>
            {quickSearches.map((qs, i) => (
              <button
                key={i}
                onClick={() => { setQuery(qs); searchFigures(qs); }}
                className="quick-btn"
              >
                {qs}
              </button>
            ))}
          </div>

          {results.length > 0 && (
            <div className="results-count">
              Found {results.length} matching figures
            </div>
          )}

          <div className="results-grid">
            {results.map((fig, idx) => (
              <div 
                key={idx} 
                className="figure-card"
                onClick={() => handleFigureClick(fig)}
              >
                <div className="figure-image-container">
                  <img 
                    src={`/api/figures/${fig.figure_id}`}
                    alt={fig.description || 'Figure'}
                    loading="lazy"
                    onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'flex'; }}
                  />
                  <div className="image-placeholder" style={{display: 'none'}}>
                    <span>No preview</span>
                  </div>
                </div>
                <div className="figure-info">
                  <div className="figure-source">
                    📄 {fig.source_pdf}
                  </div>
                  <div className="figure-page">
                    Page {fig.page_number + 1}
                  </div>
                  <div className="figure-score">
                    Match: {(fig.score * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            ))}
          </div>

          {loading && results.length === 0 && (
            <div className="loading-message">
              <div className="spinner"></div>
              <span>Searching figures...</span>
            </div>
          )}
        </div>

        {selectedFigure && (
          <div className="figure-detail-overlay" onClick={() => setSelectedFigure(null)}>
            <div className="figure-detail" onClick={(e) => e.stopPropagation()}>
              <button className="close-detail" onClick={() => setSelectedFigure(null)}>×</button>
              <img 
                src={`/api/figures/${selectedFigure.figure_id}`}
                alt={selectedFigure.description || 'Figure'}
                className="detail-image"
              />
              <div className="detail-info">
                <h3>Figure Details</h3>
                <p><strong>Source:</strong> {selectedFigure.source_pdf}</p>
                <p><strong>Page:</strong> {selectedFigure.page_number + 1}</p>
                {selectedFigure.description && (
                  <div className="detail-description">
                    <strong>Description:</strong>
                    <p>{selectedFigure.description}</p>
                  </div>
                )}
                {selectedFigure.vision_chunk && (
                  <div className="vision-chunk">
                    <strong>Context:</strong>
                    <p>{selectedFigure.vision_chunk.text?.substring(0, 500)}...</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default FigureSearchModal
