import React from 'react';
import { Smartphone, Eye, CheckCircle2, AlertTriangle, XCircle, Sliders, Activity } from 'lucide-react';

export default function Sidebar({ activeMode, setActiveMode, currentPreset, setPreset, isProcessing }) {
  return (
    <aside className="sidebar">
      {/* Brand Header */}
      <div className="sidebar-header">
        <span className="brand-badge">DRISHYA • SIH26038</span>
        <h1 className="brand-title">DRISHYA (दृष्य)</h1>
        <p className="brand-subtitle">AI Retinal Tele-Screening & Triage</p>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        <div className="nav-label">Operating Modes</div>
        
        <button
          id="btn-health-worker-mode"
          className={`nav-button ${activeMode === 'health-worker' ? 'active' : ''}`}
          onClick={() => setActiveMode('health-worker')}
          disabled={isProcessing}
        >
          <Smartphone size={16} />
          <div>
            <div>Health Worker Portal</div>
            <div style={{ fontSize: '10px', opacity: 0.8 }}>3-Tap ASHA Screening</div>
          </div>
        </button>

        <button
          id="btn-judge-inspector-mode"
          className={`nav-button ${activeMode === 'judge-inspector' ? 'active' : ''}`}
          onClick={() => setActiveMode('judge-inspector')}
          disabled={isProcessing}
        >
          <Sliders size={16} />
          <div>
            <div>Judge Pipeline Inspector</div>
            <div style={{ fontSize: '10px', opacity: 0.8 }}>Visual Multi-Stage AI</div>
          </div>
        </button>

        <div className="nav-label" style={{ marginTop: '20px' }}>Demo Quick-Load Presets</div>

        <button
          id="preset-grade2"
          className={`nav-button preset-btn ${currentPreset === 'grade2' ? 'active' : ''}`}
          onClick={() => setPreset('grade2')}
          disabled={isProcessing}
        >
          <AlertTriangle size={14} color="#D97706" />
          <span>Grade 2: Moderate NPDR (Real)</span>
        </button>

        <button
          id="preset-grade0"
          className={`nav-button preset-btn ${currentPreset === 'grade0' ? 'active' : ''}`}
          onClick={() => setPreset('grade0')}
          disabled={isProcessing}
        >
          <CheckCircle2 size={14} color="#16A34A" />
          <span>Grade 0: Normal Retina</span>
        </button>

        <button
          id="preset-grade3"
          className={`nav-button preset-btn ${currentPreset === 'grade3' ? 'active' : ''}`}
          onClick={() => setPreset('grade3')}
          disabled={isProcessing}
        >
          <AlertTriangle size={14} color="#DC2626" />
          <span>Grade 3: Severe NPDR</span>
        </button>

        <button
          id="preset-ungradable"
          className={`nav-button preset-btn ${currentPreset === 'ungradable' ? 'active' : ''}`}
          onClick={() => setPreset('ungradable')}
          disabled={isProcessing}
        >
          <XCircle size={14} color="#DC2626" />
          <span>Blurry Scan (IQA Retake Demo)</span>
        </button>
      </nav>

      {/* Footer Info */}
      <div className="sidebar-footer">
        <div style={{ fontWeight: 700, marginBottom: '2px' }}>PHC Rampur (Zone 4)</div>
        <div>Telemedicine Hub: District Hospital</div>
        <div style={{ marginTop: '6px', fontSize: '10px' }}>FP16 Model • Offline Edge Ready</div>
      </div>
    </aside>
  );
}
