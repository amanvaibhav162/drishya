import React from 'react';
import { Smartphone, Sliders } from 'lucide-react';

export default function Sidebar({ activeMode, setActiveMode, isProcessing }) {
  return (
    <aside className="sidebar">
      {/* Brand Header */}
      <div className="sidebar-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <img
            src="/assets/drishyalogo.jpeg"
            alt="DRISHYA Logo"
            style={{
              height: '38px',
              maxWidth: '68px',
              objectFit: 'contain',
              borderRadius: '6px',
            }}
          />
          <div>
            <h1 className="brand-title" style={{ fontSize: '17px', margin: 0, lineHeight: 1.2 }}>DRISHYA (दृष्य)</h1>
            <p className="brand-subtitle" style={{ margin: 0, fontSize: '11px' }}>Retinal Tele-Screening</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">

        <button
          id="btn-health-worker-mode"
          className={`nav-button ${activeMode === 'health-worker' ? 'active' : ''}`}
          onClick={() => setActiveMode('health-worker')}
          disabled={isProcessing}
        >
          <Smartphone size={16} />
          <div>
            <div>Health Worker Portal</div>

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
          </div>
        </button>
      </nav>

      {/* Footer Info */}
      <div className="sidebar-footer">
        <div style={{ fontWeight: 700, marginBottom: '2px' }}>PHC Rampur (Zone 4)</div>
        <div>Telemedicine Hub: District Hospital</div>
        <div className="text-micro" style={{ marginTop: '6px' }}>FP16 Model • Offline Edge Ready</div>
      </div>
    </aside>
  );
}
