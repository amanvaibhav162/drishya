import React from 'react';
import { Smartphone, Sliders, X } from 'lucide-react';
import { useLanguage } from '../context/useLanguage';

export default function Sidebar({
  activeMode,
  setActiveMode,
  isProcessing,
  isOpen = false,
  onClose
}) {
  const { t } = useLanguage();

  const handleSelectMode = (mode) => {
    setActiveMode(mode);
    if (onClose) onClose();
  };

  return (
    <>
      {/* Mobile Drawer Backdrop */}
      <div
        className={`sidebar-backdrop ${isOpen ? 'open' : ''}`}
        onClick={onClose}
        aria-hidden="true"
      />

      <aside className={`sidebar ${isOpen ? 'open' : ''}`} aria-label="Sidebar Navigation">
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
              <h1 className="brand-title" style={{ fontSize: '17px', margin: 0, lineHeight: 1.2 }}>दृष्य</h1>
              <p className="brand-subtitle" style={{ margin: 0, fontSize: '11px' }}>{t('brand_subtitle')}</p>
            </div>
          </div>

          {/* Close button for mobile drawer */}
          <button
            type="button"
            className="sidebar-close-btn"
            onClick={onClose}
            aria-label="Close navigation"
          >
            <X size={18} />
          </button>
        </div>

        {/* Navigation */}
        <nav className="sidebar-nav">
          <button
            id="btn-health-worker-mode"
            className={`nav-button ${activeMode === 'health-worker' ? 'active' : ''}`}
            onClick={() => handleSelectMode('health-worker')}
            disabled={isProcessing}
          >
            <Smartphone size={16} />
            <div>
              <div>{t('hw_portal')}</div>
            </div>
          </button>

          {/* Hidden on screens < 1000px via responsive CSS and filtered on mobile */}
          <button
            id="btn-judge-inspector-mode"
            className={`nav-button nav-item-judge ${activeMode === 'judge-inspector' ? 'active' : ''}`}
            onClick={() => handleSelectMode('judge-inspector')}
            disabled={isProcessing}
          >
            <Sliders size={16} />
            <div>
              <div>{t('judge_portal')}</div>
            </div>
          </button>
        </nav>

        {/* Footer Info */}
        <div className="sidebar-footer">
          <div style={{ fontWeight: 700, marginBottom: '2px' }}>{t('phc_center')}</div>
          <div>{t('telemed_hub')}</div>
          <div className="text-micro" style={{ marginTop: '6px' }}>{t('edge_tag')}</div>
        </div>
      </aside>
    </>
  );
}
