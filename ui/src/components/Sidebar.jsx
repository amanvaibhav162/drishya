import React from 'react';
import {
  Home,
  FileText,
  Users,
  BarChart2,
  Settings,
  HelpCircle,
  Building2,
  X
} from 'lucide-react';
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
        <div className="sidebar-header" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px 18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <img
              src="/assets/drishyalogo.jpeg"
              alt="DRISHYA Logo"
              style={{
                height: '34px',
                width: 'auto',
                objectFit: 'contain',
                borderRadius: '6px',
              }}
            />
            <div>
              <h1 className="brand-title" style={{ fontSize: '18px', fontWeight: 900, margin: 0, lineHeight: 1.1, color: '#1E293B' }}>दृष्य</h1>
              <p className="brand-subtitle" style={{ margin: 0, fontSize: '10px', color: '#64748B', fontWeight: 600 }}>{t('brand_subtitle', 'Retinal Tele-Screening')}</p>
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
        <nav className="sidebar-nav-exact">
          {/* 1. Health Worker Portal (Active) */}
          <button
            id="btn-health-worker-mode"
            className={`nav-link-exact ${activeMode === 'health-worker' ? 'active' : ''}`}
            onClick={() => handleSelectMode('health-worker')}
            disabled={isProcessing}
          >
            <Home size={18} />
            <span>{t('hw_portal', 'Health Worker Portal')}</span>
          </button>

          {/* 2. Patient Records */}
          <button
            id="btn-patient-records-mode"
            type="button"
            className={`nav-link-exact ${activeMode === 'patient-records' ? 'active' : ''}`}
            onClick={() => handleSelectMode('patient-records')}
          >
            <FileText size={18} />
            <span>{t('nav_patient_records', 'Patient Records')}</span>
          </button>

          {/* 3. Screening Queue */}
          <button
            type="button"
            className="nav-link-exact"
            onClick={() => alert('Screening Queue: 0 waiting patients in PHC queue.')}
          >
            <Users size={18} />
            <span>{t('nav_screening_queue', 'Screening Queue')}</span>
          </button>

          {/* 4. Reports (Allows switching to Judge Inspector Mode) */}
          <button
            id="btn-judge-inspector-mode"
            type="button"
            className={`nav-link-exact ${activeMode === 'judge-inspector' ? 'active' : ''}`}
            onClick={() => handleSelectMode('judge-inspector')}
            title="Inspect AI Tensors & Grad-CAM++ Saliency"
          >
            <BarChart2 size={18} />
            <span>{activeMode === 'judge-inspector' ? t('judge_portal', 'Inspector Mode') : t('nav_reports', 'Reports')}</span>
          </button>

          {/* 5. Settings */}
          <button
            type="button"
            className="nav-link-exact"
            onClick={() => alert('Opening Edge Device Configuration...')}
          >
            <Settings size={18} />
            <span>{t('nav_settings', 'Settings')}</span>
          </button>

          {/* 7. Support & Help */}
          <button
            type="button"
            className="nav-link-exact"
            onClick={() => alert('Connecting to District Tele-Ophthalmology Support Desk...')}
          >
            <HelpCircle size={18} />
            <span>{t('nav_support', 'Support & Help')}</span>
          </button>
        </nav>

        {/* Sidebar Middle Banner Card */}
        <div className="sidebar-community-card">
          <div className="sidebar-retina-icon-circle" />
          <div className="sidebar-community-title">
            {t('banner_sidebar_title', 'Clear Vision Stronger Communities')}
          </div>
          <div className="sidebar-community-sub">
            {t('banner_sidebar_sub', 'Screen Today for a Brighter Tomorrow')}
          </div>
        </div>

        {/* Sidebar Footer Info */}
        <div className="sidebar-footer-exact">
          <div className="sidebar-phc-name">
            <Building2 size={15} style={{ color: '#1E293B', flexShrink: 0 }} />
            <span>{t('phc_center', 'PHC Rampur (Zone 4)')}</span>
          </div>
          <div className="sidebar-telemed-hub">
            {t('telemed_hub', 'Telemedicine Hub: District Hospital')}
          </div>
          <div className="sidebar-offline-indicator">
            <span className="sidebar-offline-dot" />
            <span>{t('offline_sync_text', 'Offline Mode • Data will sync automatically')}</span>
          </div>
        </div>
      </aside>
    </>
  );
}

