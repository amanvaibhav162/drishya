import React, { useRef, useState, useEffect } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  FileText,
  Download,
  Send,
  RefreshCw,
  User,
  Phone,
  Calendar,
  Upload,
  RotateCcw,
  ArrowRight,
  Activity,
  CreditCard,
  Lightbulb,
  Sparkles
} from 'lucide-react';
import { useLanguage } from '../context/useLanguage';

export default function HealthWorkerMode({
  patientInfo,
  setPatientInfo,
  uploadedImage,
  onImageSelected,
  onClearImage,
  screeningResult,
  onRunScreening,
  isProcessing,
  currentStep = 1,
  onOpenPdfModal,
  onNavigateToRecords
}) {
  const { t } = useLanguage();
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);

  // Real screening records pulled from Supabase backend storage
  const [recentPatients, setRecentPatients] = useState([]);
  const [isLoadingPatients, setIsLoadingPatients] = useState(true);

  const fetchRecentScreenings = async () => {
    setIsLoadingPatients(true);
    try {
      let res = await fetch('/api/screenings?limit=8');
      if (!res.ok && (res.status === 404 || res.status === 502)) {
        res = await fetch('http://localhost:8000/api/screenings?limit=8');
      }
      if (res.ok) {
        const data = await res.json();
        setRecentPatients(Array.isArray(data) ? data : []);
      }
    } catch (err) {
      console.error('Error fetching screenings from Supabase backend:', err);
    } finally {
      setIsLoadingPatients(false);
    }
  };

  useEffect(() => {
    fetchRecentScreenings();
  }, []);

  // Whenever a new patient screening finishes, auto-refresh the list from Supabase
  useEffect(() => {
    if (screeningResult) {
      fetchRecentScreenings();
    }
  }, [screeningResult]);

  const getInitials = (name) => {
    if (!name || name.toLowerCase().includes('anonymous')) return 'PT';
    const parts = name.trim().split(/\s+/);
    if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  };

  const formatDate = (isoString) => {
    if (!isoString) return 'Recent';
    try {
      const d = new Date(isoString);
      if (isNaN(d.getTime())) return isoString.slice(0, 10);
      return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
    } catch {
      return 'Recent';
    }
  };

  const handleSelectRecentPatient = (p) => {
    setPatientInfo({
      name: p.patient_name || '',
      age: p.patient_age && p.patient_age !== 'N/A' ? p.patient_age : '',
      gender: p.patient_gender && p.patient_gender !== 'N/A' ? p.patient_gender : 'Male',
      phone: p.patient_phone && p.patient_phone !== 'N/A' ? p.patient_phone : '',
      abhaId: p.abha_id && p.abha_id !== 'N/A' ? p.abha_id : ''
    });
  };

  const handleClearAll = () => {
    setPatientInfo({
      name: '',
      age: '',
      gender: 'Male',
      phone: '',
      abhaId: ''
    });
    onClearImage();
  };

  const pipelineStages = [
    {
      step: 1,
      title: t('stage_1_title', 'Image Ingestion & Edge IQA'),
      subtext: t('stage_1_sub', 'Verifying optical focus, retinal illumination, and field-of-view...'),
      pct: 18,
    },
    {
      step: 2,
      title: t('stage_2_title', 'Normalization & Preprocessing'),
      subtext: t('stage_2_sub', 'Applying CLAHE adaptive histogram and green-channel contrast enhancement...'),
      pct: 38,
    },
    {
      step: 3,
      title: t('stage_3_title', 'EfficientNetV2 Deep Multi-Task Inference'),
      subtext: t('stage_3_sub', 'Evaluating ICDR severity grading & referable DR probability...'),
      pct: 65,
    },
    {
      step: 4,
      title: t('stage_4_title', 'Visual AI Evidence & Attention Map'),
      subtext: t('stage_4_sub', 'Generating Grad-CAM++ neural activation map & localizing microaneurysms...'),
      pct: 85,
    },
    {
      step: 5,
      title: t('stage_5_title', 'Compiling Certified Clinical Report'),
      subtext: t('stage_5_sub', 'Synthesizing biomarker findings and generating audit-ready PDF...'),
      pct: 95,
    },
    {
      step: 6,
      title: t('stage_6_title', 'Diagnostic Analysis Complete'),
      subtext: t('stage_6_sub', 'Finalizing clinical findings and triage delivery...'),
      pct: 100,
    },
  ];

  const handleLoadSampleScan = async (url = '/fundus_image.png', filename = 'fundus_image.png', e) => {
    if (e && e.stopPropagation) e.stopPropagation();
    try {
      const response = await fetch(url);
      const blob = await response.blob();
      const file = new File([blob], filename, { type: 'image/jpeg' });
      onImageSelected(file, url, filename);
    } catch (err) {
      console.error('Failed to load sample scan', err);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      const previewUrl = URL.createObjectURL(file);
      onImageSelected(file, previewUrl, file.name);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      const previewUrl = URL.createObjectURL(file);
      onImageSelected(file, previewUrl, file.name);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const isUngradable = screeningResult && !screeningResult.iqaPass;
  const activeStage = pipelineStages[Math.min(Math.max((currentStep || 1) - 1, 0), pipelineStages.length - 1)];

  // Localized grade title & recommendation
  const localizedGradeTitle = screeningResult?.grade !== undefined && screeningResult?.grade >= 0 && screeningResult?.grade <= 4
    ? t(`grade_${screeningResult.grade}_title`, screeningResult.gradeTitle)
    : (screeningResult?.gradeTitle || '');

  const localizedAction = screeningResult?.referable
    ? t('action_referral', screeningResult?.actionRecommendation)
    : t('action_routine', screeningResult?.actionRecommendation);

  return (
    <div className="portal-layout-grid">
      {/* ── Left Column: Primary Clinical Screening Flow ──────────────────────── */}
      <div className="portal-main-flow">
        {/* Card 1: Patient Registration & Tagging */}
        <div className="card-step">
          <div className="card-step-header">
            <div className="step-title-wrapper">
              <div className="step-number-circle">1</div>
              <div>
                <h2 className="step-main-title">
                  {t('step1_title', 'Patient Registration & Tagging')}
                </h2>
                <p className="step-main-subtitle">
                  {t('step1_sub', 'Enter patient details to create a new screening record')}
                </p>
              </div>
            </div>
          </div>

          {/* Form Row 1: Full Name & Phone Number */}
          <div className="portal-form-row-1">
            <div>
              <label className="portal-input-label" htmlFor="input-patient-name">
                <User size={13} style={{ color: '#475569' }} />
                <span>{t('patient_name', 'Patient Full Name')}</span>
                <span className="portal-req-star">*</span>
              </label>
              <input
                id="input-patient-name"
                type="text"
                className="portal-form-control"
                value={patientInfo.name}
                onChange={(e) => setPatientInfo({ ...patientInfo, name: e.target.value })}
                placeholder={t('patient_name_placeholder', 'e.g. Ramesh Kumar')}
              />
            </div>

            <div>
              <label className="portal-input-label" htmlFor="input-patient-phone">
                <Phone size={13} style={{ color: '#475569' }} />
                <span>{t('mobile_num', 'Mobile Number')}</span>
              </label>
              <input
                id="input-patient-phone"
                type="text"
                className="portal-form-control"
                value={patientInfo.phone}
                onChange={(e) => setPatientInfo({ ...patientInfo, phone: e.target.value })}
                placeholder={t('mobile_placeholder', '+91 98765 43210')}
              />
            </div>
          </div>

          {/* Form Row 2: Age, Gender, ABHA ID */}
          <div className="portal-form-row-2">
            <div>
              <label className="portal-input-label" htmlFor="input-patient-age">
                <Calendar size={13} style={{ color: '#475569' }} />
                <span>{t('age', 'Age')}</span>
                <span className="portal-req-star">*</span>
              </label>
              <input
                id="input-patient-age"
                type="number"
                min="1"
                max="120"
                className="portal-form-control"
                value={patientInfo.age}
                onChange={(e) => setPatientInfo({ ...patientInfo, age: e.target.value })}
                placeholder={t('age_placeholder', 'e.g. 52')}
              />
            </div>

            <div>
              <label className="portal-input-label" htmlFor="input-patient-gender">
                <User size={13} style={{ color: '#475569' }} />
                <span>{t('gender', 'Gender')}</span>
                <span className="portal-req-star">*</span>
              </label>
              <select
                id="input-patient-gender"
                className="portal-form-control"
                value={patientInfo.gender || 'Male'}
                onChange={(e) => setPatientInfo({ ...patientInfo, gender: e.target.value })}
                style={{ cursor: 'pointer' }}
              >
                <option value="Male">{t('gender_male', 'Male')}</option>
                <option value="Female">{t('gender_female', 'Female')}</option>
                <option value="Other">{t('gender_other', 'Other')}</option>
              </select>
            </div>

            <div>
              <label className="portal-input-label" htmlFor="input-patient-abha">
                <CreditCard size={13} style={{ color: '#475569' }} />
                <span>{t('abha_id', 'ABHA Health ID (Ayushman Bharat)')}</span>
              </label>
              <input
                id="input-patient-abha"
                type="text"
                className="portal-form-control"
                value={patientInfo.abhaId}
                onChange={(e) => setPatientInfo({ ...patientInfo, abhaId: e.target.value })}
                placeholder={t('abha_placeholder', 'e.g. 14-8921-3401-9210')}
              />
            </div>
          </div>
        </div>

        {/* Card 2: Retinal Fundus Scan Ingestion */}
        <div className="card-step">
          <div className="card-step-header">
            <div className="step-title-wrapper">
              <div className="step-number-circle">2</div>
              <div>
                <h2 className="step-main-title">
                  {t('step2_title', 'Retinal Fundus Scan Ingestion')}
                </h2>
                <p className="step-main-subtitle">
                  {t('step2_sub', 'Upload the retinal fundus image captured using the device')}
                </p>
              </div>
            </div>
          </div>


          {/* Upload Dropzone (When no image is loaded) */}
          {!uploadedImage?.previewUrl ? (
            <div>
              <input
                type="file"
                ref={fileInputRef}
                accept="image/*"
                style={{ display: 'none' }}
                onChange={handleFileChange}
              />
              <div
                className={`portal-dropzone-exact ${isDragOver ? 'drag-over' : ''}`}
                onClick={() => fileInputRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    fileInputRef.current?.click();
                  }
                }}
              >
                <div className="portal-upload-circle">
                  <Upload size={22} />
                </div>
                <div className="portal-drop-title">
                  {t('dropzone_title', 'Drag & drop retinal fundus photo here')}{' '}
                  <span className="portal-browse-link">{t('click_to_browse', 'or click to browse')}</span>
                </div>
                <div className="portal-drop-hint">
                  {t('dropzone_subtitle', 'Supports high-resolution PNG, JPG, JPEG (min. 512×512)')}
                </div>

                <div style={{ marginTop: '12px', display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center' }}>
                  <button
                    id="btn-sample-normal"
                    type="button"
                    className="sample-scan-pill"
                    onClick={(e) => handleLoadSampleScan('/assets/odir_1029_left.jpg', 'odir_1029_left.jpg', e)}
                    title="Load normal fundus scan (Grade 0: No DR)"
                  >
                    <Sparkles size={12} />
                    <span>Sample: Normal (Grade 0)</span>
                  </button>

                  <button
                    id="btn-sample-moderate"
                    type="button"
                    className="sample-scan-pill"
                    onClick={(e) => handleLoadSampleScan('/assets/odir_1007_left.jpg', 'odir_1007_left.jpg', e)}
                    title="Load moderate DR scan (Grade 2: Moderate NPDR)"
                  >
                    <Sparkles size={12} />
                    <span>Sample: Moderate DR (Grade 2)</span>
                  </button>

                  <button
                    id="btn-sample-severe"
                    type="button"
                    className="sample-scan-pill"
                    onClick={(e) => handleLoadSampleScan('/fundus_image.png', 'fundus_image.png', e)}
                    title="Load severe DR scan (Grade 3: Severe NPDR)"
                  >
                    <Sparkles size={12} />
                    <span>Sample: Severe NPDR (Grade 3)</span>
                  </button>
                </div>
              </div>
            </div>
          ) : (
            /* Active Retinal Scan Loaded View */
            <div>
              <div className="image-canvas-wrapper" style={{ height: '280px', marginBottom: '14px', borderRadius: '0', overflow: 'hidden' }}>
                <img
                  src={uploadedImage.previewUrl}
                  alt="Fundus Capture"
                  className="retina-img"
                  style={{ width: '100%', height: '100%', objectFit: 'contain', backgroundColor: '#000000' }}
                />
              </div>

              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '14px',
                padding: '8px 12px',
                backgroundColor: '#F8FAFC',
                borderRadius: '0',
                border: '1px solid #E2E8F0'
              }}>
                <span style={{ fontSize: '12px', color: '#475569', fontWeight: 600 }}>
                  📁 {uploadedImage.name || 'Retinal Scan Loaded'}
                </span>
                <button
                  type="button"
                  className="btn-portal-clear"
                  style={{ padding: '4px 10px', fontSize: '11.5px' }}
                  onClick={onClearImage}
                  disabled={isProcessing}
                >
                  <RefreshCw size={12} /> {t('change_photo', 'Change Photo')}
                </button>
              </div>

              {/* Quality Triage Banner (Only after screening has run) */}
              {screeningResult && (
                isUngradable ? (
                  <div style={{
                    backgroundColor: 'var(--status-danger-bg)',
                    border: '1px solid var(--status-danger-border)',
                    borderRadius: '0',
                    padding: '12px 16px',
                    marginBottom: '14px',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '12px'
                  }}>
                    <AlertCircle color="#DC2626" size={24} style={{ flexShrink: 0, marginTop: '2px' }} />
                    <div>
                      <div style={{ color: '#991B1B', fontWeight: 800, fontSize: '14px' }}>
                        {t('iqa_failed_title', 'Ungradable (Quality Rejected)')}
                      </div>
                      <div style={{ color: '#B91C1C', marginTop: '3px', fontSize: '12.5px' }}>
                        {t('iqa_retake_action', '🛑 RETAKE SCAN REQUIRED: Recapture retinal photo immediately before patient leaves.')} (Q = {screeningResult.qScore || screeningResult.iqaScore})
                      </div>
                      <button
                        className="btn btn-danger text-caption"
                        style={{ marginTop: '10px', padding: '6px 14px' }}
                        onClick={onClearImage}
                      >
                        <RefreshCw size={13} /> {t('retake_btn', 'Recapture Retinal Scan')}
                      </button>
                    </div>
                  </div>
                ) : (
                  <div style={{
                    backgroundColor: 'var(--status-pass-bg)',
                    border: '1px solid var(--status-pass-border)',
                    borderRadius: '0',
                    padding: '10px 16px',
                    marginBottom: '14px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px'
                  }}>
                    <CheckCircle2 color="#16A34A" size={20} />
                    <div>
                      <div style={{ color: '#166534', fontWeight: 700, fontSize: '13px' }}>
                        {t('iqa_passed_title', 'Image Quality Passed (IQA)')} (Q = {screeningResult.iqaScore})
                      </div>
                      <div style={{ color: '#15803D', fontSize: '11.5px' }}>
                        {t('iqa_passed_sub', 'Good focus, clarity, and illumination detected for diagnostic classification.')}
                      </div>
                    </div>
                  </div>
                )
              )}
            </div>
          )}

          {/* Action Bar Below Dropzone (Always Visible) */}
          <div className="portal-action-bar">
            <button
              type="button"
              className="btn-portal-clear"
              onClick={handleClearAll}
              disabled={isProcessing}
            >
              <RotateCcw size={14} />
              <span>{t('clear_btn', 'Clear')}</span>
            </button>

            <button
              id="btn-process-scan"
              type="button"
              className="btn-portal-process"
              onClick={onRunScreening}
              disabled={isProcessing}
            >
              <span>{isProcessing ? t('analyzing_btn', 'Processing...') : t('process_scan_btn', 'Process Scan')}</span>
              <ArrowRight size={15} />
            </button>
          </div>

          {/* Active 5-Stage Pipeline Progress Card (During Screening) */}
          {isProcessing && (
            <div className="processing-card" id="screening-progress-card" style={{ marginTop: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <RefreshCw size={15} className="spin-icon" style={{ color: 'var(--brand-primary)' }} />
                  <span style={{ fontSize: '13px', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.2px' }}>
                    {t('analyzing_btn', 'Screening In Progress...')}
                  </span>
                </div>
                <div style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '11px',
                  fontWeight: 700,
                  color: 'var(--text-primary)',
                  backgroundColor: 'var(--bg-subtle)',
                  border: '1px solid var(--border-light)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '2px 8px'
                }}>
                  {activeStage.pct}%
                </div>
              </div>

              {/* Animated Progress Bar Track */}
              <div className="progress-track" style={{ marginBottom: '10px' }}>
                <div
                  className="progress-bar-fill"
                  style={{ width: `${activeStage.pct}%`, backgroundColor: '#BD4319' }}
                >
                  <div className="progress-shimmer" />
                </div>
              </div>

              {/* Dynamic Stage Title and Counter */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '3px' }}>
                <span style={{ fontSize: '12px', fontWeight: 700, color: '#0F172A' }}>
                  {activeStage.title}
                </span>
                <span className="text-micro" style={{ color: 'var(--text-muted)', fontWeight: 600 }}>
                  Stage {Math.min(currentStep || 1, 5)} of 5
                </span>
              </div>

              <div className="text-micro" style={{ color: 'var(--text-secondary)', marginBottom: '12px', minHeight: '18px' }}>
                {activeStage.subtext}
              </div>

              {/* 5-Stage Indicator Blocks */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '4px' }}>
                {[1, 2, 3, 4, 5].map((sNum) => {
                  const isDone = (currentStep || 1) > sNum;
                  const isCurrent = (currentStep || 1) === sNum;
                  return (
                    <div
                      key={sNum}
                      style={{
                        height: '4px',
                        borderRadius: '0',
                        backgroundColor: isDone ? '#BD4319' : (isCurrent ? '#F5A88A' : '#E2E8F0'),
                        transition: 'background-color 0.3s ease'
                      }}
                    />
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Result Card (When Screened & Passed IQA) */}
        {screeningResult && !isUngradable && (
          <div className="card-step" style={{ backgroundColor: '#FFFFFF', border: '1.5px solid #E8D3C4' }}>
            <div className="card-step-header" style={{ marginBottom: '14px' }}>
              <div className="step-title-wrapper">
                <CheckCircle2 color="#16A34A" size={24} />
                <div>
                  <h2 className="step-main-title">{t('triage_title', 'Clinical Triage & Diagnosis')}</h2>
                  <p className="step-main-subtitle">AI Screening Report ready for clinical review</p>
                </div>
              </div>
            </div>

            <div style={{
              border: '1px solid var(--border-strong)',
              borderRadius: '0',
              padding: '16px',
              backgroundColor: screeningResult.referable ? 'var(--status-warn-bg)' : 'var(--status-pass-bg)',
              marginBottom: '16px'
            }}>
              <div style={{ fontSize: '11px', fontWeight: 800, textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.5px' }}>
                {t('icdr_grading', 'ICDR Severity Grading')}
              </div>
              <div style={{ fontSize: '20px', fontWeight: 800, color: 'var(--text-primary)', marginTop: '3px' }}>
                {localizedGradeTitle}
              </div>
              <div style={{ fontSize: '13px', fontWeight: 700, color: screeningResult.referable ? '#B45309' : '#15803D', marginTop: '4px' }}>
                {localizedAction}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="action-btn-row">
              <button
                id="btn-preview-pdf"
                className="btn btn-outline"
                onClick={onOpenPdfModal}
              >
                <FileText size={16} /> {t('view_report_btn', 'Preview 1-Page PDF Report')}
              </button>

              <button
                id="btn-download-pdf-hw"
                className="btn btn-primary"
                style={{ backgroundColor: '#BD4319', borderColor: '#BD4319' }}
                onClick={onOpenPdfModal}
              >
                <Download size={16} /> {t('download_pdf_btn', 'Download Certified Clinical PDF')}
              </button>
            </div>

            <button
              className="btn btn-outline"
              style={{ width: '100%', marginTop: '10px' }}
              onClick={() => alert(`Forwarding report to Tele-Ophthalmology Hub for ${patientInfo.name || 'Patient'}...`)}
            >
              <Send size={14} /> {t('telemed_hub', 'Telemedicine Hub: District Hospital')}
            </button>
          </div>
        )}
      </div>

      {/* ── Right Auxiliary Column: Banners, Tips & Recent Patients ──────────── */}
      <aside className="portal-auxiliary-column" aria-label="Quick Tips and Recent Patients">
        {/* Banner Card: Healthy Eyes Brighter Lives */}
        <div className="widget-healthy-eyes-banner">
          <h3 className="widget-healthy-title">
            {t('healthy_eyes_title', 'Healthy Eyes Brighter Lives')}
          </h3>
          <p className="widget-healthy-sub">
            {t('healthy_eyes_sub', 'Early detection of retinal diseases can prevent vision loss.')}
          </p>
        </div>

        {/* Quick Tips Card */}
        <div className="widget-quick-tips-card">
          <div className="widget-tips-header">
            <Lightbulb size={16} style={{ color: '#D97706' }} />
            <span>{t('quick_tips_title', 'Quick Tips')}</span>
          </div>

          <div className="widget-tip-item">
            <span className="widget-tip-dot">•</span>
            <span>{t('tip_1', 'Ensure the image is well-focused')}</span>
          </div>
          <div className="widget-tip-item">
            <span className="widget-tip-dot">•</span>
            <span>{t('tip_2', 'Capture both eyes (OD & OS) when possible')}</span>
          </div>
          <div className="widget-tip-item">
            <span className="widget-tip-dot">•</span>
            <span>{t('tip_3', 'Check patient details before submitting')}</span>
          </div>
          <div className="widget-tip-item">
            <span className="widget-tip-dot">•</span>
            <span>{t('tip_4', 'Sync data when back online')}</span>
          </div>
        </div>

        {/* Recent Patients Card (Connected to Supabase Backend Storage) */}
        <div className="widget-recent-card">
          <div className="widget-recent-header">
            <span className="widget-recent-title">
              {t('recent_patients_title', 'Recent Patients')}
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <button
                type="button"
                className="widget-recent-refresh-btn"
                onClick={fetchRecentScreenings}
                title="Refresh from Supabase Backend"
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  color: '#64748B',
                  display: 'inline-flex',
                  alignItems: 'center',
                  padding: '3px',
                  borderRadius: '0'
                }}
              >
                <RefreshCw size={12} className={isLoadingPatients ? 'spin-icon' : ''} />
              </button>
              <button
                type="button"
                className="widget-recent-viewall"
                onClick={() => {
                  if (onNavigateToRecords) {
                    onNavigateToRecords();
                  } else {
                    alert(`Showing ${recentPatients.length} patient records synced with Supabase.`);
                  }
                }}
                title="View all patient records in database"
              >
                <span>{t('view_all', 'View All')}</span>
                <span>→</span>
              </button>
            </div>
          </div>

          {isLoadingPatients && recentPatients.length === 0 ? (
            <div style={{ padding: '16px 0', textAlign: 'center', fontSize: '12px', color: '#94A3B8' }}>
              Syncing patient records with Supabase...
            </div>
          ) : recentPatients.length === 0 ? (
            <div style={{ padding: '16px 0', textAlign: 'center', fontSize: '12px', color: '#94A3B8' }}>
              No screening records found in Supabase database.
            </div>
          ) : (
            recentPatients.map((p, idx) => {
              const displayName = p.patient_name || 'Anonymous Patient';
              const isReferral = Boolean(p.referable_dr) || (p.icdr_grade !== undefined && p.icdr_grade >= 2);
              const statusText = isReferral ? 'Referral' : (p.icdr_grade !== undefined && p.icdr_grade !== null ? `Grade ${p.icdr_grade}` : 'Screened');

              return (
                <div
                  key={p.id || idx}
                  className="widget-patient-row"
                  onClick={() => handleSelectRecentPatient(p)}
                  title={`Click to load ${displayName}'s details into form`}
                >
                  <div className="widget-patient-left">
                    <div className="widget-patient-avatar">{getInitials(displayName)}</div>
                    <div>
                      <div className="widget-patient-name">{displayName}</div>
                      <div className="widget-patient-demog">
                        {p.patient_age && p.patient_age !== 'N/A' ? `${p.patient_age} yrs` : 'Adult'} • {p.patient_gender && p.patient_gender !== 'N/A' ? p.patient_gender : 'Patient'}
                      </div>
                    </div>
                  </div>

                  <div className="widget-patient-right">
                    <span className="widget-patient-date">{formatDate(p.created_at)}</span>
                    <span className={isReferral ? 'status-badge-screened' : 'status-badge-pending'}>
                      {statusText}
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Bottom Tagline & Heartbeat/EKG Status */}
        <div className="portal-ekg-footer">
          <Activity size={14} className="portal-ekg-icon" />
          <span>{t('tagline_empower', 'Detect • Diagnose • Empower')}</span>
        </div>
      </aside>
    </div>
  );
}
