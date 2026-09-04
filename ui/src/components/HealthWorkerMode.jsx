import React, { useRef, useState } from 'react';
import { AlertCircle, CheckCircle2, FileText, Download, Send, RefreshCw, User, Phone, Calendar, Upload } from 'lucide-react';
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
  onOpenPdfModal
}) {
  const { t } = useLanguage();
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const pipelineStages = [
    {
      step: 1,
      title: t('stage_1_title'),
      subtext: t('stage_1_sub'),
      pct: 18,
    },
    {
      step: 2,
      title: t('stage_2_title'),
      subtext: t('stage_2_sub'),
      pct: 38,
    },
    {
      step: 3,
      title: t('stage_3_title'),
      subtext: t('stage_3_sub'),
      pct: 65,
    },
    {
      step: 4,
      title: t('stage_4_title'),
      subtext: t('stage_4_sub'),
      pct: 85,
    },
    {
      step: 5,
      title: t('stage_5_title'),
      subtext: t('stage_5_sub'),
      pct: 95,
    },
    {
      step: 6,
      title: t('stage_6_title'),
      subtext: t('stage_6_sub'),
      pct: 100,
    },
  ];

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
    <div style={{ maxWidth: '640px', margin: '0 auto' }}>
      {/* Patient Header Card */}
      <div className="card" style={{ marginBottom: '16px' }}>
        <h2 className="text-h2" style={{ marginBottom: '12px' }}>
          {t('step1_title')}
        </h2>
        <div className="patient-form-row-1">
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">
              <User size={12} style={{ display: 'inline', marginRight: '4px' }} />
              {t('patient_name')}
            </label>
            <input
              id="input-patient-name"
              type="text"
              className="form-input"
              value={patientInfo.name}
              onChange={(e) => setPatientInfo({ ...patientInfo, name: e.target.value })}
              placeholder={t('patient_name_placeholder')}
            />
          </div>

          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">
              <Phone size={12} style={{ display: 'inline', marginRight: '4px' }} />
              {t('mobile_num')}
            </label>
            <input
              id="input-patient-phone"
              type="text"
              className="form-input"
              value={patientInfo.phone}
              onChange={(e) => setPatientInfo({ ...patientInfo, phone: e.target.value })}
              placeholder={t('mobile_placeholder')}
            />
          </div>
        </div>

        <div className="patient-form-row-2">
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">
              <Calendar size={12} style={{ display: 'inline', marginRight: '4px' }} />
              {t('age')}
            </label>
            <input
              id="input-patient-age"
              type="number"
              min="1"
              max="120"
              className="form-input"
              value={patientInfo.age}
              onChange={(e) => setPatientInfo({ ...patientInfo, age: e.target.value })}
              placeholder={t('age_placeholder')}
            />
          </div>

          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">
              <User size={12} style={{ display: 'inline', marginRight: '4px' }} />
              {t('gender')}
            </label>
            <select
              id="input-patient-gender"
              className="form-input"
              value={patientInfo.gender || 'Male'}
              onChange={(e) => setPatientInfo({ ...patientInfo, gender: e.target.value })}
              style={{ cursor: 'pointer' }}
            >
              <option value="Male">{t('gender_male')}</option>
              <option value="Female">{t('gender_female')}</option>
              <option value="Other">{t('gender_other')}</option>
            </select>
          </div>

          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">
              <FileText size={12} style={{ display: 'inline', marginRight: '4px' }} />
              {t('abha_id')}
            </label>
            <input
              id="input-patient-abha"
              type="text"
              className="form-input"
              value={patientInfo.abhaId}
              onChange={(e) => setPatientInfo({ ...patientInfo, abhaId: e.target.value })}
              placeholder={t('abha_placeholder')}
            />
          </div>
        </div>
      </div>

      {/* Retinal Capture Card */}
      <div className="card" style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <h2 className="text-h2">{t('step2_title')}</h2>
          <span className="badge badge-neutral">{t('eye_examined')}: OS</span>
        </div>

        {/* Upload Dropzone when no image is loaded */}
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
              className={`dropzone ${isDragOver ? 'drag-active' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
            >
              <div style={{
                width: '48px',
                height: '48px',
                borderRadius: '50%',
                backgroundColor: 'var(--brand-light)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '4px'
              }}>
                <Upload size={24} color="var(--brand-primary)" />
              </div>
              <div className="text-body" style={{ fontWeight: 700 }}>
                {t('dropzone_title')}
              </div>
              <div className="text-micro" style={{ color: 'var(--text-muted)' }}>
                {t('dropzone_subtitle')}
              </div>
            </div>
          </div>
        ) : (
          /* Active Retinal Scan View */
          <div>
            <div className="image-canvas-wrapper" style={{ height: '280px', marginBottom: '12px' }}>
              <img
                src={uploadedImage.previewUrl}
                alt="Fundus Capture"
                className="retina-img"
              />
            </div>

            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '14px',
              padding: '6px 10px',
              backgroundColor: 'var(--bg-subtle)',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border-light)'
            }}>
              <span className="text-micro" style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>
                📁 {uploadedImage.name || 'Retinal Scan Loaded'}
              </span>
              <button
                type="button"
                className="btn btn-outline text-micro"
                style={{ padding: '3px 8px' }}
                onClick={onClearImage}
                disabled={isProcessing}
              >
                <RefreshCw size={11} /> {t('change_photo')}
              </button>
            </div>

            {/* Quality Triage Banner (Shown only after screening has run) */}
            {screeningResult && (
              isUngradable ? (
                <div style={{
                  backgroundColor: 'var(--status-danger-bg)',
                  border: '1px solid var(--status-danger-border)',
                  borderRadius: 'var(--radius-md)',
                  padding: '12px 16px',
                  marginBottom: '14px',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '12px'
                }}>
                  <AlertCircle color="#DC2626" size={24} style={{ flexShrink: 0, marginTop: '2px' }} />
                  <div>
                    <div className="text-body" style={{ color: '#991B1B', fontWeight: 800 }}>
                      {t('iqa_failed_title')}
                    </div>
                    <div className="text-caption" style={{ color: '#B91C1C', marginTop: '2px' }}>
                      {t('iqa_retake_action')} (Q = {screeningResult.qScore || screeningResult.iqaScore})
                    </div>
                    <button
                      className="btn btn-danger text-caption"
                      style={{ marginTop: '10px', padding: '6px 12px' }}
                      onClick={onClearImage}
                    >
                      <RefreshCw size={14} /> {t('retake_btn')}
                    </button>
                  </div>
                </div>
              ) : (
                <div style={{
                  backgroundColor: 'var(--status-pass-bg)',
                  border: '1px solid var(--status-pass-border)',
                  borderRadius: 'var(--radius-md)',
                  padding: '10px 14px',
                  marginBottom: '14px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px'
                }}>
                  <CheckCircle2 color="#16A34A" size={20} />
                  <div>
                    <div className="text-caption" style={{ color: '#166534', fontWeight: 700 }}>
                      {t('iqa_passed_title')} (Q = {screeningResult.iqaScore})
                    </div>
                    <div className="text-micro" style={{ color: '#15803D' }}>
                      {t('iqa_passed_sub')}
                    </div>
                  </div>
                </div>
              )
            )}

            {/* Run Screening Action Button / Dynamic Pipeline Progress Card */}
            {!screeningResult && (
              isProcessing ? (
                <div className="processing-card" id="screening-progress-card">
                  {/* Header: Rotating Spinner, Live Title & Progress Percent */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <RefreshCw size={15} className="spin-icon" style={{ color: 'var(--brand-primary)' }} />
                      <span style={{ fontSize: '13px', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.2px' }}>
                        {t('analyzing_btn')}
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
                      style={{ width: `${activeStage.pct}%` }}
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

                  {/* Real-time Explanatory Subtext */}
                  <div className="text-micro" style={{ color: 'var(--text-secondary)', marginBottom: '12px', minHeight: '18px' }}>
                    {activeStage.subtext}
                  </div>

                  {/* 5-Stage Micro Segment Indicators */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '4px' }}>
                    {[1, 2, 3, 4, 5].map((sNum) => {
                      const isDone = (currentStep || 1) > sNum;
                      const isCurrent = (currentStep || 1) === sNum;
                      return (
                        <div
                          key={sNum}
                          style={{
                            height: '4px',
                            borderRadius: '2px',
                            backgroundColor: isDone ? '#0F172A' : (isCurrent ? '#475569' : '#E2E8F0'),
                            transition: 'background-color 0.3s ease'
                          }}
                        />
                      );
                    })}
                  </div>
                </div>
              ) : (
                <button
                  id="btn-run-screening"
                  className="btn btn-primary btn-lg"
                  style={{ width: '100%' }}
                  onClick={onRunScreening}
                >
                  {t('run_screening_btn')}
                </button>
              )
            )}
          </div>
        )}
      </div>

      {/* Result Card (When Screened & Passed IQA) */}
      {screeningResult && !isUngradable && (
        <div className="card" style={{ marginBottom: '16px' }}>
          <h2 className="text-h2" style={{ marginBottom: '12px' }}>
            {t('triage_title')}
          </h2>

          <div style={{
            border: '1px solid var(--border-strong)',
            borderRadius: 'var(--radius-md)',
            padding: '14px',
            backgroundColor: screeningResult.referable ? 'var(--status-warn-bg)' : 'var(--status-pass-bg)',
            marginBottom: '16px'
          }}>
            <div className="text-micro" style={{ fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
              {t('icdr_grading')}
            </div>
            <div className="text-h1" style={{ color: 'var(--text-primary)', marginTop: '2px' }}>
              {localizedGradeTitle}
            </div>
            <div className="text-caption" style={{ fontWeight: 700, color: screeningResult.referable ? '#B45309' : '#15803D', marginTop: '4px' }}>
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
              <FileText size={16} /> {t('view_report_btn')}
            </button>

            <button
              id="btn-download-pdf-hw"
              className="btn btn-primary"
              onClick={onOpenPdfModal}
            >
              <Download size={16} /> {t('download_pdf_btn')}
            </button>
          </div>

          <button
            className="btn btn-outline"
            style={{ width: '100%', marginTop: '10px' }}
            onClick={() => alert(`Forwarding report to Tele-Ophthalmology Hub for ${patientInfo.name}...`)}
          >
            <Send size={14} /> {t('telemed_hub')}
          </button>
        </div>
      )}
    </div>
  );
}
