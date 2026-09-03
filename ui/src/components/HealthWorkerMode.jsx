
import React, { useRef, useState } from 'react';
import { AlertCircle, CheckCircle2, FileText, Download, Send, RefreshCw, User, Phone, Calendar, Upload, Image as ImageIcon } from 'lucide-react';

export default function HealthWorkerMode({
  patientInfo,
  setPatientInfo,
  uploadedImage,
  onImageSelected,
  onClearImage,
  screeningResult,
  onRunScreening,
  isProcessing,
  onOpenPdfModal
}) {
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);

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

  return (
    <div style={{ maxWidth: '640px', margin: '0 auto' }}>
      {/* Patient Header Card */}
      <div className="card" style={{ marginBottom: '16px' }}>
        <h2 className="text-h2" style={{ marginBottom: '12px' }}>
          1. Patient Registration & Tagging
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '12px', marginBottom: '12px' }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">
              <User size={12} style={{ display: 'inline', marginRight: '4px' }} />
              Patient Full Name
            </label>
            <input
              id="input-patient-name"
              type="text"
              className="form-input"
              value={patientInfo.name}
              onChange={(e) => setPatientInfo({ ...patientInfo, name: e.target.value })}
              placeholder="e.g. Ramesh Kumar"
            />
          </div>

          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">
              <Phone size={12} style={{ display: 'inline', marginRight: '4px' }} />
              Mobile Number
            </label>
            <input
              id="input-patient-phone"
              type="text"
              className="form-input"
              value={patientInfo.phone}
              onChange={(e) => setPatientInfo({ ...patientInfo, phone: e.target.value })}
              placeholder="+91 98765 43210"
            />
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.1fr 1.3fr', gap: '12px' }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">
              <Calendar size={12} style={{ display: 'inline', marginRight: '4px' }} />
              Age (Years)
            </label>
            <input
              id="input-patient-age"
              type="number"
              min="1"
              max="120"
              className="form-input"
              value={patientInfo.age}
              onChange={(e) => setPatientInfo({ ...patientInfo, age: e.target.value })}
              placeholder="e.g. 54"
            />
          </div>

          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">
              <User size={12} style={{ display: 'inline', marginRight: '4px' }} />
              Gender
            </label>
            <select
              id="input-patient-gender"
              className="form-input"
              value={patientInfo.gender || 'Male'}
              onChange={(e) => setPatientInfo({ ...patientInfo, gender: e.target.value })}
              style={{ cursor: 'pointer' }}
            >
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
          </div>

          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label">
              <FileText size={12} style={{ display: 'inline', marginRight: '4px' }} />
              ABHA ID
            </label>
            <input
              id="input-patient-abha"
              type="text"
              className="form-input"
              value={patientInfo.abhaId}
              onChange={(e) => setPatientInfo({ ...patientInfo, abhaId: e.target.value })}
              placeholder="91-4820-1940-52"
            />
          </div>
        </div>
      </div>

      {/* Retinal Capture Card */}
      <div className="card" style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <h2 className="text-h2">2. Retinal Image Ingestion</h2>
          <span className="badge badge-neutral">Eye: Left Eye (OS)</span>
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
                Click to upload or drag & drop fundus scan
              </div>
              <div className="text-micro" style={{ color: 'var(--text-muted)' }}>
                Supports PNG, JPG, or DICOM exports (Standard 45° FOV)
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
                <RefreshCw size={11} /> Change Scan
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
                      🛑 RETAKE SCAN REQUIRED (Quality Failed)
                    </div>
                    <div className="text-caption" style={{ color: '#B91C1C', marginTop: '2px' }}>
                      The captured scan is blurry or underexposed (Q = {screeningResult.qScore || screeningResult.iqaScore}). Please steady camera and recapture before the patient leaves.
                    </div>
                    <button
                      className="btn btn-danger text-caption"
                      style={{ marginTop: '10px', padding: '6px 12px' }}
                      onClick={onClearImage}
                    >
                      <RefreshCw size={14} /> Recapture Eye Photo
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
                      IMAGE QUALITY APPROVED (Q = {screeningResult.iqaScore})
                    </div>
                    <div className="text-micro" style={{ color: '#15803D' }}>
                      Focus and illumination verified. Ready for diagnostic grading.
                    </div>
                  </div>
                </div>
              )
            )}

            {/* Run Screening Action Button */}
            {!screeningResult && (
              <button
                id="btn-run-screening"
                className="btn btn-primary btn-lg"
                style={{ width: '100%' }}
                onClick={onRunScreening}
                disabled={isProcessing}
              >
                {isProcessing ? (
                  <>
                    <RefreshCw size={18} className="spin-icon" /> Running AI Analysis...
                  </>
                ) : (
                  'Process Scan & Grade Retinopathy'
                )}
              </button>
            )}
          </div>
        )}
      </div>

      {/* Result Card (When Screened & Passed IQA) */}
      {screeningResult && !isUngradable && (
        <div className="card" style={{ marginBottom: '16px' }}>
          <h2 className="text-h2" style={{ marginBottom: '12px' }}>
            3. Diagnostic Finding & Clinical Action
          </h2>

          <div style={{
            border: '1px solid var(--border-strong)',
            borderRadius: 'var(--radius-md)',
            padding: '14px',
            backgroundColor: screeningResult.referable ? 'var(--status-warn-bg)' : 'var(--status-pass-bg)',
            marginBottom: '16px'
          }}>
            <div className="text-micro" style={{ fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
              ICDR Severity Grading
            </div>
            <div className="text-h1" style={{ color: 'var(--text-primary)', marginTop: '2px' }}>
              {screeningResult.gradeTitle}
            </div>
            <div className="text-caption" style={{ fontWeight: 700, color: screeningResult.referable ? '#B45309' : '#15803D', marginTop: '4px' }}>
              {screeningResult.actionRecommendation}
            </div>
          </div>

          {/* Action Buttons */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            <button
              id="btn-preview-pdf"
              className="btn btn-outline"
              onClick={onOpenPdfModal}
            >
              <FileText size={16} /> Preview Report
            </button>

            <button
              id="btn-download-pdf-hw"
              className="btn btn-primary"
              onClick={onOpenPdfModal}
            >
              <Download size={16} /> Download 1-Page PDF
            </button>
          </div>

          <button
            className="btn btn-outline"
            style={{ width: '100%', marginTop: '10px' }}
            onClick={() => alert(`Forwarding report to Tele-Ophthalmology Hub for ${patientInfo.name}...`)}
          >
            <Send size={14} /> Send to District Specialist
          </button>
        </div>
      )}
    </div>
  );
}
