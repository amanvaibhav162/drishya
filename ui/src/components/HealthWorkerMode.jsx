import React, { useState } from 'react';
import { Camera, AlertCircle, CheckCircle2, FileText, Download, Send, RefreshCw, User, Phone } from 'lucide-react';

export default function HealthWorkerMode({ 
  presetData, 
  patientInfo, 
  setPatientInfo, 
  onRunScreening, 
  isProcessing,
  onOpenPdfModal
}) {
  const [retakeTriggered, setRetakeTriggered] = useState(false);

  const isUngradable = presetData.grade === 'ungradable';

  return (
    <div style={{ maxWidth: '640px', margin: '0 auto' }}>
      {/* Patient Header Card */}
      <div className="card" style={{ marginBottom: '16px' }}>
        <h2 style={{ fontSize: '15px', fontWeight: 800, marginBottom: '12px' }}>
          1. Patient Registration & Tagging
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
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
      </div>

      {/* Retinal Capture Card */}
      <div className="card" style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
          <h2 style={{ fontSize: '15px', fontWeight: 800 }}>2. Retinal Image Ingestion</h2>
          <span className="badge badge-neutral">Eye: Left Eye (OS)</span>
        </div>

        <div className="image-canvas-wrapper" style={{ height: '280px', marginBottom: '14px' }}>
          <img
            src={presetData.rawImg}
            alt="Fundus Capture"
            className="retina-img"
          />
        </div>

        {/* Quality Triage Banner */}
        {isUngradable ? (
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
              <div style={{ color: '#991B1B', fontWeight: 800, fontSize: '13px' }}>
                🛑 RETAKE SCAN REQUIRED (Quality Failed)
              </div>
              <div style={{ color: '#B91C1C', fontSize: '12px', marginTop: '2px' }}>
                The captured scan is blurry and underexposed. Please steady the camera and click again now before the patient leaves.
              </div>
              <button
                className="btn btn-danger"
                style={{ marginTop: '10px', padding: '6px 12px', fontSize: '12px' }}
                onClick={() => alert("Prompting Camera to Recapture Image...")}
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
              <div style={{ color: '#166534', fontWeight: 700, fontSize: '12px' }}>
                IMAGE QUALITY APPROVED (Q = {presetData.iqaScore})
              </div>
              <div style={{ color: '#15803D', fontSize: '11px' }}>
                Focus and illumination verified. Ready for diagnostic grading.
              </div>
            </div>
          </div>
        )}

        {!isUngradable && (
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
              '⚡ Process Scan & Grade Retinopathy'
            )}
          </button>
        )}
      </div>

      {/* Result Card (When Not Ungradable) */}
      {!isUngradable && (
        <div className="card" style={{ marginBottom: '16px' }}>
          <h2 style={{ fontSize: '15px', fontWeight: 800, marginBottom: '12px' }}>
            3. Diagnostic Finding & Clinical Action
          </h2>

          <div style={{
            border: '1px solid var(--border-strong)',
            borderRadius: 'var(--radius-md)',
            padding: '14px',
            backgroundColor: presetData.referable ? 'var(--status-warn-bg)' : 'var(--status-pass-bg)',
            marginBottom: '16px'
          }}>
            <div style={{ fontSize: '11px', fontWeight: 700, textTransform: 'uppercase', color: 'var(--text-muted)' }}>
              ICDR Severity Grading
            </div>
            <div style={{ fontSize: '18px', fontWeight: 800, color: 'var(--text-primary)', marginTop: '2px' }}>
              {presetData.gradeTitle}
            </div>
            <div style={{ fontSize: '12px', fontWeight: 700, color: presetData.referable ? '#B45309' : '#15803D', marginTop: '4px' }}>
              {presetData.actionRecommendation}
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
