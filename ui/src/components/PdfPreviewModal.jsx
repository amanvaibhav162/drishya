import React from 'react';
import { X, Download, Printer, CheckCircle2, ShieldCheck } from 'lucide-react';

export default function PdfPreviewModal({ isOpen, onClose, presetData, patientInfo }) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldCheck size={18} color="#2563EB" />
            <span style={{ fontWeight: 800, fontSize: '14px' }}>
              Official Clinical Report Preview • ICDR Tele-Screening
            </span>
          </div>
          <button
            className="btn btn-outline"
            style={{ padding: '4px 8px' }}
            onClick={onClose}
          >
            <X size={16} />
          </button>
        </div>

        {/* Modal Body: Standard 1-Page PDF Layout */}
        <div className="modal-body">
          <div style={{
            border: '1px solid #CBD5E1',
            borderRadius: '6px',
            padding: '20px',
            backgroundColor: '#FFFFFF',
            fontSize: '11px',
            color: '#0F172A'
          }}>
            {/* Top Bar */}
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '2px solid #0F172A', paddingBottom: '8px', marginBottom: '12px' }}>
              <div>
                <div style={{ fontSize: '15px', fontWeight: 900 }}>DRISHYA TELE-OPHTHALMOLOGY REPORT</div>
                <div style={{ fontSize: '10px', color: '#64748B' }}>National Eye Care Program • ICDR Clinical Standard</div>
              </div>
              <div style={{ textAlign: 'right', fontSize: '10px' }}>
                <div><b>Report ID:</b> DSH-2026-88492-OS</div>
                <div><b>Date:</b> 30-Aug-2026 | <b>Facility:</b> PHC Rampur (Zone 4)</div>
              </div>
            </div>

            {/* Patient Demographics */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '8px', backgroundColor: '#F8FAFC', padding: '8px', border: '1px solid #E2E8F0', borderRadius: '4px', marginBottom: '12px' }}>
              <div><b>Patient:</b> {patientInfo.name}</div>
              <div><b>Age/Sex:</b> 54 Yrs / Male</div>
              <div><b>ABHA ID:</b> {patientInfo.abhaId}</div>
              <div><b>Eye:</b> <b>Left Eye (OS)</b></div>
              <div><b>History:</b> Type 2 DM (11 Yrs)</div>
              <div><b>HbA1c:</b> 8.6% (Uncontrolled)</div>
              <div><b>HTN:</b> Yes (140/90)</div>
              <div><b>Visual Acuity:</b> 6/9</div>
            </div>

            {/* Diagnostic Triage Banner */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1.2fr 1fr 0.8fr',
              gap: '10px',
              border: '1px solid #CBD5E1',
              borderRadius: '4px',
              padding: '10px',
              marginBottom: '12px',
              backgroundColor: '#FFFFFF'
            }}>
              <div>
                <div style={{ fontSize: '9px', color: '#64748B', fontWeight: 700 }}>PRIMARY FINDING</div>
                <div style={{ fontSize: '13px', fontWeight: 900 }}>{presetData.gradeTitle}</div>
                <div style={{ fontSize: '9px', color: '#475569' }}>{presetData.gradeDesc}</div>
              </div>
              <div>
                <div style={{ fontSize: '9px', color: '#64748B', fontWeight: 700 }}>TRIAGE RECOMMENDATION</div>
                <div style={{ fontSize: '12px', fontWeight: 800, color: presetData.referable ? '#B91C1C' : '#15803D' }}>
                  {presetData.referable ? 'REFERABLE DR (Refer to Specialist)' : 'NON-REFERABLE (Routine Annual Screening)'}
                </div>
                <div style={{ fontSize: '9px', color: '#64748B' }}>Slit-lamp & OCT exam required</div>
              </div>
              <div>
                <div style={{ fontSize: '9px', color: '#64748B', fontWeight: 700 }}>QUALITY & CONFIDENCE</div>
                <div style={{ fontSize: '11px', fontWeight: 700 }}>IQA: <span style={{ color: '#16A34A' }}>PASS (Q={presetData.iqaScore})</span></div>
                <div style={{ fontSize: '10px', color: '#475569' }}>Confidence: {presetData.confidence}</div>
              </div>
            </div>

            {/* 3 Image Panels */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', marginBottom: '12px' }}>
              <div style={{ textAlign: 'center' }}>
                <img src={presetData.preprocessedImg} alt="Panel A" style={{ width: '100%', height: '140px', objectFit: 'contain', backgroundColor: '#000', borderRadius: '4px' }} />
                <div style={{ fontSize: '9px', fontWeight: 700, marginTop: '4px' }}>(a) Preprocessed Retina</div>
                <div style={{ fontSize: '8px', color: '#64748B' }}>Normalized 384x384</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <img src={presetData.lesionsImg} alt="Panel B" style={{ width: '100%', height: '140px', objectFit: 'contain', backgroundColor: '#000', borderRadius: '4px' }} />
                <div style={{ fontSize: '9px', fontWeight: 700, marginTop: '4px' }}>(b) Lesions Overlay</div>
                <div style={{ fontSize: '8px', color: '#64748B' }}>Red: MAs | Yellow: Exudates</div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <img src={presetData.gradcamImg} alt="Panel C" style={{ width: '100%', height: '140px', objectFit: 'contain', backgroundColor: '#000', borderRadius: '4px' }} />
                <div style={{ fontSize: '9px', fontWeight: 700, marginTop: '4px' }}>(c) Grad-CAM++ Attention</div>
                <div style={{ fontSize: '8px', color: '#64748B' }}>Neural Saliency Focus</div>
              </div>
            </div>

            {/* Biomarker Table */}
            <table className="data-table" style={{ marginBottom: '12px', fontSize: '10px' }}>
              <thead>
                <tr>
                  <th style={{ padding: '4px 8px' }}>Biomarker / Feature</th>
                  <th style={{ padding: '4px 8px' }}>Detected Value</th>
                  <th style={{ padding: '4px 8px' }}>Reference Threshold</th>
                  <th style={{ padding: '4px 8px' }}>Clinical Relevance</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ padding: '4px 8px' }}>Microaneurysms (MAs)</td>
                  <td style={{ padding: '4px 8px' }}><b>{presetData.biomarkers.mas}</b></td>
                  <td style={{ padding: '4px 8px' }}>0 (Normal) | 1-5 (Mild) | &gt;5 (Moderate)</td>
                  <td style={{ padding: '4px 8px' }}>Active microvascular leakage</td>
                </tr>
                <tr>
                  <td style={{ padding: '4px 8px' }}>Hard Exudates (Lipids)</td>
                  <td style={{ padding: '4px 8px' }}><b>{presetData.biomarkers.exudates}</b></td>
                  <td style={{ padding: '4px 8px' }}>Absent in Normal / Mild</td>
                  <td style={{ padding: '4px 8px' }}>Lipoprotein deposits near arcade</td>
                </tr>
                <tr>
                  <td style={{ padding: '4px 8px' }}>Hemorrhage Spread</td>
                  <td style={{ padding: '4px 8px' }}><b>{presetData.biomarkers.hemorrhages}</b></td>
                  <td style={{ padding: '4px 8px' }}>4 quadrants = Severe (4:2:1 Rule)</td>
                  <td style={{ padding: '4px 8px' }}>Below severe NPDR threshold</td>
                </tr>
              </tbody>
            </table>

            {/* Action Plan & Doctor Sign-Off */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', border: '1px solid #E2E8F0', padding: '8px', borderRadius: '4px', backgroundColor: '#FAFAFA' }}>
              <div>
                <div style={{ fontWeight: 800, fontSize: '10px' }}>CLINICAL ACTION PLAN:</div>
                <div style={{ fontSize: '9px', marginTop: '2px' }}>• Schedule dilated fundus exam + OCT within <b>3-4 weeks</b> at District Hospital.</div>
                <div style={{ fontSize: '9px' }}>• Consult primary physician for HbA1c control. Repeat tele-screening in <b>6 months</b>.</div>
              </div>
              <div>
                <div style={{ fontWeight: 800, fontSize: '10px' }}>OPHTHALMOLOGIST SIGN-OFF:</div>
                <div style={{ fontSize: '9px', marginTop: '2px' }}><b>Reviewer:</b> Dr. Rajesh Varma, MD (Ophthal) • Reg: MCI-49218</div>
                <div style={{ fontSize: '9px' }}><b>Status:</b> [ <b>X</b> ] Concur with AI Grade   [ ] Re-evaluate</div>
                <div style={{ fontSize: '8px', color: '#64748B', fontStyle: 'italic' }}>Verified Digital Tele-Sign • 30-Aug-2026</div>
              </div>
            </div>
          </div>
        </div>

        {/* Modal Actions */}
        <div style={{ padding: '16px 20px', borderTop: '1px solid var(--border-light)', display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
          <button className="btn btn-outline" onClick={() => window.print()}>
            <Printer size={16} /> Print Report
          </button>
          <button
            className="btn btn-primary"
            onClick={() => {
              // Trigger direct file download from backend / static assets
              const link = document.createElement('a');
              link.href = '/assets/DRISHYA_Clinical_Report.pdf';
              link.download = `DRISHYA_Report_${patientInfo.name.replace(/\s+/g, '_')}.pdf`;
              link.click();
              onClose();
            }}
          >
            <Download size={16} /> Download 1-Page PDF
          </button>
        </div>
      </div>
    </div>
  );
}
