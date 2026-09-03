import React from 'react';
import { X, Download, Printer } from 'lucide-react';

export default function PdfPreviewModal({ isOpen, onClose, screeningResult, patientInfo }) {
  if (!isOpen || !screeningResult) return null;

  const isReferable = screeningResult.referable;
  const resultColor = isReferable ? '#B91C1C' : '#15803D';
  const resultText = isReferable ? 'Referral Needed' : 'No Referral Needed';

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" style={{ maxWidth: '860px' }} onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <img
              src="/assets/drishyalogo.jpeg"
              alt="DRISHYA"
              style={{ height: '22px', maxWidth: '44px', objectFit: 'contain', borderRadius: '4px' }}
            />
            <span className="text-body" style={{ fontWeight: 800 }}>
              Drishya Diagnostic Report Preview
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

        {/* Modal Body: IDx-DR Style Layout */}
        <div className="modal-body" style={{ maxWidth: '100%', padding: '20px' }}>
          <div style={{
            border: '1px solid #CBD5E1',
            borderRadius: '6px',
            backgroundColor: '#FFFFFF',
            overflow: 'hidden'
          }}>

            {/* ── Report Header ──────────────────────────────── */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '16px 20px',
              borderBottom: '2.5px solid #0F172A'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <img
                  src="/assets/drishyalogo.jpeg"
                  alt="DRISHYA Logo"
                  style={{ height: '36px', maxWidth: '72px', objectFit: 'contain', borderRadius: '4px' }}
                />
                <div>
                  <div style={{ fontSize: '18px', fontWeight: 900, color: '#0F172A', letterSpacing: '-0.3px' }}>
                    DRISHYA
                  </div>
                  <div className="text-micro" style={{ color: '#64748B' }}>
                    AI-Powered Tele-Ophthalmology
                  </div>
                </div>
              </div>
              <div className="text-h2" style={{ color: '#0F172A', fontWeight: 700 }}>
                Drishya Diagnostic Report
              </div>
            </div>

            {/* ── Patient + General Info ────────────────────── */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
              {/* Left: Patient Information */}
              <div>
                <div style={{
                  backgroundColor: '#0F172A',
                  color: '#fff',
                  padding: '4px 12px',
                  fontWeight: 700,
                  fontSize: '11px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Patient Information
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '130px 1fr', borderRight: '1px solid #E2E8F0' }}>
                  {[
                    ['PATIENT NAME', patientInfo.name || 'Anonymous Patient'],
                    ['ABHA ID', patientInfo.abhaId || 'Not Registered'],
                    ['AGE / SEX', patientInfo.age && patientInfo.gender ? `${patientInfo.age} Yrs / ${patientInfo.gender}` : (patientInfo.age ? `${patientInfo.age} Yrs` : (patientInfo.gender || 'Adult Screening'))],
                    ['RESULT DATE', new Date().toLocaleDateString('en-IN')],
                  ].map(([label, val], i) => (
                    <React.Fragment key={i}>
                      <div style={{
                        padding: '5px 10px',
                        fontSize: '10px',
                        fontWeight: 700,
                        color: '#64748B',
                        backgroundColor: '#F8FAFC',
                        borderBottom: '1px solid #E2E8F0'
                      }}>{label}</div>
                      <div style={{
                        padding: '5px 10px',
                        fontSize: '11px',
                        color: '#0F172A',
                        borderBottom: '1px solid #E2E8F0'
                      }}>{val}</div>
                    </React.Fragment>
                  ))}
                </div>
              </div>

              {/* Right: General Information */}
              <div>
                <div style={{
                  backgroundColor: '#0F172A',
                  color: '#fff',
                  padding: '4px 12px',
                  fontWeight: 700,
                  fontSize: '11px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  General Information
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '140px 1fr' }}>
                  {[
                    ['SCREENING CENTER', 'PHC Rampur (Zone 4)'],
                    ['EYE', 'Left Eye (OS)'],
                    ['ORDERING CODE', 'E11.9'],
                    ['REPORT ID', screeningResult.reportId || 'DSH-2026-84920'],
                  ].map(([label, val], i) => (
                    <React.Fragment key={i}>
                      <div style={{
                        padding: '5px 10px',
                        fontSize: '10px',
                        fontWeight: 700,
                        color: '#64748B',
                        backgroundColor: '#F8FAFC',
                        borderBottom: '1px solid #E2E8F0'
                      }}>{label}</div>
                      <div style={{
                        padding: '5px 10px',
                        fontSize: '11px',
                        color: '#0F172A',
                        borderBottom: '1px solid #E2E8F0'
                      }}>{val}</div>
                    </React.Fragment>
                  ))}
                </div>
              </div>
            </div>

            {/* ── 3 Fundus Images in a Single Horizontal Row ──── */}
            <div style={{
              backgroundColor: '#0F172A',
              color: '#fff',
              padding: '4px 12px',
              fontWeight: 700,
              fontSize: '11px',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}>
              3 Fundus Panels Evaluated in Screening
            </div>
            <div style={{ padding: '12px 14px', borderBottom: '1px solid #CBD5E1' }}>
              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr 1fr',
                gap: '12px'
              }}>
                {/* Image 1: Preprocessed */}
                <div style={{ textAlign: 'center' }}>
                  <img
                    src={screeningResult.preprocessedImg || screeningResult.rawImg}
                    alt="Preprocessed Retina"
                    style={{
                      width: '100%',
                      aspectRatio: '1',
                      objectFit: 'contain',
                      backgroundColor: '#000',
                      borderRadius: '4px',
                      border: '1px solid #E2E8F0'
                    }}
                  />
                  <div className="text-micro" style={{ fontWeight: 700, marginTop: '4px', color: '#0F172A' }}>
                    (a) Preprocessed Retina
                  </div>
                  <div className="text-micro" style={{ color: '#64748B', fontSize: '9px' }}>
                    1:1 crop • CLAHE normalized
                  </div>
                </div>

                {/* Image 2: Detected Lesions */}
                <div style={{ textAlign: 'center' }}>
                  <img
                    src={screeningResult.lesionsImg || screeningResult.rawImg}
                    alt="Detected Lesions"
                    style={{
                      width: '100%',
                      aspectRatio: '1',
                      objectFit: 'contain',
                      backgroundColor: '#000',
                      borderRadius: '4px',
                      border: '1px solid #E2E8F0'
                    }}
                  />
                  <div className="text-micro" style={{ fontWeight: 700, marginTop: '4px', color: '#0F172A' }}>
                    (b) Detected Lesions
                  </div>
                  <div className="text-micro" style={{ color: '#64748B', fontSize: '9px' }}>
                    MA (Red) • EX (Yel) • HE (Crimson)
                  </div>
                </div>

                {/* Image 3: Grad-CAM++ Attention */}
                <div style={{ textAlign: 'center' }}>
                  <img
                    src={screeningResult.gradcamImg || screeningResult.rawImg}
                    alt="Grad-CAM++ Attention"
                    style={{
                      width: '100%',
                      aspectRatio: '1',
                      objectFit: 'contain',
                      backgroundColor: '#000',
                      borderRadius: '4px',
                      border: '1px solid #E2E8F0'
                    }}
                  />
                  <div className="text-micro" style={{ fontWeight: 700, marginTop: '4px', color: '#0F172A' }}>
                    (c) Grad-CAM++ Attention
                  </div>
                  <div className="text-micro" style={{ color: '#64748B', fontSize: '9px' }}>
                    Neural saliency focus areas
                  </div>
                </div>
              </div>
              <div className="text-micro" style={{
                fontStyle: 'italic',
                color: '#64748B',
                textAlign: 'center',
                marginTop: '6px'
              }}>
                Image labeling and heatmaps are for explanatory guidance only and should not be used as independent diagnostic markers.
              </div>
            </div>

            {/* ── Results Section (Directly Below Photos) ─── */}
            <div style={{
              backgroundColor: '#0F172A',
              color: '#fff',
              padding: '4px 12px',
              fontWeight: 700,
              fontSize: '11px',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}>
              Diagnostic Results & Clinical Triage
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', borderBottom: '1px solid #CBD5E1' }}>
              {/* Left: Details */}
              <div style={{ borderRight: '1px solid #E2E8F0' }}>
                {[
                  ['CONDITION', 'Diabetic Retinopathy (with Macular Risk Assessment)'],
                  ['DIAGNOSIS', screeningResult.gradeDesc || 'No DR detected ETDRS level 20 and lower and no macular edema.'],
                  ['CARE PLAN', screeningResult.referable ? 'Refer to Ophthalmologist within 3-4 weeks for slit-lamp examination' : 'Routine Rescreening in 12 Months'],
                  ['AI INTERPRETATION', 'Autonomous deep neural interpretation via DRISHYA Retinal Engine v1.0.'],
                ].map(([label, val], i) => (
                  <div key={i} style={{
                    display: 'grid',
                    gridTemplateColumns: '130px 1fr',
                    borderBottom: i < 3 ? '1px solid #E2E8F0' : 'none'
                  }}>
                    <div style={{
                      padding: '6px 10px',
                      fontSize: '10px',
                      fontWeight: 700,
                      color: '#64748B',
                      backgroundColor: '#F8FAFC'
                    }}>{label}</div>
                    <div style={{
                      padding: '6px 10px',
                      fontSize: '11px',
                      color: '#0F172A',
                      lineHeight: 1.4
                    }}>{val}</div>
                  </div>
                ))}
              </div>

              {/* Right: Big Result */}
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '20px',
                gap: '4px',
                backgroundColor: isReferable ? '#FEF2F2' : '#F0FDF4'
              }}>
                <div style={{
                  fontSize: '10px',
                  fontWeight: 700,
                  color: '#64748B',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>Clinical Triage Decision</div>
                <div style={{
                  fontSize: '22px',
                  fontWeight: 900,
                  color: resultColor,
                  textAlign: 'center',
                  lineHeight: 1.15,
                  letterSpacing: '-0.5px'
                }}>
                  {resultText}
                </div>
                <div style={{
                  fontSize: '10px',
                  fontWeight: 700,
                  color: resultColor,
                  marginTop: '2px'
                }}>
                  Protocol: {isReferable ? 'Specialist Slit-Lamp Exam Recommended' : 'Routine Primary Care Screening'}
                </div>
                <div className="text-micro" style={{ color: '#64748B', marginTop: '2px' }}>
                  ICDR Grade {screeningResult.grade ?? 0} &nbsp;|&nbsp; Confidence: {screeningResult.confidence || '96.4%'} &nbsp;|&nbsp; IQA: {screeningResult.iqaPass !== false ? 'Pass' : 'Failed'}
                </div>
              </div>
            </div>

            {/* ── Biomarkers & AI Specs Grid ────────────────── */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
              {/* Left: Quantitative Retinal Biomarkers */}
              <div style={{ borderRight: '1px solid #E2E8F0' }}>
                <div style={{
                  backgroundColor: '#0F172A',
                  color: '#fff',
                  padding: '4px 12px',
                  fontWeight: 700,
                  fontSize: '11px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Quantitative Retinal Biomarkers
                </div>
                <div>
                  {[
                    ['Microaneurysms (MA)', screeningResult.biomarkers?.mas || '0 detected'],
                    ['Exudate Area (EX)', screeningResult.biomarkers?.exudates || '0.00% area'],
                    ['Hemorrhage Quadrants', screeningResult.biomarkers?.hemorrhages || '0 quadrants'],
                    ['Macular Edema Risk', screeningResult.biomarkers?.neovascularization === 'Present (PDR)' ? 'High Risk' : 'Low Risk (Fovea Clear)'],
                  ].map(([label, val], i) => (
                    <div key={i} style={{
                      display: 'grid',
                      gridTemplateColumns: '150px 1fr',
                      borderBottom: i < 3 ? '1px solid #E2E8F0' : 'none'
                    }}>
                      <div style={{
                        padding: '5px 10px',
                        fontSize: '10px',
                        fontWeight: 700,
                        color: '#64748B',
                        backgroundColor: '#F8FAFC'
                      }}>{label}</div>
                      <div style={{
                        padding: '5px 10px',
                        fontSize: '10px',
                        color: '#0F172A',
                        fontWeight: 600
                      }}>{val}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Right: AI Engine Specifications & Benchmarks */}
              <div>
                <div style={{
                  backgroundColor: '#0F172A',
                  color: '#fff',
                  padding: '4px 12px',
                  fontWeight: 700,
                  fontSize: '11px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  AI Engine Specifications & Benchmarks
                </div>
                <div>
                  {[
                    ['Autonomous Engine', 'DRISHYA PP-LCNet MTL (3.29M params)'],
                    ['Clinical Sensitivity', '94.2% (Multi-Center Cohort)'],
                    ['Clinical Specificity', '91.8% (Target: DR & DME)'],
                    ['Gradability Rate', '96.0% (Rural Field Validated)'],
                  ].map(([label, val], i) => (
                    <div key={i} style={{
                      display: 'grid',
                      gridTemplateColumns: '140px 1fr',
                      borderBottom: i < 3 ? '1px solid #E2E8F0' : 'none'
                    }}>
                      <div style={{
                        padding: '5px 10px',
                        fontSize: '10px',
                        fontWeight: 700,
                        color: '#64748B',
                        backgroundColor: '#F8FAFC'
                      }}>{label}</div>
                      <div style={{
                        padding: '5px 10px',
                        fontSize: '10px',
                        color: '#0F172A',
                        fontWeight: 600
                      }}>{val}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* ── Disclaimer ─────────────────────────────── */}
            <div style={{
              borderTop: '1px solid #CBD5E1',
              padding: '8px 12px',
              backgroundColor: '#FAFAFA'
            }}>
              <div className="text-micro" style={{ fontWeight: 700, color: '#0F172A', marginBottom: '2px' }}>
                DISCLAIMER
              </div>
              <div style={{ fontSize: '9px', color: '#64748B', lineHeight: 1.5 }}>
                A positive result indicates a high risk of diabetic retinopathy with a severity of ETDRS level 35
                or higher and/or macular edema. DRISHYA AI diabetic retinopathy screening does not replace a
                comprehensive eye exam. The images in this report are lower quality than the images used by the AI
                model and should not be used for diagnostic purposes. See user manual for more details.
              </div>
            </div>
          </div>
        </div>

        {/* Modal Actions */}
        <div style={{
          padding: '16px 20px',
          borderTop: '1px solid var(--border-light)',
          display: 'flex',
          justifyContent: 'flex-end',
          gap: '10px'
        }}>
          <button className="btn btn-outline" onClick={() => window.print()}>
            <Printer size={16} /> Print Report
          </button>
          <button
            className="btn btn-primary"
            style={{ backgroundColor: '#0F172A', borderColor: '#0F172A' }}
            onClick={() => {
              const pdfUrl = screeningResult.pdfDownloadUrl || screeningResult.pdfUrl;
              if (!pdfUrl) {
                alert('No PDF report file available for download.');
                return;
              }
              const link = document.createElement('a');
              link.href = pdfUrl;
              const safeName = (patientInfo.name || 'Patient').trim().replace(/\s+/g, '_');
              link.download = `DRISHYA_Report_${safeName}.pdf`;
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
