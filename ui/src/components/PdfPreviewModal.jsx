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
              borderBottom: '2.5px solid #2BA882'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <img
                  src="/assets/drishyalogo.jpeg"
                  alt="DRISHYA Logo"
                  style={{ height: '36px', maxWidth: '72px', objectFit: 'contain', borderRadius: '4px' }}
                />
                <div>
                  <div style={{ fontSize: '18px', fontWeight: 900, color: '#1F8A6C', letterSpacing: '-0.3px' }}>
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
                  backgroundColor: '#2BA882',
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
                    ['PATIENT NAME', patientInfo.name],
                    ['ABHA ID', patientInfo.abhaId],
                    ['AGE / SEX', '54 Yrs / Male'],
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
                  backgroundColor: '#2BA882',
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

            {/* ── Results Section ─────────────────────────── */}
            <div style={{
              backgroundColor: '#2BA882',
              color: '#fff',
              padding: '4px 12px',
              fontWeight: 700,
              fontSize: '11px',
              textTransform: 'uppercase',
              letterSpacing: '0.5px'
            }}>
              Results
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 0.9fr' }}>
              {/* Left: Details */}
              <div style={{ borderRight: '1px solid #E2E8F0' }}>
                {[
                  ['CONDITION', 'Diabetic Retinopathy'],
                  ['DIAGNOSIS', screeningResult.gradeDesc || 'No DR detected ETDRS level 20 and lower and no macular edema.'],
                  ['DIAGNOSIS CODE', 'E11.9'],
                  ['CARE PLAN', screeningResult.referable ? 'Refer within 3-4 weeks' : 'Retest in 12 months'],
                  ['INTERPRETATION', 'Results were produced by DRISHYA, an AI system that provides automated retinal interpretation.'],
                ].map(([label, val], i) => (
                  <div key={i} style={{
                    display: 'grid',
                    gridTemplateColumns: '130px 1fr',
                    borderBottom: '1px solid #E2E8F0'
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
                padding: '24px',
                gap: '6px'
              }}>
                <div style={{
                  fontSize: '12px',
                  fontWeight: 700,
                  color: '#1F8A6C',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>Result</div>
                <div style={{
                  fontSize: '28px',
                  fontWeight: 900,
                  color: resultColor,
                  textAlign: 'center',
                  lineHeight: 1.15,
                  letterSpacing: '-0.5px'
                }}>
                  {resultText}
                </div>
              </div>
            </div>

            {/* ── Bottom Half: AI Facts + Fundus Images ──── */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.15fr' }}>

              {/* Left: AI Facts */}
              <div style={{ borderRight: '1px solid #E2E8F0' }}>
                <div style={{
                  backgroundColor: '#2BA882',
                  color: '#fff',
                  padding: '4px 12px',
                  fontWeight: 700,
                  fontSize: '11px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Augmented Intelligence Facts
                </div>
                <div style={{ padding: '8px 10px' }}>
                  <div className="text-micro" style={{ fontStyle: 'italic', color: '#64748B', marginBottom: '6px' }}>
                    The table below describes the AI model providing the interpretation.
                  </div>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '10px' }}>
                    <tbody>
                      {[
                        { label: 'AI Description', val: '', isHeader: true },
                        { label: 'Product Name', val: 'DRISHYA Retinal AI v1.0' },
                        { label: 'Type of Diagnostic', val: 'Autonomous AI' },
                        { label: 'Disease', val: 'Diabetic retinopathy, inclusive of macular edema' },
                        { label: 'Intended For', val: 'Adults with diabetes (Rx only)' },
                        { label: 'AI Performance Data', val: '', isHeader: true },
                        { label: 'Confidence', val: screeningResult.confidence || '96.4%' },
                        { label: 'Sensitivity', val: '94.2%' },
                        { label: 'Specificity', val: '91.8%' },
                        { label: 'Diagnosability', val: '96.0%' },
                      ].map((row, i) => (
                        <tr key={i} style={{
                          backgroundColor: row.isHeader ? '#E6F7F1' : 'transparent'
                        }}>
                          <td style={{
                            padding: '3px 6px',
                            fontWeight: row.isHeader ? 700 : 400,
                            color: '#334155',
                            borderBottom: '1px solid #E2E8F0',
                            fontSize: '10px',
                          }}>{row.label}</td>
                          <td style={{
                            padding: '3px 6px',
                            color: '#334155',
                            borderBottom: '1px solid #E2E8F0',
                            fontSize: '10px',
                          }}>{row.val}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Right: 3 Fundus Images - PROMINENTLY DISPLAYED */}
              <div>
                <div style={{
                  backgroundColor: '#2BA882',
                  color: '#fff',
                  padding: '4px 12px',
                  fontWeight: 700,
                  fontSize: '11px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  3 Fundus Images Used in Exam
                </div>
                <div style={{ padding: '10px' }}>
                  {/* Top row: 2 images side by side */}
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: '8px',
                    marginBottom: '8px'
                  }}>
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
                      <div className="text-micro" style={{ fontWeight: 700, marginTop: '4px' }}>
                        (a) Preprocessed Retina
                      </div>
                    </div>
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
                      <div className="text-micro" style={{ fontWeight: 700, marginTop: '4px' }}>
                        (b) Detected Lesions
                      </div>
                    </div>
                  </div>

                  {/* Bottom row: Grad-CAM large and centered */}
                  <div style={{ textAlign: 'center' }}>
                    <img
                      src={screeningResult.gradcamImg || screeningResult.rawImg}
                      alt="Grad-CAM++ Attention"
                      style={{
                        width: '60%',
                        aspectRatio: '1',
                        objectFit: 'contain',
                        backgroundColor: '#000',
                        borderRadius: '4px',
                        border: '2px solid #2BA882'
                      }}
                    />
                    <div className="text-micro" style={{ fontWeight: 700, marginTop: '4px' }}>
                      (c) Grad-CAM++ Attention Map
                    </div>
                    <div className="text-micro" style={{ color: '#64748B' }}>
                      Neural saliency focus — areas of highest diagnostic interest
                    </div>
                  </div>

                  <div className="text-micro" style={{
                    fontStyle: 'italic',
                    color: '#64748B',
                    textAlign: 'center',
                    marginTop: '8px',
                    borderTop: '1px solid #E2E8F0',
                    paddingTop: '4px'
                  }}>
                    Image orientation and labeling is for reference only and should not be used for diagnostic purposes.
                  </div>
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
            style={{ backgroundColor: '#2BA882', borderColor: '#2BA882' }}
            onClick={() => {
              const link = document.createElement('a');
              link.href = screeningResult.pdfDownloadUrl || screeningResult.pdfUrl || '/assets/DRISHYA_Clinical_Report.pdf';
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
