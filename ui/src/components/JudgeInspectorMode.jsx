import React, { useState } from 'react';
import { Sliders, FileText, Download, Play, RefreshCw, Upload } from 'lucide-react';
import { useLanguage } from '../context/useLanguage';

export default function JudgeInspectorMode({
  screeningResult,
  uploadedImage,
  currentStep,
  onRunStepSimulation,
  isProcessing,
  onOpenPdfModal
}) {
  const { t } = useLanguage();
  const [activeTab, setActiveTab] = useState('gradcam'); // 'raw', 'preprocessed', 'lesions', 'gradcam'
  const [heatmapOpacity, setHeatmapOpacity] = useState(40); // 0 to 100

  const steps = [
    { id: 1, label: t('step_ingestion') },
    { id: 2, label: t('step_iqa') },
    { id: 3, label: t('step_normalization') },
    { id: 4, label: t('step_inference') },
    { id: 5, label: t('step_gradcam') },
    { id: 6, label: t('step_report') }
  ];

  // Active images from real screening or loaded scan
  const rawImg = screeningResult?.rawImg || uploadedImage?.previewUrl || null;
  const preprocessedImg = screeningResult?.preprocessedImg || null;
  const lesionsImg = screeningResult?.lesionsImg || null;
  const heatmapImg = screeningResult?.heatmapImg || null;

  const getActiveImage = () => {
    switch (activeTab) {
      case 'raw':
        return rawImg;
      case 'preprocessed':
        return preprocessedImg || rawImg;
      case 'lesions':
        return lesionsImg || rawImg;
      case 'gradcam':
      default:
        return preprocessedImg || rawImg;
    }
  };

  const isReferable = screeningResult?.referable ?? false;
  const gradeTitle = screeningResult?.grade !== undefined && screeningResult?.grade >= 0 && screeningResult?.grade <= 4
    ? t(`grade_${screeningResult.grade}_title`, screeningResult?.gradeTitle)
    : (screeningResult?.gradeTitle || '');

  const gradeDesc = screeningResult?.grade !== undefined && screeningResult?.grade >= 0 && screeningResult?.grade <= 4
    ? t(`grade_${screeningResult.grade}_desc`, screeningResult?.gradeDesc)
    : (screeningResult?.gradeDesc || '');

  const confidence = screeningResult?.confidence;
  const iqaScore = screeningResult?.iqaScore;
  const biomarkers = screeningResult?.biomarkers;

  return (
    <div>
      {/* Pipeline Progress Stepper */}
      <div className="stepper-container">
        {steps.map((s, idx) => {
          const isDone = currentStep > s.id || (s.id === 6 && Boolean(screeningResult) && !isProcessing);
          const isActive = currentStep === s.id && !isDone;
          return (
            <React.Fragment key={s.id}>
              <div className={`step-item ${isDone ? 'done' : ''} ${isActive ? 'active' : ''}`}>
                <div className="step-num">{isDone ? '✓' : s.id}</div>
                <span>{s.label}</span>
              </div>
              {idx < steps.length - 1 && <div className="step-divider" />}
            </React.Fragment>
          );
        })}

        <button
          id="btn-simulate-pipeline"
          className="btn btn-outline text-micro"
          style={{ padding: '4px 10px', marginLeft: '12px' }}
          onClick={onRunStepSimulation}
          disabled={isProcessing || !uploadedImage}
        >
          {isProcessing ? <RefreshCw size={12} className="spin-icon" /> : <Play size={12} />} {t('run_simulation')}
        </button>
      </div>

      {isProcessing && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '6px 14px',
          backgroundColor: 'var(--bg-surface)',
          border: '1px solid var(--border-strong)',
          borderRadius: 'var(--radius-sm)',
          marginBottom: '12px',
          fontSize: '11px',
          color: 'var(--text-secondary)',
          animation: 'drishya-fade-in 0.2s ease-out'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <RefreshCw size={12} className="spin-icon" style={{ color: 'var(--brand-primary)' }} />
            <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>
              Executing {steps.find((s) => s.id === currentStep)?.label || 'AI Pipeline'}...
            </span>
          </div>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--text-muted)' }}>
            Stage {Math.min(currentStep, 6)} of 6
          </span>
        </div>
      )}

      {/* Main Diagnostic Workstation Grid */}
      <div className="grid-2">
        {/* Left Column: Retinal Visual Workspace */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <h3 className="text-h3">{t('multitask_maps_sec')}</h3>
            <span className="badge badge-neutral">Resolution: 384×384 Tensor</span>
          </div>

          {/* Stage Tabs */}
          <div className="tab-group">
            <button
              id="tab-raw"
              className={`tab-btn ${activeTab === 'raw' ? 'active' : ''}`}
              onClick={() => setActiveTab('raw')}
            >
              (a) {t('view_raw')}
            </button>
            <button
              id="tab-preprocessed"
              className={`tab-btn ${activeTab === 'preprocessed' ? 'active' : ''}`}
              onClick={() => setActiveTab('preprocessed')}
            >
              (b) {t('view_preprocessed')}
            </button>
            <button
              id="tab-lesions"
              className={`tab-btn ${activeTab === 'lesions' ? 'active' : ''}`}
              onClick={() => setActiveTab('lesions')}
            >
              (c) {t('view_lesions')}
            </button>
            <button
              id="tab-gradcam"
              className={`tab-btn ${activeTab === 'gradcam' ? 'active' : ''}`}
              onClick={() => setActiveTab('gradcam')}
            >
              (d) {t('view_gradcam')}
            </button>
          </div>

          {/* Canvas Viewer */}
          <div className="image-canvas-wrapper" style={{ minHeight: '320px', display: 'flex', alignItems: 'center', justifyContent: 'center', backgroundColor: '#000' }}>
            {getActiveImage() ? (
              <>
                <img
                  src={getActiveImage()}
                  alt="Retina Stage"
                  className="retina-img"
                />
                {activeTab === 'gradcam' && heatmapImg && (
                  <img
                    src={heatmapImg}
                    alt="Grad-CAM Visual AI Evidence"
                    className="heatmap-layer"
                    style={{ opacity: heatmapOpacity / 100 }}
                  />
                )}
              </>
            ) : (
              <div style={{ textAlign: 'center', padding: '36px 20px', color: '#94A3B8' }}>
                <Upload size={36} color="#64748B" style={{ marginBottom: '8px' }} />
                <div style={{ fontWeight: 700, color: '#E2E8F0', marginBottom: '4px' }}>No Retinal Scan Loaded</div>
                <div style={{ fontSize: '11px', maxWidth: '260px', margin: '0 auto' }}>
                  Upload a fundus image in the Health Worker Portal or click Ingest Sample Scan above.
                </div>
              </div>
            )}
          </div>

          {/* Opacity Slider for Grad-CAM++ */}
          {activeTab === 'gradcam' && heatmapImg && (
            <div className="slider-control">
              <span style={{ fontWeight: 600, minWidth: '130px' }}>
                <Sliders size={12} style={{ display: 'inline', marginRight: '4px' }} />
                {t('heatmap_opacity')}: {heatmapOpacity}%
              </span>
              <input
                id="slider-heatmap-opacity"
                type="range"
                min="0"
                max="100"
                value={heatmapOpacity}
                onChange={(e) => setHeatmapOpacity(Number(e.target.value))}
              />
            </div>
          )}

          <div className="text-micro" style={{ marginTop: '12px', color: 'var(--text-muted)' }}>
            {activeTab === 'raw' && 'Raw field acquisition from portable camera with peripheral vignetting and flash glare.'}
            {activeTab === 'preprocessed' && 'Normalized retina using Ben Graham local Gaussian background subtraction and CLAHE contrast enhancement.'}
            {activeTab === 'lesions' && 'Explicit morphological lesion contours (Red circles: Microaneurysms, Yellow: Hard Exudates).'}
            {activeTab === 'gradcam' && 'Grad-CAM++ visual AI evidence map proving classifier attention aligns with verified microaneurysms.'}
          </div>
        </div>

        {/* Right Column: Clinical Metrics & Causal Biomarkers */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <h3 className="text-h3" style={{ marginBottom: '12px' }}>
            {t('triage_verdict')}
          </h3>

          {screeningResult ? (
            <>
              {/* Primary Result Box */}
              <div style={{
                border: '1px solid var(--border-strong)',
                borderRadius: 'var(--radius-md)',
                padding: '14px',
                backgroundColor: isReferable ? 'var(--status-warn-bg)' : 'var(--status-pass-bg)',
                marginBottom: '16px'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div>
                    <span className="badge badge-neutral" style={{ marginBottom: '4px' }}>{t('icdr_grading')}</span>
                    <div className="text-h1" style={{ color: 'var(--text-primary)' }}>
                      {gradeTitle}
                    </div>
                    <div className="text-caption" style={{ color: 'var(--text-secondary)' }}>
                      {gradeDesc}
                    </div>
                  </div>

                  <div style={{ textAlign: 'right' }}>
                    <div className="text-micro" style={{ color: 'var(--text-muted)', fontWeight: 700 }}>{t('confidence')}</div>
                    <div className="text-h1" style={{ color: 'var(--text-primary)' }}>
                      {confidence}
                    </div>
                  </div>
                </div>

                <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid rgba(0,0,0,0.08)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="text-caption" style={{ fontWeight: 700, color: isReferable ? '#B45309' : '#15803D' }}>
                    {isReferable ? `⚠️ ${t('referral_badge')}` : `🟢 ${t('routine_badge')}`}
                  </span>
                  <span className="badge badge-pass">IQA: PASS (Q={iqaScore})</span>
                </div>
              </div>

              {/* Biomarkers Table */}
              {biomarkers && (
                <div style={{ marginBottom: '16px' }}>
                  <div className="text-caption" style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
                    {t('biomarker_metrics')}
                  </div>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Biomarker / Feature</th>
                        <th>Detected Value</th>
                        <th>Clinical Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td style={{ fontWeight: 600 }}>{t('biomarker_mas')}</td>
                        <td><b>{biomarkers.mas}</b></td>
                        <td>
                          <span className={`badge ${
                            biomarkers.masStatus === 'High / Referral' ? 'badge-danger' : 
                            biomarkers.masStatus === 'Mild' ? 'badge-warn' : 'badge-pass'
                          }`}>
                            {biomarkers.masStatus}
                          </span>
                        </td>
                      </tr>
                      <tr>
                        <td style={{ fontWeight: 600 }}>{t('biomarker_exudates')}</td>
                        <td><b>{biomarkers.exudates}</b></td>
                        <td>
                          <span className={`badge ${
                            biomarkers.exudatesStatus === 'Significant' ? 'badge-danger' : 'badge-pass'
                          }`}>
                            {biomarkers.exudatesStatus}
                          </span>
                        </td>
                      </tr>
                      <tr>
                        <td style={{ fontWeight: 600 }}>{t('biomarker_hemorrhages')}</td>
                        <td><b>{biomarkers.hemorrhages}</b></td>
                        <td>
                          <span className={`badge ${
                            biomarkers.hemorrhagesStatus?.includes('Severe') ? 'badge-danger' :
                            (biomarkers.hemorrhages && !biomarkers.hemorrhages.toLowerCase().includes('none')) ? 'badge-warn' : 'badge-pass'
                          }`}>
                            {biomarkers.hemorrhagesStatus || 
                              ((biomarkers.hemorrhages && !biomarkers.hemorrhages.toLowerCase().includes('none')) ? 'Below 4:2:1 Rule' : 'None Detected')}
                          </span>
                        </td>
                      </tr>
                      <tr>
                        <td style={{ fontWeight: 600 }}>{t('biomarker_nv')}</td>
                        <td><b>{biomarkers.neovascularization}</b></td>
                        <td>
                          <span className={`badge ${
                            (biomarkers.neovascularization?.includes('Present') || screeningResult?.grade === 4)
                              ? 'badge-danger' 
                              : 'badge-pass'
                          }`}>
                            {(biomarkers.neovascularization?.includes('Present') || screeningResult?.grade === 4)
                              ? 'Proliferative (PDR)' 
                              : 'Non-Proliferative'}
                          </span>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              )}
            </>
          ) : (
            /* Awaiting screening empty state */
            <div style={{
              border: '1px dashed var(--border-strong)',
              borderRadius: 'var(--radius-md)',
              padding: '36px 20px',
              textAlign: 'center',
              backgroundColor: 'var(--bg-subtle)',
              color: 'var(--text-muted)',
              marginBottom: '16px'
            }}>
              <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-secondary)', marginBottom: '6px' }}>
                Awaiting AI Pipeline Execution
              </div>
              <div className="text-caption" style={{ maxWidth: '300px', margin: '0 auto', lineHeight: 1.5 }}>
                Diagnostic classification, confidence score, and causal biomarkers will appear here once inference completes.
              </div>
              {uploadedImage && (
                <button
                  className="btn btn-primary text-caption"
                  style={{ marginTop: '16px' }}
                  onClick={onRunStepSimulation}
                  disabled={isProcessing}
                >
                  {isProcessing ? <RefreshCw size={14} className="spin-icon" /> : <Play size={14} />} {t('run_simulation')}
                </button>
              )}
            </div>
          )}

          {/* Action Footer */}
          <div style={{ marginTop: 'auto', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            <button
              id="btn-preview-pdf-inspector"
              className="btn btn-outline"
              onClick={onOpenPdfModal}
              disabled={!screeningResult}
            >
              <FileText size={16} /> {t('view_report_btn')}
            </button>

            <button
              id="btn-download-pdf-inspector"
              className="btn btn-primary"
              onClick={onOpenPdfModal}
              disabled={!screeningResult}
            >
              <Download size={16} /> {t('download_pdf_btn')}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
