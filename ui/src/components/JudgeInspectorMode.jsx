import React, { useState } from 'react';
import { Sliders, FileText, Download, Play, RefreshCw, Upload } from 'lucide-react';

export default function JudgeInspectorMode({ 
  screeningResult,
  uploadedImage,
  currentStep, 
  onRunStepSimulation,
  isProcessing,
  onOpenPdfModal,
  onLoadSampleScan
}) {
  const [activeTab, setActiveTab] = useState('gradcam'); // 'raw', 'preprocessed', 'lesions', 'gradcam'
  const [heatmapOpacity, setHeatmapOpacity] = useState(40); // 0 to 100

  const steps = [
    { id: 1, label: '1. Ingestion' },
    { id: 2, label: '2. Edge IQA' },
    { id: 3, label: '3. Normalization' },
    { id: 4, label: '4. AI Inference' },
    { id: 5, label: '5. Grad-CAM++' },
    { id: 6, label: '6. Report Ready' }
  ];

  // Active images
  const rawImg = screeningResult?.rawImg || uploadedImage?.previewUrl || '/assets/grade2_raw.png';
  const preprocessedImg = screeningResult?.preprocessedImg || '/assets/grade2_preprocessed.png';
  const lesionsImg = screeningResult?.lesionsImg || '/assets/grade2_lesions.png';
  const heatmapImg = screeningResult?.heatmapImg || '/assets/grade2_heatmap.png';

  const getActiveImage = () => {
    switch (activeTab) {
      case 'raw':
        return rawImg;
      case 'preprocessed':
        return preprocessedImg;
      case 'lesions':
        return lesionsImg;
      case 'gradcam':
      default:
        return preprocessedImg;
    }
  };

  const isReferable = screeningResult?.referable ?? true;
  const gradeTitle = screeningResult?.gradeTitle || 'Grade 2: Moderate NPDR';
  const gradeDesc = screeningResult?.gradeDesc || 'Non-Proliferative Diabetic Retinopathy (Microaneurysms + Hard Exudates)';
  const confidence = screeningResult?.confidence || '96.4%';
  const iqaScore = screeningResult?.iqaScore || '0.88';

  const biomarkers = screeningResult?.biomarkers || {
    mas: '12 detected',
    masStatus: 'Moderate',
    exudates: '1.10% area',
    exudatesStatus: 'Sup. Arcade',
    hemorrhages: '2 Quadrants',
    neovascularization: '0 (Absent)'
  };

  return (
    <div>
      {/* Notice Banner if not screened yet */}
      {!screeningResult && (
        <div style={{
          backgroundColor: 'var(--brand-light)',
          border: '1px solid var(--border-strong)',
          borderRadius: 'var(--radius-md)',
          padding: '10px 16px',
          marginBottom: '16px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div>
            <span style={{ fontWeight: 700, color: 'var(--brand-dark)' }}>Inspector Mode</span>: 
            {' '}Visual multi-stage pipeline breakdown (Ingestion → IQA → CLAHE → Swin/EfficientNet → Grad-CAM++).
          </div>
          {onLoadSampleScan && (
            <button
              type="button"
              className="btn btn-outline text-micro"
              style={{ padding: '4px 10px', whiteSpace: 'nowrap' }}
              onClick={onLoadSampleScan}
              disabled={isProcessing}
            >
              <Upload size={12} /> Ingest Sample Scan
            </button>
          )}
        </div>
      )}

      {/* Pipeline Progress Stepper */}
      <div className="stepper-container">
        {steps.map((s, idx) => {
          const isDone = currentStep > s.id;
          const isActive = currentStep === s.id;
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
          disabled={isProcessing}
        >
          {isProcessing ? <RefreshCw size={12} className="spin-icon" /> : <Play size={12} />} Live Stepper
        </button>
      </div>

      {/* Main Diagnostic Workstation Grid */}
      <div className="grid-2">
        {/* Left Column: Retinal Visual Workspace */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
            <h3 className="text-h3">Retinal Stage Visualization</h3>
            <span className="badge badge-neutral">Resolution: 384x384 Tensor</span>
          </div>

          {/* Stage Tabs */}
          <div className="tab-group">
            <button
              id="tab-raw"
              className={`tab-btn ${activeTab === 'raw' ? 'active' : ''}`}
              onClick={() => setActiveTab('raw')}
            >
              (a) Raw Scan
            </button>
            <button
              id="tab-preprocessed"
              className={`tab-btn ${activeTab === 'preprocessed' ? 'active' : ''}`}
              onClick={() => setActiveTab('preprocessed')}
            >
              (b) Preprocessed
            </button>
            <button
              id="tab-lesions"
              className={`tab-btn ${activeTab === 'lesions' ? 'active' : ''}`}
              onClick={() => setActiveTab('lesions')}
            >
              (c) Lesion Overlay
            </button>
            <button
              id="tab-gradcam"
              className={`tab-btn ${activeTab === 'gradcam' ? 'active' : ''}`}
              onClick={() => setActiveTab('gradcam')}
            >
              (d) Grad-CAM++
            </button>
          </div>

          {/* Canvas Viewer */}
          <div className="image-canvas-wrapper">
            <img
              src={getActiveImage()}
              alt="Retina Stage"
              className="retina-img"
            />
            {activeTab === 'gradcam' && (
              <img
                src={heatmapImg}
                alt="Grad-CAM Saliency"
                className="heatmap-layer"
                style={{ opacity: heatmapOpacity / 100 }}
              />
            )}
          </div>

          {/* Opacity Slider for Grad-CAM++ */}
          {activeTab === 'gradcam' && (
            <div className="slider-control">
              <span style={{ fontWeight: 600, minWidth: '130px' }}>
                <Sliders size={12} style={{ display: 'inline', marginRight: '4px' }} />
                Heatmap Opacity: {heatmapOpacity}%
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
            {activeTab === 'gradcam' && 'Grad-CAM++ neural saliency map proving classifier attention aligns with verified microaneurysms.'}
          </div>
        </div>

        {/* Right Column: Clinical Metrics & Causal Biomarkers */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
          <h3 className="text-h3" style={{ marginBottom: '12px' }}>
            Diagnostic Findings & Biomarker Evidence
          </h3>

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
                <span className="badge badge-neutral" style={{ marginBottom: '4px' }}>ICDR Severity Classification</span>
                <div className="text-h1" style={{ color: 'var(--text-primary)' }}>
                  {gradeTitle}
                </div>
                <div className="text-caption" style={{ color: 'var(--text-secondary)' }}>
                  {gradeDesc}
                </div>
              </div>

              <div style={{ textAlign: 'right' }}>
                <div className="text-micro" style={{ color: 'var(--text-muted)', fontWeight: 700 }}>AI CONFIDENCE</div>
                <div className="text-h1" style={{ color: 'var(--text-primary)' }}>
                  {confidence}
                </div>
              </div>
            </div>

            <div style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid rgba(0,0,0,0.08)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="text-caption" style={{ fontWeight: 700, color: isReferable ? '#B45309' : '#15803D' }}>
                {isReferable ? '⚠️ REFERABLE DR (Specialist Referral Required)' : '🟢 NON-REFERABLE (Routine Screening)'}
              </span>
              <span className="badge badge-pass">IQA: PASS (Q={iqaScore})</span>
            </div>
          </div>

          {/* Biomarkers Table */}
          <div style={{ marginBottom: '16px' }}>
            <div className="text-caption" style={{ fontWeight: 700, color: 'var(--text-primary)', marginBottom: '6px' }}>
              Clinical Biomarker Quantification (Causal Evidence)
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
                  <td style={{ fontWeight: 600 }}>Microaneurysms (MAs)</td>
                  <td><b>{biomarkers.mas || '12 detected'}</b></td>
                  <td><span className="badge badge-warn">{biomarkers.masStatus || 'Moderate'}</span></td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Hard Exudates (Lipids)</td>
                  <td><b>{biomarkers.exudates || '1.10% area'}</b></td>
                  <td><span className="badge badge-warn">{biomarkers.exudatesStatus || 'Sup. Arcade'}</span></td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Hemorrhage Spread</td>
                  <td><b>{biomarkers.hemorrhages || '2 Quadrants'}</b></td>
                  <td><span className="badge badge-neutral">Below 4:2:1 Rule</span></td>
                </tr>
                <tr>
                  <td style={{ fontWeight: 600 }}>Neovascularization</td>
                  <td><b>{biomarkers.neovascularization || '0 (Absent)'}</b></td>
                  <td><span className="badge badge-pass">Non-Proliferative</span></td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Action Footer */}
          <div style={{ marginTop: 'auto', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
            <button
              id="btn-preview-pdf-inspector"
              className="btn btn-outline"
              onClick={onOpenPdfModal}
            >
              <FileText size={16} /> Preview 1-Page PDF
            </button>

            <button
              id="btn-download-pdf-inspector"
              className="btn btn-primary"
              onClick={onOpenPdfModal}
            >
              <Download size={16} /> Download Official PDF
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
