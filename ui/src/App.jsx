import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import HealthWorkerMode from './components/HealthWorkerMode';
import JudgeInspectorMode from './components/JudgeInspectorMode';
import PdfPreviewModal from './components/PdfPreviewModal';

// Preset Clinical Data
const PRESETS = {
  grade2: {
    grade: 'grade2',
    gradeTitle: 'Grade 2: Moderate NPDR',
    gradeDesc: 'Non-Proliferative Diabetic Retinopathy (Microaneurysms + Hard Exudates)',
    referable: true,
    confidence: '96.4%',
    iqaScore: '0.88 (PASS)',
    actionRecommendation: '⚠️ REFERRAL REQUIRED: Slit-lamp exam at District Hospital within 4 weeks.',
    rawImg: '/assets/grade2_raw.png',
    preprocessedImg: '/assets/grade2_preprocessed.png',
    lesionsImg: '/assets/grade2_lesions.png',
    heatmapImg: '/assets/grade2_heatmap.png',
    gradcamImg: '/assets/grade2_gradcam.png',
    biomarkers: {
      mas: '12 detected',
      masStatus: 'Moderate',
      exudates: '1.10% area cluster',
      exudatesStatus: 'Sup. Arcade',
      hemorrhages: '2 Quadrants',
      neovascularization: '0 (Absent)'
    }
  },
  grade0: {
    grade: 'grade0',
    gradeTitle: 'Grade 0: Normal Retina',
    gradeDesc: 'No abnormalities or diabetic lesions detected.',
    referable: false,
    confidence: '99.1%',
    iqaScore: '0.94 (PASS)',
    actionRecommendation: '🟢 ROUTINE SCREENING: Rescreen at Primary Health Center in 12 months.',
    rawImg: '/assets/grade0_raw.png',
    preprocessedImg: '/assets/grade0_preprocessed.png',
    lesionsImg: '/assets/grade0_lesions.png',
    heatmapImg: '/assets/grade0_heatmap.png',
    gradcamImg: '/assets/grade0_gradcam.png',
    biomarkers: {
      mas: '0 detected',
      masStatus: 'Normal',
      exudates: '0.00% (None)',
      exudatesStatus: 'Absent',
      hemorrhages: '0 Quadrants',
      neovascularization: '0 (Absent)'
    }
  },
  grade3: {
    grade: 'grade3',
    gradeTitle: 'Grade 3: Severe NPDR',
    gradeDesc: 'Severe Non-Proliferative DR (4:2:1 Rule: Severe Hemorrhages in 4 Quadrants)',
    referable: true,
    confidence: '95.2%',
    iqaScore: '0.82 (PASS)',
    actionRecommendation: '🛑 URGENT REFERRAL: Immediate retinal specialist evaluation (within 1-2 weeks).',
    rawImg: '/assets/grade3_raw.png',
    preprocessedImg: '/assets/grade3_preprocessed.png',
    lesionsImg: '/assets/grade3_lesions.png',
    heatmapImg: '/assets/grade3_heatmap.png',
    gradcamImg: '/assets/grade3_gradcam.png',
    biomarkers: {
      mas: '34 detected',
      masStatus: 'Severe',
      exudates: '3.45% area',
      exudatesStatus: 'Extensive',
      hemorrhages: '4 Quadrants',
      neovascularization: '0 (Absent)'
    }
  },
  ungradable: {
    grade: 'ungradable',
    gradeTitle: 'UNGRADABLE SCAN (IQA REJECT)',
    gradeDesc: 'Severe motion blur & underexposure detected by on-device edge IQA.',
    referable: false,
    confidence: '0.0%',
    iqaScore: '0.24 (REJECT)',
    actionRecommendation: '🛑 RETAKE SCAN IMMEDIATELY before patient leaves clinic.',
    rawImg: '/assets/ungradable_raw.png',
    preprocessedImg: '/assets/ungradable_raw.png',
    lesionsImg: '/assets/ungradable_raw.png',
    heatmapImg: '/assets/grade0_heatmap.png',
    gradcamImg: '/assets/ungradable_raw.png',
    biomarkers: {
      mas: 'Ungradable',
      masStatus: 'N/A',
      exudates: 'Ungradable',
      exudatesStatus: 'N/A',
      hemorrhages: 'Ungradable',
      neovascularization: 'N/A'
    }
  }
};

export default function App() {
  const [activeMode, setActiveMode] = useState('health-worker'); // 'health-worker' or 'judge-inspector'
  const [currentPresetKey, setCurrentPresetKey] = useState('grade2');
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(6);
  const [showPdfModal, setShowPdfModal] = useState(false);
  
  const [patientInfo, setPatientInfo] = useState({
    name: 'Ramesh Kumar',
    phone: '+91 98765 43210',
    abhaId: '91-4820-1940-52'
  });

  const currentPreset = PRESETS[currentPresetKey];

  // Live step-by-step pipeline simulation
  const handleRunSimulation = () => {
    setIsProcessing(true);
    setCurrentStep(1);

    const stepIntervals = [
      setTimeout(() => setCurrentStep(2), 600),
      setTimeout(() => setCurrentStep(3), 1200),
      setTimeout(() => setCurrentStep(4), 1800),
      setTimeout(() => setCurrentStep(5), 2400),
      setTimeout(() => {
        setCurrentStep(6);
        setIsProcessing(false);
      }, 3000)
    ];
  };

  const handleSetPreset = (key) => {
    setCurrentPresetKey(key);
    handleRunSimulation();
  };

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <Sidebar
        activeMode={activeMode}
        setActiveMode={setActiveMode}
        currentPreset={currentPresetKey}
        setPreset={handleSetPreset}
        isProcessing={isProcessing}
      />

      {/* Main Workspace */}
      <main className="main-content">
        {/* Top App Bar */}
        <header className="top-bar">
          <div className="top-bar-left">
            <span className="top-title">
              {activeMode === 'health-worker' ? '📱 ASHA Health Worker Screening Portal' : '🔬 Judge Pipeline & Explainability Inspector'}
            </span>
          </div>

          <div className="top-meta">
            <span className="meta-pill">🏥 PHC Rampur (Zone 4)</span>
            <span className="meta-pill">🆔 ABHA: {patientInfo.abhaId}</span>
            <span className="meta-pill" style={{ backgroundColor: '#DCFCE7', color: '#166534', borderColor: '#86EFAC' }}>
              ⚡ Model: FP16 EfficientNet-B4 MTL
            </span>
          </div>
        </header>

        {/* Workspace Body */}
        <div className="workspace">
          {activeMode === 'health-worker' ? (
            <HealthWorkerMode
              presetData={currentPreset}
              patientInfo={patientInfo}
              setPatientInfo={setPatientInfo}
              onRunScreening={handleRunSimulation}
              isProcessing={isProcessing}
              onOpenPdfModal={() => setShowPdfModal(true)}
            />
          ) : (
            <JudgeInspectorMode
              presetData={currentPreset}
              currentStep={currentStep}
              onRunStepSimulation={handleRunSimulation}
              isProcessing={isProcessing}
              onOpenPdfModal={() => setShowPdfModal(true)}
            />
          )}
        </div>
      </main>

      {/* 1-Page Clinical PDF Modal */}
      <PdfPreviewModal
        isOpen={showPdfModal}
        onClose={() => setShowPdfModal(false)}
        presetData={currentPreset}
        patientInfo={patientInfo}
      />
    </div>
  );
}
