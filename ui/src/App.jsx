import React, { useState, useEffect } from 'react';
import { Menu } from 'lucide-react';
import Sidebar from './components/Sidebar';
import HealthWorkerMode from './components/HealthWorkerMode';
import JudgeInspectorMode from './components/JudgeInspectorMode';
import PdfPreviewModal from './components/PdfPreviewModal';
import LanguageSelector from './components/LanguageSelector';
import { LanguageProvider } from './context/LanguageContext';
import { useLanguage } from './context/useLanguage';

function AppContent() {
  const { t } = useLanguage();
  const [activeMode, setActiveMode] = useState('health-worker'); // 'health-worker' or 'judge-inspector'
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState(1);
  const [showPdfModal, setShowPdfModal] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Automatically fall back to Health Worker portal on screens < 1000px
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 1000 && activeMode === 'judge-inspector') {
        setActiveMode('health-worker');
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [activeMode]);

  // Patient registration info
  const [patientInfo, setPatientInfo] = useState({
    name: '',
    age: '',
    gender: 'Male',
    phone: '',
    abhaId: ''
  });

  // Uploaded retinal fundus scan
  const [uploadedImage, setUploadedImage] = useState(null);

  // Clinical screening analysis result
  const [screeningResult, setScreeningResult] = useState(null);

  // File selection handler
  const handleImageSelected = (file, previewUrl, name) => {
    setUploadedImage({ file, previewUrl, name });
    setScreeningResult(null);
    setCurrentStep(1);
  };

  // Clear image to allow recapture or new upload
  const handleClearImage = () => {
    setUploadedImage(null);
    setScreeningResult(null);
    setCurrentStep(1);
  };

  // Live step-by-step pipeline execution
  const handleRunScreening = async () => {
    if (!uploadedImage?.file) {
      alert(t('alert_upload_first', 'Please upload or select a retinal fundus scan first.'));
      return;
    }

    setIsProcessing(true);
    setCurrentStep(1);

    const stepTimers = [
      setTimeout(() => setCurrentStep(2), 500),
      setTimeout(() => setCurrentStep(3), 1100),
      setTimeout(() => setCurrentStep(4), 1900),
      setTimeout(() => setCurrentStep(5), 2700),
    ];

    try {
      const formData = new FormData();
      formData.append('file', uploadedImage.file);
      formData.append('name', patientInfo.name.trim() || 'Anonymous Patient');
      formData.append('age', patientInfo.age.trim() || 'N/A');
      formData.append('gender', patientInfo.gender || 'N/A');
      formData.append('phone', patientInfo.phone.trim() || 'N/A');
      formData.append('abha_id', patientInfo.abhaId.trim() || 'N/A');

      let response;
      try {
        response = await fetch('/api/screen-patient', {
          method: 'POST',
          body: formData
        });
      } catch {
        response = await fetch('http://localhost:8000/api/screen-patient', {
          method: 'POST',
          body: formData
        });
      }

      if (response.ok) {
        const data = await response.json();

        if (!data.success && data.iqa_pass === false) {
          setScreeningResult({
            success: false,
            iqaPass: false,
            iqaScore: data.q_score,
            grade: -1,
            gradeTitle: data.grade_title || 'Ungradable (Quality Rejected)',
            gradeDesc: data.action || 'Scan failed illumination or focus requirements.',
            actionRecommendation: data.action || '🛑 RETAKE SCAN REQUIRED: Recapture retinal photo immediately before patient leaves.',
            confidence: 'N/A',
            referable: false,
            rawImg: uploadedImage.previewUrl,
            preprocessedImg: null,
            lesionsImg: null,
            heatmapImg: null,
            gradcamImg: null,
            biomarkers: null,
            pdfDownloadUrl: null
          });
        } else {
          setScreeningResult({
            success: data.success,
            iqaPass: data.iqa_pass ?? true,
            iqaScore: data.q_score,
            grade: data.grade,
            gradeTitle: data.grade_title,
            gradeDesc: data.grade_desc,
            actionRecommendation: data.action || (data.referable_dr ? 'Refer to Ophthalmologist within 2-4 weeks' : 'Routine Rescreening in 12 Months'),
            confidence: data.confidence,
            referable: data.referable_dr ?? false,
            rawImg: data.files?.raw_path ? `/outputs/${data.files.raw_path.split('/').pop()}` : uploadedImage.previewUrl,
            preprocessedImg: data.files?.preprocessed_path ? `/outputs/${data.files.preprocessed_path.split('/').pop()}` : null,
            lesionsImg: data.files?.lesion_path ? `/outputs/${data.files.lesion_path.split('/').pop()}` : null,
            heatmapImg: data.files?.heatmap_path ? `/outputs/${data.files.heatmap_path.split('/').pop()}` : null,
            gradcamImg: data.files?.gradcam_path ? `/outputs/${data.files.gradcam_path.split('/').pop()}` : null,
            biomarkers: {
              mas: `${data.biomarkers?.microaneurysms ?? 0} detected`,
              masStatus: (data.biomarkers?.microaneurysms ?? 0) > 5 ? 'High / Referral' : (data.biomarkers?.microaneurysms ?? 0) > 0 ? 'Mild' : 'None',
              exudates: `${data.biomarkers?.exudate_area_pct ?? '0.00%'} area`,
              exudatesStatus: parseFloat(data.biomarkers?.exudate_area_pct || '0') > 1 ? 'Significant' : 'Low / Absent',
              hemorrhages: data.biomarkers?.hemorrhage_quadrants ? `${data.biomarkers.hemorrhage_quadrants} Quadrants` : 'None detected',
              hemorrhagesStatus: (data.biomarkers?.hemorrhage_quadrants || 0) >= 4 ? 'Meets 4:2:1 Rule (Severe)' : (data.biomarkers?.hemorrhage_quadrants || 0) > 0 ? 'Below 4:2:1 Rule' : 'None Detected',
              neovascularization: data.grade === 4 ? 'Present (PDR)' : '0 (Absent)',
              nvStatus: data.grade === 4 ? 'Proliferative (PDR)' : 'Non-Proliferative',
              macularRisk: data.biomarkers?.macular_risk || 'Low Risk',
              macularDetail: data.biomarkers?.macular_detail || 'No lesions in macular zone'
            },
            pdfDownloadUrl: data.pdf_download_url
          });
        }
        setCurrentStep(6);
        await new Promise((resolve) => setTimeout(resolve, 350));
      } else {
        const errorData = await response.json().catch(() => ({}));
        stepTimers.forEach(clearTimeout);
        setCurrentStep(1);
        alert(`Screening failed (${response.status}): ${errorData.detail || response.statusText}`);
      }
    } catch (err) {
      stepTimers.forEach(clearTimeout);
      setCurrentStep(1);
      console.error('Inference error:', err);
      alert(`Could not connect to AI Engine: ${err.message}. Please ensure the server is running.`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar Navigation (Desktop sidebar or Mobile slide-over drawer) */}
      <Sidebar
        activeMode={activeMode}
        setActiveMode={setActiveMode}
        isProcessing={isProcessing}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      {/* Main Workspace */}
      <main className="main-content">
        {/* Top App Bar */}
        <header className="top-bar">
          <div className="top-bar-left">
            <button
              type="button"
              id="btn-hamburger"
              className="hamburger-btn"
              onClick={() => setSidebarOpen(true)}
              aria-label="Open Navigation Menu"
            >
              <Menu size={18} />
            </button>
            <span className="top-title">
              {activeMode === 'health-worker' ? t('top_title_hw') : t('top_title_judge')}
            </span>
          </div>
          <div className="top-bar-right" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <LanguageSelector />
          </div>
        </header>

        {/* Workspace Body */}
        <div className="workspace">
          {activeMode === 'health-worker' ? (
            <HealthWorkerMode
              patientInfo={patientInfo}
              setPatientInfo={setPatientInfo}
              uploadedImage={uploadedImage}
              onImageSelected={handleImageSelected}
              onClearImage={handleClearImage}
              screeningResult={screeningResult}
              onRunScreening={handleRunScreening}
              isProcessing={isProcessing}
              currentStep={currentStep}
              onOpenPdfModal={() => setShowPdfModal(true)}
            />
          ) : (
            <JudgeInspectorMode
              screeningResult={screeningResult}
              uploadedImage={uploadedImage}
              currentStep={currentStep}
              onRunStepSimulation={handleRunScreening}
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
        screeningResult={screeningResult}
        patientInfo={patientInfo}
      />
    </div>
  );
}

export default function App() {
  return (
    <LanguageProvider>
      <AppContent />
    </LanguageProvider>
  );
}
