import React, { useState, useEffect, useRef } from 'react';
import { Menu, AlertCircle, AlertTriangle, Info, X, ShieldCheck } from 'lucide-react';
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
  const [notification, setNotification] = useState(null);
  const stepTimersRef = useRef([]);

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

  // Accessible keyboard listener (Escape closes mobile drawer) and scroll locking
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && sidebarOpen) {
        setSidebarOpen(false);
      }
    };
    if (sidebarOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = '';
    };
  }, [sidebarOpen]);

  // Auto-dismiss clinical notifications after 7 seconds
  useEffect(() => {
    if (!notification) return;
    const timer = setTimeout(() => {
      setNotification(null);
    }, 7000);
    return () => clearTimeout(timer);
  }, [notification]);

  // Clean up any pending step timers on unmount
  useEffect(() => {
    return () => {
      stepTimersRef.current.forEach(clearTimeout);
    };
  }, []);

  const showNotification = (type, title, message) => {
    setNotification({ type, title, message });
  };

  const clearNotification = () => {
    setNotification(null);
  };

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
    clearNotification();
  };

  // Clear image to allow recapture or new upload
  const handleClearImage = () => {
    setUploadedImage(null);
    setScreeningResult(null);
    setCurrentStep(1);
    clearNotification();
  };

  // Live step-by-step pipeline execution
  const handleRunScreening = async () => {
    if (!uploadedImage?.file) {
      showNotification(
        'warn',
        t('alert_upload_first', 'Please upload or select a retinal fundus scan first.'),
        'Capture a new fundus photograph or choose a pre-loaded validation sample before running AI analysis.'
      );
      return;
    }

    clearNotification();
    setIsProcessing(true);
    setCurrentStep(1);

    stepTimersRef.current.forEach(clearTimeout);
    stepTimersRef.current = [
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
        if (!response.ok && (response.status === 404 || response.status === 502)) {
          response = await fetch('http://localhost:8000/api/screen-patient', {
            method: 'POST',
            body: formData
          });
        }
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
          showNotification(
            'warn',
            'Image Quality Assessment (IQA) Rejected',
            data.action || 'The uploaded scan has insufficient focus or illumination. Please recapture immediately.'
          );
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
        stepTimersRef.current.forEach(clearTimeout);
        setCurrentStep(1);
        showNotification(
          'error',
          `Screening Request Failed (${response.status})`,
          errorData.detail || response.statusText || 'The screening request could not be processed by the server.'
        );
      }
    } catch (err) {
      stepTimersRef.current.forEach(clearTimeout);
      setCurrentStep(1);
      console.error('Inference error:', err);
      showNotification(
        'error',
        'Could Not Connect to AI Engine',
        `Unable to reach the DRISHYA screening engine (${err.message}). Ensure the backend server is running on port 8000.`
      );
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
              onClick={() => setSidebarOpen((prev) => !prev)}
              aria-label="Toggle Navigation Menu"
              aria-expanded={sidebarOpen}
              aria-controls="sidebar-navigation"
            >
              <Menu size={18} />
            </button>
            <span className="top-title">
              {activeMode === 'health-worker' ? t('top_title_hw') : t('top_title_judge')}
            </span>
          </div>
          <div className="top-bar-right">
            <span className="edge-ready-pill" title="100% Offline Edge Inference Ready">
              <span className="edge-ready-dot" />
              <ShieldCheck size={13} />
              <span>{t('edge_tag')}</span>
            </span>
            <LanguageSelector />
          </div>
        </header>

        {/* Workspace Body */}
        <div className="workspace">
          {/* Accessible Clinical Notice / Error Banner */}
          {notification && (
            <div
              className={`clinical-notice-banner ${notification.type}`}
              role="alert"
              aria-live="polite"
            >
              <div className="clinical-notice-icon">
                {notification.type === 'error' && <AlertCircle size={18} />}
                {notification.type === 'warn' && <AlertTriangle size={18} />}
                {notification.type === 'info' && <Info size={18} />}
              </div>
              <div className="clinical-notice-content">
                <div className="clinical-notice-title">{notification.title}</div>
                {notification.message && (
                  <div className="clinical-notice-desc">{notification.message}</div>
                )}
              </div>
              <button
                type="button"
                className="clinical-notice-close"
                onClick={clearNotification}
                aria-label="Dismiss notification"
              >
                <X size={15} />
              </button>
            </div>
          )}

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
