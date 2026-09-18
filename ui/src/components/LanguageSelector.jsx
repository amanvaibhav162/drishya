import React from 'react';
import { Languages } from 'lucide-react';
import { useLanguage } from '../context/useLanguage';

export default function LanguageSelector() {
  const { language, setLanguage } = useLanguage();

  return (
    <div className="language-selector" role="group" aria-label="Language selection">
      <Languages size={15} className="lang-icon" aria-hidden="true" />
      <button
        type="button"
        id="btn-lang-en"
        className={`lang-btn ${language === 'en' ? 'active' : ''}`}
        onClick={() => setLanguage('en')}
        title="Switch to English"
        aria-label="Switch to English"
        aria-pressed={language === 'en'}
      >
        <span className="lang-full">English</span>
        <span className="lang-short">EN</span>
        <span className="lang-mini">EN</span>
      </button>
      <span className="lang-divider" aria-hidden="true" />
      <button
        type="button"
        id="btn-lang-hi"
        className={`lang-btn ${language === 'hi' ? 'active' : ''}`}
        onClick={() => setLanguage('hi')}
        title="हिन्दी में बदलें"
        aria-label="हिन्दी में बदलें"
        aria-pressed={language === 'hi'}
      >
        <span className="lang-full">हिन्दी</span>
        <span className="lang-short">हिन्दी</span>
        <span className="lang-mini">हिं</span>
      </button>
      <span className="lang-divider" aria-hidden="true" />
      <button
        type="button"
        id="btn-lang-mr"
        className={`lang-btn ${language === 'mr' ? 'active' : ''}`}
        onClick={() => setLanguage('mr')}
        title="मराठीमध्ये बदला"
        aria-label="मराठीमध्ये बदला"
        aria-pressed={language === 'mr'}
      >
        <span className="lang-full">मराठी</span>
        <span className="lang-short">मराठी</span>
        <span className="lang-mini">मरा</span>
      </button>
    </div>
  );
}

