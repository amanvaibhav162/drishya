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
        aria-pressed={language === 'en'}
      >
        <span>English</span>
      </button>
      <span className="lang-divider" aria-hidden="true" />
      <button
        type="button"
        id="btn-lang-hi"
        className={`lang-btn ${language === 'hi' ? 'active' : ''}`}
        onClick={() => setLanguage('hi')}
        title="हिन्दी में बदलें"
        aria-pressed={language === 'hi'}
      >
        <span>हिन्दी</span>
      </button>
      <span className="lang-divider" aria-hidden="true" />
      <button
        type="button"
        id="btn-lang-mr"
        className={`lang-btn ${language === 'mr' ? 'active' : ''}`}
        onClick={() => setLanguage('mr')}
        title="मराठीमध्ये बदला"
        aria-pressed={language === 'mr'}
      >
        <span>मराठी</span>
      </button>
    </div>
  );
}
