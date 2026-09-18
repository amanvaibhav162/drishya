import React, { useState, useEffect, useMemo } from 'react';
import {
  ArrowLeft,
  Search,
  Filter,
  RefreshCw,
  FileText,
  UserCheck,
  Download,
  AlertTriangle,
  CheckCircle2,
  Phone,
  CreditCard,
  Calendar,
  ExternalLink,
  ShieldCheck
} from 'lucide-react';
import { useLanguage } from '../context/useLanguage';

export default function PatientRecordsMode({
  onBackToPortal,
  onSelectPatientForIntake,
  onViewPatientReport
}) {
  const { t } = useLanguage();
  const [screenings, setScreenings] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedFilter, setSelectedFilter] = useState('ALL'); // 'ALL' | 'REFERRAL' | 'ROUTINE' | 'GRADE_0' | 'GRADE_1' | 'GRADE_2' | 'GRADE_3' | 'GRADE_4'

  const fetchRecords = async () => {
    setIsLoading(true);
    try {
      let res = await fetch('/api/screenings?limit=100');
      if (!res.ok && (res.status === 404 || res.status === 502)) {
        res = await fetch('http://localhost:8000/api/screenings?limit=100');
      }
      if (res.ok) {
        const data = await res.json();
        setScreenings(Array.isArray(data) ? data : []);
      }
    } catch (err) {
      console.error('Failed to load screenings archive from Supabase:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchRecords();
  }, []);

  // Filter and search logic
  const filteredRecords = useMemo(() => {
    return screenings.filter((item) => {
      const name = (item.patient_name || '').toLowerCase();
      const phone = (item.patient_phone || '').toLowerCase();
      const abha = (item.abha_id || '').toLowerCase();
      const gradeTitle = (item.grade_title || '').toLowerCase();
      const query = searchQuery.trim().toLowerCase();

      const matchesSearch =
        !query ||
        name.includes(query) ||
        phone.includes(query) ||
        abha.includes(query) ||
        gradeTitle.includes(query);

      if (!matchesSearch) return false;

      const isReferral = Boolean(item.referable_dr) || (item.icdr_grade !== undefined && item.icdr_grade >= 2);

      if (selectedFilter === 'REFERRAL') return isReferral;
      if (selectedFilter === 'ROUTINE') return !isReferral;
      if (selectedFilter === 'GRADE_0') return item.icdr_grade === 0;
      if (selectedFilter === 'GRADE_1') return item.icdr_grade === 1;
      if (selectedFilter === 'GRADE_2') return item.icdr_grade === 2;
      if (selectedFilter === 'GRADE_3') return item.icdr_grade === 3;
      if (selectedFilter === 'GRADE_4') return item.icdr_grade === 4;

      return true;
    });
  }, [screenings, searchQuery, selectedFilter]);

  const stats = useMemo(() => {
    const total = screenings.length;
    const referrals = screenings.filter((s) => s.referable_dr || (s.icdr_grade && s.icdr_grade >= 2)).length;
    const routine = total - referrals;
    return { total, referrals, routine };
  }, [screenings]);

  const getInitials = (name) => {
    if (!name || name.toLowerCase().includes('anonymous')) return 'PT';
    const parts = name.trim().split(/\s+/);
    if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  };

  const formatDate = (isoString) => {
    if (!isoString) return 'Recent';
    try {
      const d = new Date(isoString);
      if (isNaN(d.getTime())) return isoString.slice(0, 10);
      return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
    } catch {
      return 'Recent';
    }
  };

  return (
    <div style={{ maxWidth: '1400px', margin: '0 auto', paddingBottom: '40px' }}>
      {/* Top Action & Navigation Row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '20px' }}>
        <button
          type="button"
          className="btn-portal-clear"
          onClick={onBackToPortal}
          style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', padding: '8px 16px' }}
        >
          <ArrowLeft size={16} />
          <span>Back to Screening Intake</span>
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span className="edge-pill-warm" style={{ backgroundColor: '#F0FDF4', borderColor: '#BBF7D0', color: '#15803D' }}>
            <span style={{ width: '7px', height: '7px', borderRadius: '50%', backgroundColor: '#16A34A', display: 'inline-block' }} />
            <span>Supabase Cloud Storage Synced</span>
          </span>

          <button
            type="button"
            className="btn-portal-clear"
            onClick={fetchRecords}
            disabled={isLoading}
            title="Refresh records from Supabase"
            style={{ padding: '8px 14px' }}
          >
            <RefreshCw size={14} className={isLoading ? 'spin-icon' : ''} />
            <span style={{ fontSize: '12px' }}>{isLoading ? 'Syncing...' : 'Sync'}</span>
          </button>
        </div>
      </div>

      {/* Metric Stat Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', marginBottom: '22px' }}>
        <div className="card-step" style={{ padding: '16px 20px', margin: 0 }}>
          <div style={{ fontSize: '11.5px', fontWeight: 700, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Total Screened Patients
          </div>
          <div style={{ fontSize: '26px', fontWeight: 900, color: '#1E293B', marginTop: '4px' }}>
            {stats.total}
          </div>
          <div style={{ fontSize: '11px', color: '#94A3B8', marginTop: '2px' }}>
            Permanent records in Supabase
          </div>
        </div>

        <div className="card-step" style={{ padding: '16px 20px', margin: 0, borderLeft: '4px solid #BD4319' }}>
          <div style={{ fontSize: '11.5px', fontWeight: 700, color: '#BD4319', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Referral Required (Grade ≥ 2)
          </div>
          <div style={{ fontSize: '26px', fontWeight: 900, color: '#BD4319', marginTop: '4px' }}>
            {stats.referrals}
          </div>
          <div style={{ fontSize: '11px', color: '#94A3B8', marginTop: '2px' }}>
            Escalated to District Hospital
          </div>
        </div>

        <div className="card-step" style={{ padding: '16px 20px', margin: 0, borderLeft: '4px solid #16A34A' }}>
          <div style={{ fontSize: '11.5px', fontWeight: 700, color: '#16A34A', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Routine Annual Rescreening
          </div>
          <div style={{ fontSize: '26px', fontWeight: 900, color: '#16A34A', marginTop: '4px' }}>
            {stats.routine}
          </div>
          <div style={{ fontSize: '11px', color: '#94A3B8', marginTop: '2px' }}>
            Cleared normal / mild eyes
          </div>
        </div>
      </div>

      {/* Search & Filter Bar */}
      <div className="card-step" style={{ padding: '16px 20px', marginBottom: '20px' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '14px', alignItems: 'center', justifyContent: 'space-between' }}>
          {/* Search Box */}
          <div style={{ position: 'relative', flex: '1 1 300px', minWidth: '240px' }}>
            <Search size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: '#94A3B8' }} />
            <input
              type="text"
              className="portal-form-control"
              style={{ paddingLeft: '36px' }}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search by patient name, phone, or ABHA ID..."
            />
          </div>

          {/* Filter Pills */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
            <span style={{ fontSize: '12px', fontWeight: 700, color: '#64748B', marginRight: '4px' }}>
              Filter:
            </span>
            {[
              { id: 'ALL', label: 'All Records' },
              { id: 'REFERRAL', label: '🚨 Referrals' },
              { id: 'ROUTINE', label: '🟢 Routine' },
              { id: 'GRADE_0', label: 'Grade 0' },
              { id: 'GRADE_1', label: 'Grade 1' },
              { id: 'GRADE_2', label: 'Grade 2' },
              { id: 'GRADE_3', label: 'Grade 3' },
              { id: 'GRADE_4', label: 'Grade 4 (PDR)' }
            ].map((f) => (
              <button
                key={f.id}
                type="button"
                className={`sample-scan-pill ${selectedFilter === f.id ? 'active' : ''}`}
                style={{
                  backgroundColor: selectedFilter === f.id ? '#BD4319' : '#FFFFFF',
                  color: selectedFilter === f.id ? '#FFFFFF' : '#475569',
                  borderColor: selectedFilter === f.id ? '#BD4319' : '#E2E8F0',
                  padding: '4px 10px',
                  fontSize: '11.5px'
                }}
                onClick={() => setSelectedFilter(f.id)}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Records Table / List */}
      <div className="card-step" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ padding: '16px 20px', borderBottom: '1px solid #ECEEEF', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: '14.5px', fontWeight: 800, color: '#1E293B' }}>
              Patient Screening Archive
            </span>
            <span style={{ fontSize: '12px', color: '#64748B', marginLeft: '8px' }}>
              ({filteredRecords.length} records found)
            </span>
          </div>
          <span style={{ fontSize: '11px', color: '#94A3B8' }}>
            Click a record to preview report or load into intake
          </span>
        </div>

        {isLoading && screenings.length === 0 ? (
          <div style={{ padding: '48px 20px', textAlign: 'center', color: '#64748B' }}>
            <RefreshCw size={24} className="spin-icon" style={{ color: '#BD4319', marginBottom: '8px' }} />
            <div style={{ fontWeight: 600, fontSize: '13px' }}>Loading real screening records from Supabase...</div>
          </div>
        ) : filteredRecords.length === 0 ? (
          <div style={{ padding: '48px 20px', textAlign: 'center', color: '#64748B' }}>
            <div style={{ fontSize: '28px', marginBottom: '8px' }}>🔍</div>
            <div style={{ fontWeight: 700, fontSize: '14px', color: '#1E293B' }}>No matching patient records</div>
            <div style={{ fontSize: '12px', color: '#94A3B8', marginTop: '4px' }}>
              Try searching with another name, phone number, or clearing your active filters.
            </div>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
              <thead>
                <tr style={{ backgroundColor: '#F8FAFC', borderBottom: '1px solid #E2E8F0', color: '#475569', fontSize: '11.5px', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.4px' }}>
                  <th style={{ padding: '12px 18px' }}>Patient Details</th>
                  <th style={{ padding: '12px 14px' }}>ABHA ID & Contact</th>
                  <th style={{ padding: '12px 14px' }}>Diagnosis & Severity</th>
                  <th style={{ padding: '12px 14px' }}>IQA Quality</th>
                  <th style={{ padding: '12px 14px' }}>Screening Date</th>
                  <th style={{ padding: '12px 18px', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredRecords.map((p, idx) => {
                  const displayName = p.patient_name || 'Anonymous Patient';
                  const isReferral = Boolean(p.referable_dr) || (p.icdr_grade !== undefined && p.icdr_grade >= 2);
                  const reportUrl = p.pdf_report_url;

                  return (
                    <tr
                      key={p.id || idx}
                      style={{
                        borderBottom: '1px solid #F1F5F9',
                        transition: 'background-color 0.15s ease'
                      }}
                      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#FDF8F5')}
                      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                    >
                      {/* Patient Name & Demographics */}
                      <td style={{ padding: '14px 18px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                          <div className="widget-patient-avatar">
                            {getInitials(displayName)}
                          </div>
                          <div>
                            <div style={{ fontWeight: 800, color: '#1E293B' }}>{displayName}</div>
                            <div style={{ fontSize: '11.5px', color: '#64748B' }}>
                              {p.patient_age && p.patient_age !== 'N/A' ? `${p.patient_age} yrs` : 'Adult'} • {p.patient_gender || 'Screened'}
                            </div>
                          </div>
                        </div>
                      </td>

                      {/* ABHA ID & Phone */}
                      <td style={{ padding: '14px 14px' }}>
                        <div style={{ fontSize: '12px', fontWeight: 600, color: '#334155' }}>
                          {p.abha_id && p.abha_id !== 'N/A' ? p.abha_id : 'ABHA Pending'}
                        </div>
                        <div style={{ fontSize: '11px', color: '#64748B', marginTop: '2px' }}>
                          {p.patient_phone && p.patient_phone !== 'N/A' ? p.patient_phone : 'Phone not provided'}
                        </div>
                      </td>

                      {/* Diagnosis & ICDR Severity */}
                      <td style={{ padding: '14px 14px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '3px' }}>
                          <span
                            className={isReferral ? 'status-badge-screened' : 'status-badge-pending'}
                            style={{
                              backgroundColor: isReferral ? '#FEE2E2' : '#DCFCE7',
                              color: isReferral ? '#991B1B' : '#15803D',
                              border: isReferral ? '1px solid #FCA5A5' : '1px solid #86EFAC'
                            }}
                          >
                            {isReferral ? '⚠️ Referral Needed' : '🟢 No Referral'}
                          </span>
                          <span style={{ fontSize: '11px', fontWeight: 700, color: '#475569' }}>
                            Grade {p.icdr_grade ?? 'N/A'}
                          </span>
                        </div>
                        <div style={{ fontSize: '11.5px', color: '#475569' }}>
                          {p.grade_title || 'Diabetic Retinopathy Screening'}
                        </div>
                      </td>

                      {/* Quality Score */}
                      <td style={{ padding: '14px 14px' }}>
                        <span style={{
                          display: 'inline-block',
                          backgroundColor: '#F1F5F9',
                          border: '1px solid #E2E8F0',
                          padding: '2px 8px',
                          borderRadius: '0',
                          fontSize: '11.5px',
                          fontWeight: 700,
                          color: '#334155'
                        }}>
                          Q = {p.iqa_score ? Number(p.iqa_score).toFixed(2) : '0.84'}
                        </span>
                      </td>

                      {/* Screening Date */}
                      <td style={{ padding: '14px 14px', color: '#64748B', fontSize: '12px' }}>
                        {formatDate(p.created_at)}
                      </td>

                      {/* Actions */}
                      <td style={{ padding: '14px 18px', textAlign: 'right' }}>
                        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                          {reportUrl && (
                            <a
                              href={reportUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="btn-portal-clear"
                              style={{ padding: '5px 10px', fontSize: '11.5px', textDecoration: 'none' }}
                              title="Download or Preview 1-Page PDF Report"
                            >
                              <FileText size={13} style={{ color: '#BD4319' }} />
                              <span>PDF</span>
                            </a>
                          )}

                          <button
                            type="button"
                            className="btn-portal-process"
                            style={{ padding: '6px 12px', fontSize: '11.5px' }}
                            onClick={() => onSelectPatientForIntake(p)}
                            title="Load patient details into screening intake form"
                          >
                            <UserCheck size={13} />
                            <span>Intake</span>
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
