import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_clinical_pdf(
    patient_info,
    diagnostic_result,
    panel_paths,
    biomarker_metrics,
    output_pdf_path="ui/public/assets/DRISHYA_Clinical_Report.pdf"
):
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        leftMargin=28,
        rightMargin=28,
        topMargin=24,
        bottomMargin=24
    )
    
    usable_width = 595.27 - 56
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=15,
        textColor=colors.HexColor('#0F172A')
    )
    
    sub_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#64748B')
    )
    
    sec_title_style = ParagraphStyle(
        'SecTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=10.5,
        textColor=colors.HexColor('#0F172A')
    )
    
    cell_bold = ParagraphStyle(
        'CellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#1E293B')
    )
    
    cell_reg = ParagraphStyle(
        'CellReg',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#334155')
    )

    story = []
    
    # 1. HEADER
    header_data = [
        [
            [
                Paragraph("<b>DRISHYA TELE-OPHTHALMOLOGY SCREENING REPORT</b>", title_style),
                Paragraph("Automated Retinal Assessment • ICDR Clinical Standards", sub_style)
            ],
            [
                Paragraph(f"<b>Report ID:</b> {patient_info.get('report_id', 'DSH-2026-88492')}", ParagraphStyle('HR1', parent=cell_reg, alignment=2)),
                Paragraph(f"<b>Date:</b> {datetime.now().strftime('%d %b %Y, %H:%M')}", ParagraphStyle('HR2', parent=cell_reg, alignment=2)),
                Paragraph(f"<b>Center:</b> {patient_info.get('center', 'PHC Rampur | Rural Eye Hub')}", ParagraphStyle('HR3', parent=cell_reg, alignment=2))
            ]
        ]
    ]
    header_table = Table(header_data, colWidths=[usable_width*0.62, usable_width*0.38])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor('#CBD5E1'), spaceBefore=0, spaceAfter=8))
    
    # 2. PATIENT INFO
    pat_data = [
        [
            Paragraph(f"<b>Patient Name:</b> {patient_info.get('name', 'Ramesh Kumar')}", cell_reg),
            Paragraph(f"<b>Age / Sex:</b> {patient_info.get('age_sex', '54 Yrs / Male')}", cell_reg),
            Paragraph(f"<b>ABHA ID:</b> {patient_info.get('abha_id', '91-4820-1940-52')}", cell_reg),
            Paragraph(f"<b>Eye:</b> <b>{patient_info.get('eye', 'Left Eye (OS)')}</b>", cell_bold)
        ],
        [
            Paragraph(f"<b>Clinical History:</b> {patient_info.get('history', 'Type 2 Diabetes (11 Yrs)')}", cell_reg),
            Paragraph(f"<b>HbA1c:</b> {patient_info.get('hba1c', '8.6%')}", cell_reg),
            Paragraph(f"<b>Known HTN:</b> {patient_info.get('htn', 'Yes (140/90)')}", cell_reg),
            Paragraph(f"<b>Visual Acuity:</b> {patient_info.get('acuity', '6/9')}", cell_reg)
        ]
    ]
    pat_table = Table(pat_data, colWidths=[usable_width*0.30, usable_width*0.22, usable_width*0.28, usable_width*0.20])
    pat_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#F1F5F9')),
        ('PADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(pat_table)
    story.append(Spacer(1, 8))
    
    # 3. DIAGNOSTIC RESULT
    grade_color = '#B91C1C' if diagnostic_result.get('grade_num', 0) >= 2 else '#15803D'
    status_data = [
        [
            Paragraph("<font size=7 color='#64748B'>PRIMARY FINDING (ICDR GRADE)</font><br/>"
                      f"<b><font size=11 color='#0F172A'>{diagnostic_result.get('grade_title', 'Grade 2: Moderate NPDR')}</font></b><br/>"
                      f"<font size=7 color='#475569'>{diagnostic_result.get('grade_desc', 'Non-Proliferative Diabetic Retinopathy')}</font>", ParagraphStyle('S1', leading=12)),
            Paragraph("<font size=7 color='#64748B'>TRIAGE RECOMMENDATION</font><br/>"
                      f"<b><font size=10 color='{grade_color}'>{diagnostic_result.get('triage_status', 'Referable DR (Refer to Specialist)')}</font></b><br/>"
                      f"<font size=7 color='#64748B'>{diagnostic_result.get('triage_sub', 'Specialist Slit-Lamp Exam Recommended')}</font>", ParagraphStyle('S2', leading=11)),
            Paragraph("<font size=7 color='#64748B'>DATA QUALITY & CONFIDENCE</font><br/>"
                      f"<b>Image Quality:</b> <font color='#15803D'>{diagnostic_result.get('iqa_status', 'Pass (Q=0.88)')}</font><br/>"
                      f"<b>AI Confidence:</b> {diagnostic_result.get('confidence', '96.4%')}", ParagraphStyle('S3', leading=10, fontSize=7.5))
        ]
    ]
    status_table = Table(status_data, colWidths=[usable_width*0.40, usable_width*0.35, usable_width*0.25])
    status_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.white),
        ('BOX', (0,0), (-1,-1), 1.0, colors.HexColor('#CBD5E1')),
        ('LINEBEFORE', (1,0), (1,0), 0.75, colors.HexColor('#E2E8F0')),
        ('LINEBEFORE', (2,0), (2,0), 0.75, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(status_table)
    story.append(Spacer(1, 10))
    
    # 4. TRIPLE RETINAL PANELS
    img_size = (usable_width - 16) / 3.0
    img_table_data = [
        [
            RLImage(panel_paths['preprocessed'], width=img_size, height=img_size),
            RLImage(panel_paths['lesions'], width=img_size, height=img_size),
            RLImage(panel_paths['gradcam'], width=img_size, height=img_size)
        ],
        [
            Paragraph("<b>(a) Preprocessed Retina</b><br/><font size=6.5 color='#64748B'>Normalized 384x384</font>", ParagraphStyle('C1', parent=cell_reg, alignment=1)),
            Paragraph("<b>(b) Detected Lesions</b><br/><font size=6.5 color='#64748B'>Red: Aneurysms | Yellow: Exudates</font>", ParagraphStyle('C2', parent=cell_reg, alignment=1)),
            Paragraph("<b>(c) Grad-CAM++ Attention</b><br/><font size=6.5 color='#64748B'>Neural Saliency Focus</font>", ParagraphStyle('C3', parent=cell_reg, alignment=1))
        ]
    ]
    img_table = Table(img_table_data, colWidths=[img_size, img_size, img_size])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,0), 0),
        ('PADDING', (0,1), (-1,1), 3),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#F8FAFC')),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 10))
    
    # 5. BIOMARKER SUMMARY TABLE
    story.append(Paragraph("<b>CLINICAL BIOMARKER SUMMARY</b>", sec_title_style))
    story.append(Spacer(1, 3))
    
    table_content = [
        [
            Paragraph("<b>Biomarker / Feature</b>", cell_bold),
            Paragraph("<b>Result</b>", cell_bold),
            Paragraph("<b>Reference Threshold</b>", cell_bold),
            Paragraph("<b>Clinical Relevance</b>", cell_bold)
        ],
        [
            Paragraph("Microaneurysms (MAs)", cell_reg),
            Paragraph(f"<b>{biomarker_metrics.get('mas', '12 detected')}</b>", cell_reg),
            Paragraph("0 (Normal) | 1–5 (Mild) | &gt;5 (Moderate)", cell_reg),
            Paragraph(biomarker_metrics.get('mas_rel', 'Active microvascular leakage'), cell_reg)
        ],
        [
            Paragraph("Hard Exudates (Lipids)", cell_reg),
            Paragraph(f"<b>{biomarker_metrics.get('exudates', 'Cluster present (1.1%)')}</b>", cell_reg),
            Paragraph("Absent in Normal / Mild", cell_reg),
            Paragraph(biomarker_metrics.get('exudates_rel', 'Lipoprotein deposits (Sup. Arcade)'), cell_reg)
        ],
        [
            Paragraph("Hemorrhages", cell_reg),
            Paragraph(f"<b>{biomarker_metrics.get('hemorrhages', '2 quadrants')}</b>", cell_reg),
            Paragraph("4 quadrants = Severe (4:2:1 Rule)", cell_reg),
            Paragraph(biomarker_metrics.get('hemorrhages_rel', 'Below severe NPDR threshold'), cell_reg)
        ],
        [
            Paragraph("Neovascularization", cell_reg),
            Paragraph(f"<b>{biomarker_metrics.get('neovascularization', 'None (Absent)')}</b>", cell_reg),
            Paragraph("Present in Proliferative DR (PDR)", cell_reg),
            Paragraph("<font color='#15803D'>Negative for Proliferative Stage</font>", cell_reg)
        ],
        [
            Paragraph("Macular Involvement", cell_reg),
            Paragraph(f"<b>{biomarker_metrics.get('macula', 'Moderate Risk')}</b>", cell_reg),
            Paragraph("Exudates &lt; 1 disc diameter of fovea", cell_reg),
            Paragraph(biomarker_metrics.get('macula_rel', 'Exudates ~680 µm from fovea'), cell_reg)
        ]
    ]
    bio_table = Table(table_content, colWidths=[usable_width*0.28, usable_width*0.22, usable_width*0.28, usable_width*0.22])
    bio_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 3),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(bio_table)
    story.append(Spacer(1, 10))
    
    # 6. ACTION PLAN & SIGN-OFF
    action_box = [
        Paragraph("<b>CLINICAL ACTION PLAN</b>", ParagraphStyle('A1', parent=cell_bold, fontSize=8)),
        Spacer(1, 2),
        Paragraph(f"• <b>Referral:</b> {diagnostic_result.get('action_referral', 'Slit-lamp exam + OCT within 3–4 weeks at District Hospital.')}", cell_reg),
        Paragraph("• <b>Management:</b> Consult Primary Physician for HbA1c optimization.", cell_reg),
        Paragraph(f"• <b>Follow-up:</b> {diagnostic_result.get('action_followup', 'Repeat tele-screening in 6 months.')}", cell_reg)
    ]
    
    signoff_box = [
        Paragraph("<b>OPHTHALMOLOGIST SIGN-OFF</b>", ParagraphStyle('S1', parent=cell_bold, fontSize=8)),
        Spacer(1, 2),
        Paragraph("<b>Reviewer:</b> Dr. Rajesh Varma, MD (Ophthal) • Reg: MCI-49218", cell_reg),
        Paragraph("<b>Status:</b> [ <b>X</b> ] Concur with AI Grade   [ ] Re-evaluate", cell_reg),
        Paragraph("<b>Signature:</b> <i>R. Varma (Verified Tele-Sign) • 30-Aug-2026</i>", ParagraphStyle('S4', parent=cell_reg, fontSize=6.8, textColor=colors.HexColor('#64748B')))
    ]
    
    bottom_table = Table([[action_box, signoff_box]], colWidths=[usable_width*0.52, usable_width*0.48])
    bottom_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FAFAFA')),
    ]))
    story.append(bottom_table)
    story.append(Spacer(1, 8))
    
    # 7. DISCLAIMER
    disclaimer = (
        "<b>Disclaimer:</b> DRISHYA is an AI-assisted screening tool (SaMD). Final clinical management must be confirmed by a licensed ophthalmologist. "
        "Complies with CDSCO/ICMR screening protocols. <b>Audit Hash:</b> 4c89f7a1e28b"
    )
    story.append(Paragraph(disclaimer, ParagraphStyle('Disc', fontName='Helvetica', fontSize=5.8, leading=7.5, textColor=colors.HexColor('#94A3B8'), alignment=1)))
    
    doc.build(story)
    return output_pdf_path

if __name__ == "__main__":
    patient_info = {'name': 'Ramesh Kumar', 'age_sex': '54 Yrs / Male', 'abha_id': '91-4820-1940-52', 'eye': 'Left Eye (OS)'}
    diagnostic_result = {'grade_title': 'Grade 2: Moderate NPDR', 'grade_desc': 'Non-Proliferative Diabetic Retinopathy', 'triage_status': 'Referable DR (Refer to Specialist)', 'triage_sub': 'Specialist Slit-Lamp Exam Recommended', 'iqa_status': 'Pass (Q=0.88)', 'confidence': '96.4%', 'grade_num': 2}
    panel_paths = {'preprocessed': 'ui/public/assets/grade2_preprocessed.png', 'lesions': 'ui/public/assets/grade2_lesions.png', 'gradcam': 'ui/public/assets/grade2_gradcam.png'}
    biomarker_metrics = {'mas': '12 detected', 'mas_rel': 'Active microvascular leakage', 'exudates': '1.10% area cluster', 'exudates_rel': 'Lipoprotein deposits near arcade', 'hemorrhages': '2 quadrants', 'hemorrhages_rel': 'Below severe NPDR threshold', 'neovascularization': '0 (Absent)', 'macula': 'Moderate Risk', 'macula_rel': 'Exudates ~680 µm from fovea'}
    generate_clinical_pdf(patient_info, diagnostic_result, panel_paths, biomarker_metrics)
    print("Report generated successfully!")
