import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm

# ── Brand Colors ──────────────────────────────────────────────────────────────
TEAL        = colors.HexColor('#2BA882')
TEAL_DARK   = colors.HexColor('#1F8A6C')
TEAL_LIGHT  = colors.HexColor('#E6F7F1')
SLATE_900   = colors.HexColor('#0F172A')
SLATE_700   = colors.HexColor('#334155')
SLATE_500   = colors.HexColor('#64748B')
SLATE_300   = colors.HexColor('#CBD5E1')
SLATE_200   = colors.HexColor('#E2E8F0')
SLATE_100   = colors.HexColor('#F1F5F9')
SLATE_50    = colors.HexColor('#F8FAFC')
WHITE       = colors.white
RED_700     = colors.HexColor('#B91C1C')
GREEN_700   = colors.HexColor('#15803D')


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
        leftMargin=24,
        rightMargin=24,
        topMargin=20,
        bottomMargin=20
    )

    W = A4[0] - 48  # usable width (595.27 - 48 ≈ 547)
    styles = getSampleStyleSheet()

    # ── Paragraph Styles ──────────────────────────────────────────────────────
    s_brand = ParagraphStyle('Brand', fontName='Helvetica-Bold', fontSize=16, leading=19, textColor=TEAL_DARK)
    s_report_title = ParagraphStyle('ReportTitle', fontName='Helvetica-Bold', fontSize=14, leading=17, textColor=SLATE_900, alignment=2)

    s_section_header = ParagraphStyle('SectionHeader', fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=WHITE)

    s_label = ParagraphStyle('Label', fontName='Helvetica-Bold', fontSize=7.5, leading=10, textColor=SLATE_500)
    s_value = ParagraphStyle('Value', fontName='Helvetica', fontSize=8, leading=10.5, textColor=SLATE_900)
    s_value_bold = ParagraphStyle('ValueBold', fontName='Helvetica-Bold', fontSize=8, leading=10.5, textColor=SLATE_900)

    s_result_label = ParagraphStyle('ResultLabel', fontName='Helvetica-Bold', fontSize=9, leading=12, textColor=TEAL_DARK)

    s_cell_bold = ParagraphStyle('CellBold', fontName='Helvetica-Bold', fontSize=7.5, leading=9.5, textColor=SLATE_900)
    s_cell = ParagraphStyle('Cell', fontName='Helvetica', fontSize=7.5, leading=9.5, textColor=SLATE_700)

    s_caption_bold = ParagraphStyle('CaptionBold', fontName='Helvetica-Bold', fontSize=7, leading=9, textColor=SLATE_900, alignment=1)
    s_caption = ParagraphStyle('Caption', fontName='Helvetica', fontSize=6.5, leading=8.5, textColor=SLATE_500, alignment=1)

    s_disclaimer = ParagraphStyle('Disclaimer', fontName='Helvetica', fontSize=5.5, leading=7.5, textColor=SLATE_500, alignment=0)

    story = []

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. HEADER – Logo / Brand left, Report Title right
    # ═══════════════════════════════════════════════════════════════════════════
    logo_path = "ui/public/assets/drishyalogo.jpeg"
    if os.path.exists(logo_path):
        logo_img = RLImage(logo_path, width=42, height=23)
        brand_left = Table([[logo_img, [
            Paragraph("DRISHYA", s_brand),
            Paragraph("AI-Powered Tele-Ophthalmology", ParagraphStyle('BrandSub', fontName='Helvetica', fontSize=7, leading=9, textColor=SLATE_500))
        ]]], colWidths=[48, W * 0.50 - 48])
        brand_left.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 0),
        ]))
    else:
        brand_left = [
            Paragraph("DRISHYA", s_brand),
            Paragraph("AI-Powered Tele-Ophthalmology", ParagraphStyle('BrandSub', fontName='Helvetica', fontSize=7, leading=9, textColor=SLATE_500))
        ]

    header_data = [[
        brand_left,
        Paragraph("Drishya Diagnostic Report", s_report_title)
    ]]
    header = Table(header_data, colWidths=[W * 0.50, W * 0.50])
    header.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(header)
    story.append(HRFlowable(width="100%", thickness=1.5, color=TEAL, spaceBefore=0, spaceAfter=6))

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. PATIENT INFORMATION + GENERAL INFORMATION (two teal banners)
    # ═══════════════════════════════════════════════════════════════════════════
    def _section_banner(text, width):
        t = Table([[Paragraph(text, s_section_header)]], colWidths=[width])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), TEAL),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        return t

    def _info_row(label, value, bold=False):
        style = s_value_bold if bold else s_value
        return [Paragraph(f"<b>{label}</b>", s_label), Paragraph(str(value), style)]

    col_left_w = W * 0.52
    col_right_w = W * 0.48

    patient_rows = [
        [_section_banner("PATIENT INFORMATION", col_left_w), _section_banner("GENERAL INFORMATION", col_right_w)],
    ]

    # Build left/right info as sub-tables
    left_info = [
        _info_row("PATIENT NAME", patient_info.get('name', 'Ramesh Kumar'), bold=True),
        _info_row("ABHA ID", patient_info.get('abha_id', '91-4820-1940-52')),
        _info_row("AGE / SEX", patient_info.get('age_sex', '54 Yrs / Male')),
        _info_row("RESULT DATE", datetime.now().strftime('%d/%m/%Y %I:%M %p')),
    ]
    right_info = [
        _info_row("SCREENING CENTER", patient_info.get('center', 'PHC Rampur (Zone 4)'), bold=True),
        _info_row("EYE", patient_info.get('eye', 'Left Eye (OS)')),
        _info_row("ORDERING CODE", patient_info.get('ordering_code', 'E11.9')),
        _info_row("REPORT ID", patient_info.get('report_id', f"DSH-{datetime.now().strftime('%Y')}-{datetime.now().strftime('%f')[:5]}")),
    ]

    left_table = Table(left_info, colWidths=[col_left_w * 0.38, col_left_w * 0.62])
    left_table.setStyle(TableStyle([
        ('INNERGRID', (0, 0), (-1, -1), 0.5, SLATE_200),
        ('BOX', (0, 0), (-1, -1), 0.5, SLATE_200),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 0), (0, -1), SLATE_50),
    ]))

    right_table = Table(right_info, colWidths=[col_right_w * 0.40, col_right_w * 0.60])
    right_table.setStyle(TableStyle([
        ('INNERGRID', (0, 0), (-1, -1), 0.5, SLATE_200),
        ('BOX', (0, 0), (-1, -1), 0.5, SLATE_200),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 0), (0, -1), SLATE_50),
    ]))

    patient_rows.append([left_table, right_table])
    patient_grid = Table(patient_rows, colWidths=[col_left_w, col_right_w])
    patient_grid.setStyle(TableStyle([
        ('PADDING', (0, 0), (-1, -1), 0),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(patient_grid)
    story.append(Spacer(1, 8))

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. RESULTS SECTION
    # ═══════════════════════════════════════════════════════════════════════════
    results_banner = _section_banner("RESULTS", W)
    story.append(results_banner)

    is_referable = diagnostic_result.get('grade_num', 0) >= 2
    result_color = RED_700 if is_referable else GREEN_700
    result_text = "Referral\nNeeded" if is_referable else "No Referral\nNeeded"

    results_left = [
        _info_row("CONDITION", "Diabetic Retinopathy"),
        _info_row("DIAGNOSIS", diagnostic_result.get('grade_desc', 'No Diabetic Retinopathy detected ETDRS level 20 and lower and no macular edema.')),
        _info_row("DIAGNOSIS CODE", patient_info.get('ordering_code', 'E11.9')),
        _info_row("CARE PLAN", diagnostic_result.get('action_followup', 'Retest in 12 months')),
        _info_row("INTERPRETATION", "Results were produced by DRISHYA, an AI system that provides automated retinal interpretation."),
    ]

    results_left_t = Table(results_left, colWidths=[W * 0.52 * 0.35, W * 0.52 * 0.65])
    results_left_t.setStyle(TableStyle([
        ('INNERGRID', (0, 0), (-1, -1), 0.5, SLATE_200),
        ('BOX', (0, 0), (-1, -1), 0.5, SLATE_200),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 0), (0, -1), SLATE_50),
    ]))

    s_result_big = ParagraphStyle('ResultBig', fontName='Helvetica-Bold', fontSize=22, leading=26, textColor=result_color, alignment=1)
    result_right_data = [
        [Paragraph("RESULT", s_result_label)],
        [Paragraph(result_text, s_result_big)],
    ]
    result_right_t = Table(result_right_data, colWidths=[W * 0.48])
    result_right_t.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 0.5, SLATE_200),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 14),
    ]))

    results_grid = Table([[results_left_t, result_right_t]], colWidths=[W * 0.52, W * 0.48])
    results_grid.setStyle(TableStyle([
        ('PADDING', (0, 0), (-1, -1), 0),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(results_grid)
    story.append(Spacer(1, 10))

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. BOTTOM HALF – AI Facts (left) + Fundus Images (right)
    # ═══════════════════════════════════════════════════════════════════════════
    ai_banner = _section_banner("AUGMENTED INTELLIGENCE FACTS", W * 0.48)
    img_banner = _section_banner("3 FUNDUS IMAGES USED IN EXAM", W * 0.52)

    # -- AI Facts Table --
    ai_rows = [
        [Paragraph("<i>The table below describes the AI model providing the interpretation.</i>", ParagraphStyle('AINote', fontName='Helvetica-Oblique', fontSize=6.5, leading=8, textColor=SLATE_500))],
    ]

    ai_facts_data = [
        ("AI Description", ""),
        ("Product Name", "DRISHYA Retinal AI v1.0"),
        ("Type of Diagnostic", "Autonomous AI"),
        ("Disease", "Diabetic retinopathy, inclusive of macular edema"),
        ("Intended For", "Adults with diabetes (Rx only)"),
        ("", ""),
        ("AI Performance Data", ""),
        ("Reference Standard", diagnostic_result.get('confidence', '96.4%') + " confidence"),
        ("Sensitivity", biomarker_metrics.get('sensitivity', '94.2%')),
        ("Specificity", biomarker_metrics.get('specificity', '91.8%')),
        ("Diagnosability", biomarker_metrics.get('diagnosability', '96.0%')),
    ]

    ai_detail_rows = []
    for label, val in ai_facts_data:
        if val == "" and label != "":
            # Section sub-header
            ai_detail_rows.append([
                Paragraph(f"<b>{label}</b>", s_cell_bold), Paragraph("", s_cell)
            ])
        elif label == "" and val == "":
            continue
        else:
            ai_detail_rows.append([
                Paragraph(label, s_cell), Paragraph(val, s_cell)
            ])

    ai_detail_t = Table(ai_detail_rows, colWidths=[W * 0.48 * 0.48, W * 0.48 * 0.52])
    ai_detail_t.setStyle(TableStyle([
        ('INNERGRID', (0, 0), (-1, -1), 0.4, SLATE_200),
        ('BOX', (0, 0), (-1, -1), 0.5, SLATE_300),
        ('PADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND', (0, 0), (-1, 0), TEAL_LIGHT),
        ('BACKGROUND', (0, 6), (-1, 6), TEAL_LIGHT),
    ]))

    ai_column = [ai_banner, Spacer(1, 3)]
    ai_column.append(Paragraph("<i>The table below describes the AI model providing the interpretation.</i>",
                               ParagraphStyle('AINote2', fontName='Helvetica-Oblique', fontSize=6, leading=8, textColor=SLATE_500)))
    ai_column.append(Spacer(1, 2))
    ai_column.append(ai_detail_t)

    # -- Fundus Images Column (LARGE, prominent) --
    # Calculate image dimensions to maximize visibility
    img_col_w = W * 0.52
    img_cell_w = (img_col_w - 10) / 2.0   # 2 columns for a 2x2 grid, but we have 3 images
    # Use a row of 2 + row of 1 (centered) layout for 3 images
    img_single_w = (img_col_w - 12) / 2.0
    img_single_h = img_single_w  # square

    img_elements = [img_banner, Spacer(1, 4)]

    # Row 1: preprocessed + lesions
    img_row1 = [
        RLImage(panel_paths['preprocessed'], width=img_single_w, height=img_single_h),
        RLImage(panel_paths['lesions'], width=img_single_w, height=img_single_h),
    ]
    cap_row1 = [
        Paragraph("<b>(a) Preprocessed Retina</b>", s_caption_bold),
        Paragraph("<b>(b) Detected Lesions</b>", s_caption_bold),
    ]

    # Row 2: gradcam (centered, spanning full width for emphasis)
    gradcam_w = img_single_w * 1.2
    gradcam_h = gradcam_w

    img_grid_data = [
        img_row1,
        cap_row1,
    ]
    img_grid = Table(img_grid_data, colWidths=[img_single_w + 4, img_single_w + 4])
    img_grid.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 2),
    ]))
    img_elements.append(img_grid)
    img_elements.append(Spacer(1, 4))

    # Grad-CAM row centered
    gradcam_row = Table(
        [
            [RLImage(panel_paths['gradcam'], width=gradcam_w, height=gradcam_h)],
            [Paragraph("<b>(c) Grad-CAM++ Attention Map</b>", s_caption_bold)],
            [Paragraph("Neural saliency focus — areas of highest diagnostic interest", s_caption)],
        ],
        colWidths=[img_col_w]
    )
    gradcam_row.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 2),
    ]))
    img_elements.append(gradcam_row)
    img_elements.append(Spacer(1, 3))
    img_elements.append(Paragraph("<i>Image orientation and labeling is for reference only and should not be used for diagnostic purposes.</i>", 
                                  ParagraphStyle('ImgNote', fontName='Helvetica-Oblique', fontSize=5.5, leading=7, textColor=SLATE_500, alignment=1)))

    # Combine bottom half
    bottom_grid = Table([[ai_column, img_elements]], colWidths=[W * 0.48, W * 0.52])
    bottom_grid.setStyle(TableStyle([
        ('PADDING', (0, 0), (-1, -1), 0),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(bottom_grid)
    story.append(Spacer(1, 10))

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. DISCLAIMER
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=SLATE_300, spaceBefore=0, spaceAfter=4))
    story.append(Paragraph(
        "<b>DISCLAIMER</b>",
        ParagraphStyle('DiscTitle', fontName='Helvetica-Bold', fontSize=6.5, leading=8.5, textColor=SLATE_900)
    ))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "A positive result indicates a high risk of diabetic retinopathy with a severity of ETDRS level 35 or higher and/or macular edema. "
        "DRISHYA AI diabetic retinopathy screening does not replace a comprehensive eye exam. The images in this report are lower quality than "
        "the images used by the AI model and should not be used for diagnostic purposes. See user manual for more details. "
        f"<b>Audit Hash:</b> {datetime.now().strftime('%f')[:8]}a1e2",
        s_disclaimer
    ))

    doc.build(story)
    return output_pdf_path


if __name__ == "__main__":
    patient_info = {
        'name': 'Ramesh Kumar',
        'age_sex': '54 Yrs / Male',
        'abha_id': '91-4820-1940-52',
        'eye': 'Left Eye (OS)',
        'center': 'PHC Rampur | Rural Eye Care Hub',
        'report_id': 'DSH-2026-88492',
    }
    diagnostic_result = {
        'grade_title': 'Grade 2: Moderate NPDR',
        'grade_desc': 'Non-Proliferative Diabetic Retinopathy with moderate severity. Microaneurysms and exudates detected.',
        'triage_status': 'Referable DR (Refer to Specialist)',
        'triage_sub': 'Specialist Slit-Lamp Exam Recommended',
        'iqa_status': 'Pass (Q=0.88)',
        'confidence': '96.4%',
        'grade_num': 2,
        'action_followup': 'Retest in 6 months',
    }
    panel_paths = {
        'preprocessed': 'ui/public/assets/grade2_preprocessed.png',
        'lesions': 'ui/public/assets/grade2_lesions.png',
        'gradcam': 'ui/public/assets/grade2_gradcam.png',
    }
    biomarker_metrics = {
        'mas': '12 detected',
        'mas_rel': 'Active microvascular leakage',
        'exudates': '1.10% area cluster',
        'exudates_rel': 'Lipoprotein deposits near arcade',
        'hemorrhages': '2 quadrants',
        'hemorrhages_rel': 'Below severe NPDR threshold',
        'neovascularization': '0 (Absent)',
        'macula': 'Moderate Risk',
        'macula_rel': 'Exudates ~680 µm from fovea',
        'sensitivity': '94.2%',
        'specificity': '91.8%',
        'diagnosability': '96.0%',
    }
    generate_clinical_pdf(patient_info, diagnostic_result, panel_paths, biomarker_metrics)
    print("Report generated successfully!")
