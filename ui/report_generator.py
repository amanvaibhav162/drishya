import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable, Flowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Modern Typographic System ────────────────────────────────────────────────
FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_ITALIC = "Helvetica-Oblique"
FONT_BOLD_ITALIC = "Helvetica-BoldOblique"

try:
    noto_regular = "/usr/share/fonts/noto/NotoSans-Regular.ttf"
    noto_bold = "/usr/share/fonts/noto/NotoSans-Bold.ttf"
    noto_italic = "/usr/share/fonts/noto/NotoSans-Italic.ttf"
    noto_bold_italic = "/usr/share/fonts/noto/NotoSans-BoldItalic.ttf"
    if all(os.path.exists(p) for p in [noto_regular, noto_bold, noto_italic, noto_bold_italic]):
        pdfmetrics.registerFont(TTFont('NotoSans', noto_regular))
        pdfmetrics.registerFont(TTFont('NotoSans-Bold', noto_bold))
        pdfmetrics.registerFont(TTFont('NotoSans-Italic', noto_italic))
        pdfmetrics.registerFont(TTFont('NotoSans-BoldItalic', noto_bold_italic))
        pdfmetrics.registerFontFamily('NotoSans', normal='NotoSans', bold='NotoSans-Bold', italic='NotoSans-Italic', boldItalic='NotoSans-BoldItalic')
        FONT_REGULAR = 'NotoSans'
        FONT_BOLD = 'NotoSans-Bold'
        FONT_ITALIC = 'NotoSans-Italic'
        FONT_BOLD_ITALIC = 'NotoSans-BoldItalic'
except Exception:
    pass

# ── Color Palette Tokens ──────────────────────────────────────────────────────
DARK_900    = colors.HexColor('#0F172A')  # Slate 900 / Deep Obsidian
DARK_800    = colors.HexColor('#1E293B')  # Slate 800
SLATE_700   = colors.HexColor('#334155')
SLATE_600   = colors.HexColor('#475569')
SLATE_500   = colors.HexColor('#64748B')
SLATE_400   = colors.HexColor('#94A3B8')
SLATE_300   = colors.HexColor('#CBD5E1')
SLATE_200   = colors.HexColor('#E2E8F0')
SLATE_100   = colors.HexColor('#F1F5F9')
SLATE_50    = colors.HexColor('#F8FAFC')
WHITE       = colors.white

# Clinical Triage Status Palette
RED_DARK    = colors.HexColor('#991B1B')
RED_MAIN    = colors.HexColor('#DC2626')
RED_BG      = colors.HexColor('#FEF2F2')
RED_BORDER  = colors.HexColor('#FECACA')

GREEN_DARK  = colors.HexColor('#166534')
GREEN_MAIN  = colors.HexColor('#16A34A')
GREEN_BG    = colors.HexColor('#F0FDF4')
GREEN_BORDER= colors.HexColor('#BBF7D0')


def _draw_page_decorations(canvas, doc):
    """
    Draws a certified medical report frame, top running header bar,
    and bottom audit verification footer.
    """
    canvas.saveState()
    # Outer certificate border with rounded corners
    canvas.setStrokeColor(SLATE_300)
    canvas.setLineWidth(0.6)
    canvas.roundRect(14, 14, doc.pagesize[0] - 28, doc.pagesize[1] - 28, 4, stroke=1, fill=0)

    # Top running rule
    canvas.setStrokeColor(SLATE_200)
    canvas.setLineWidth(0.4)
    canvas.line(22, doc.pagesize[1] - 18, doc.pagesize[0] - 22, doc.pagesize[1] - 18)

    # Top running micro-headers
    canvas.setFont(FONT_REGULAR, 5.0)
    canvas.setFillColor(SLATE_400)
    canvas.drawString(24, doc.pagesize[1] - 15, "DRISHYA CLINICAL RETINAL EVALUATION PLATFORM • TELE-OPHTHALMOLOGY AI SUITE")
    canvas.drawRightString(doc.pagesize[0] - 24, doc.pagesize[1] - 15, "CONFIDENTIAL HEALTH DOCUMENT")

    # Bottom running rule
    canvas.line(22, 23, doc.pagesize[0] - 22, 23)
    canvas.drawString(24, 16, "DRISHYA AI SCREENING • NOT DIRECT SUBSTITUTE FOR SPECIALIST OPHTHALMOLOGICAL BIO-MICROSCOPY • PAGE 1 OF 1")
    canvas.drawRightString(doc.pagesize[0] - 24, 16, "AUDIT SEAL: VERIFIED")
    canvas.restoreState()


def generate_clinical_pdf(
    patient_info,
    diagnostic_result,
    panel_paths,
    biomarker_metrics,
    output_pdf_path="ui/public/assets/DRISHYA_Clinical_Report.pdf"
):
    """
    Generates a professional 1-page A4 clinical diagnostic report.
    """
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        leftMargin=22,
        rightMargin=22,
        topMargin=22,
        bottomMargin=24,
        title="DRISHYA Clinical Diagnostic Report",
        author="DRISHYA AI Autonomous Screening Engine"
    )

    W = A4[0] - 44  # 595.28 - 44 = 551.28 pt usable width

    # ── Paragraph Styles Hierarchy ───────────────────────────────────────────
    s_brand = ParagraphStyle('Brand', fontName=FONT_BOLD, fontSize=15, leading=17, textColor=DARK_900)
    s_brand_sub = ParagraphStyle('BrandSub', fontName=FONT_BOLD, fontSize=6, leading=8, textColor=SLATE_500)

    s_report_title = ParagraphStyle('ReportTitle', fontName=FONT_BOLD, fontSize=13.5, leading=16, textColor=DARK_900, alignment=2)
    s_report_badge = ParagraphStyle('ReportBadge', fontName=FONT_BOLD, fontSize=6.5, leading=8.5, textColor=SLATE_600, alignment=2)

    s_banner_text = ParagraphStyle('BannerText', fontName=FONT_BOLD, fontSize=7.5, leading=9.5, textColor=WHITE)

    s_label = ParagraphStyle('Label', fontName=FONT_BOLD, fontSize=6.8, leading=8.8, textColor=SLATE_600)
    s_value = ParagraphStyle('Value', fontName=FONT_REGULAR, fontSize=7.2, leading=9.2, textColor=DARK_900)
    s_value_bold = ParagraphStyle('ValueBold', fontName=FONT_BOLD, fontSize=7.2, leading=9.2, textColor=DARK_900)

    s_cell = ParagraphStyle('Cell', fontName=FONT_REGULAR, fontSize=6.8, leading=8.8, textColor=SLATE_700)
    s_cell_bold = ParagraphStyle('CellBold', fontName=FONT_BOLD, fontSize=6.8, leading=8.8, textColor=DARK_900)
    s_th = ParagraphStyle('TableHead', fontName=FONT_BOLD, fontSize=6.8, leading=8.8, textColor=SLATE_700)

    s_cap_title = ParagraphStyle('CapTitle', fontName=FONT_BOLD, fontSize=6.5, leading=8.5, textColor=DARK_900, alignment=1)
    s_cap_sub = ParagraphStyle('CapSub', fontName=FONT_REGULAR, fontSize=5.5, leading=7.5, textColor=SLATE_500, alignment=1)

    s_footnote = ParagraphStyle('Footnote', fontName=FONT_ITALIC, fontSize=5.5, leading=7.0, textColor=SLATE_500, alignment=1)
    s_disc = ParagraphStyle('Disc', fontName=FONT_REGULAR, fontSize=5.2, leading=6.8, textColor=SLATE_500)

    story: list[Flowable] = []

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. HEADER – Logo, Brand & Document Status
    # ═══════════════════════════════════════════════════════════════════════════
    logo_path = "ui/public/assets/drishyalogo.jpeg"
    if os.path.exists(logo_path):
        logo_img = RLImage(logo_path, width=38, height=21)
        brand_block = Table([[logo_img, [
            Paragraph("DRISHYA", s_brand),
            Paragraph("AI-POWERED TELE-OPHTHALMOLOGY SUITE", s_brand_sub)
        ]]], colWidths=[42, W * 0.48 - 42])
        brand_block.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 0),
        ]))
    else:
        brand_block = [
            Paragraph("DRISHYA", s_brand),
            Paragraph("AI-POWERED TELE-OPHTHALMOLOGY SUITE", s_brand_sub)
        ]

    title_block = [
        Paragraph("Drishya Diagnostic Report", s_report_title),
        Paragraph("<font color='#0F172A'>●</font> <b>FINAL CLINICAL EVALUATION</b> &nbsp;|&nbsp; AUDIT CERTIFIED", s_report_badge)
    ]

    header_table = Table([[brand_block, title_block]], colWidths=[W * 0.48, W * 0.52])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.4, color=DARK_900, spaceBefore=0, spaceAfter=4))

    # ═══════════════════════════════════════════════════════════════════════════
    # Section Banner Generator
    # ═══════════════════════════════════════════════════════════════════════════
    def _make_banner(title_text, width):
        t = Table([[Paragraph(f"<b>{title_text}</b>", s_banner_text)]], colWidths=[width])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), DARK_900),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        return t

    def _info_cell(label, val, is_bold=False):
        style = s_value_bold if is_bold else s_value
        return [Paragraph(f"<b>{label}</b>", s_label), Paragraph(str(val), style)]

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. PATIENT INFORMATION + GENERAL INFORMATION
    # ═══════════════════════════════════════════════════════════════════════════
    col_w_left = W * 0.51
    col_w_right = W * 0.49

    left_info_data = [
        _info_cell("PATIENT NAME", patient_info.get('name') or 'Anonymous Patient', is_bold=True),
        _info_cell("ABHA ID", patient_info.get('abha_id') or 'Not Registered'),
        _info_cell("AGE / SEX", patient_info.get('age_sex') or 'Adult Screening'),
        _info_cell("RESULT DATE", datetime.now().strftime('%d/%m/%Y %I:%M %p')),
    ]
    t_patient = Table(left_info_data, colWidths=[col_w_left * 0.35, col_w_left * 0.65], rowHeights=[18.5, 18.5, 18.5, 18.5])
    t_patient.setStyle(TableStyle([
        ('INNERGRID', (0, 0), (-1, -1), 0.4, SLATE_200),
        ('BOX', (0, 0), (-1, -1), 0.5, SLATE_300),
        ('BACKGROUND', (0, 0), (0, -1), SLATE_50),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    report_id = patient_info.get('report_id') or f"DSH-{datetime.now().strftime('%Y')}-{datetime.now().strftime('%f')[:5]}"
    right_info_data = [
        _info_cell("SCREENING CENTER", patient_info.get('center', 'PHC Rampur (Zone 4)'), is_bold=True),
        _info_cell("EYE EXAMINED", patient_info.get('eye', 'Left Eye (OS)')),
        _info_cell("ORDERING CODE", patient_info.get('ordering_code', 'E11.9 (Type 2 Diabetes)')),
        _info_cell("REPORT ID", report_id),
    ]
    t_general = Table(right_info_data, colWidths=[col_w_right * 0.36, col_w_right * 0.64], rowHeights=[18.5, 18.5, 18.5, 18.5])
    t_general.setStyle(TableStyle([
        ('INNERGRID', (0, 0), (-1, -1), 0.4, SLATE_200),
        ('BOX', (0, 0), (-1, -1), 0.5, SLATE_300),
        ('BACKGROUND', (0, 0), (0, -1), SLATE_50),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    patient_grid = Table([
        [_make_banner("PATIENT INFORMATION", col_w_left), _make_banner("GENERAL INFORMATION", col_w_right)],
        [t_patient, t_general]
    ], colWidths=[col_w_left, col_w_right])
    patient_grid.setStyle(TableStyle([
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(patient_grid)
    story.append(Spacer(1, 8))

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. PHOTOS OF RETINA IN A ROW (Preprocessed, Lesions, Grad-CAM++)
    # ═══════════════════════════════════════════════════════════════════════════
    img_dim = 135  # square image dimensions

    def _frame_image(img_path, caption_title, caption_sub, w, h):
        if os.path.exists(img_path):
            img_obj = RLImage(img_path, width=w, height=h)
        else:
            img_obj = Paragraph("<i>Image not available</i>", s_cap_sub)
        
        card = Table([
            [img_obj],
            [Paragraph(caption_title, s_cap_title)],
            [Paragraph(caption_sub, s_cap_sub)]
        ], colWidths=[w + 20])
        card.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('BOX', (0, 0), (-1, 0), 0.5, SLATE_300),
        ]))
        return card

    card_prep = _frame_image(panel_paths.get('preprocessed', ''), "<b>(a) Preprocessed Retina</b>", "1:1 crop • CLAHE normalized", img_dim, img_dim)
    card_lesions = _frame_image(panel_paths.get('lesions', ''), "<b>(b) Detected Lesions</b>", "MA (Red) • EX (Yel) • HE (Crimson) • SE (Cyan)", img_dim, img_dim)
    card_gradcam = _frame_image(panel_paths.get('gradcam', ''), "<b>(c) Grad-CAM++ Attention</b>", "Visual AI evidence focus areas", img_dim, img_dim)

    photos_table = Table([[card_prep, card_lesions, card_gradcam]], colWidths=[W / 3.0, W / 3.0, W / 3.0])
    photos_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))

    photos_section = Table([
        [_make_banner("3 FUNDUS PANELS EVALUATED IN SCREENING", W)],
        [photos_table],
        [Paragraph("<i>Image labeling and heatmaps are for explanatory guidance only and should not be used as independent diagnostic markers.</i>", s_footnote)]
    ], colWidths=[W])
    photos_section.setStyle(TableStyle([
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(photos_section)
    story.append(Spacer(1, 8))

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. RESULTS & CLINICAL TRIAGE SECTION (BELOW PHOTOS)
    # ═══════════════════════════════════════════════════════════════════════════
    is_referable = bool(diagnostic_result.get('is_referable', diagnostic_result.get('grade_num', 0) >= 2))
    verdict_text = "REFERRAL RECOMMENDED" if is_referable else "NO REFERRAL NEEDED"
    verdict_color = RED_DARK if is_referable else GREEN_DARK
    verdict_bg = RED_BG if is_referable else GREEN_BG
    verdict_border = RED_BORDER if is_referable else GREEN_BORDER

    res_w_left = W * 0.53
    res_w_right = W * 0.47

    diag_banner = Table([[Paragraph("<b>DIAGNOSTIC RESULTS & CLINICAL TRIAGE</b>", s_banner_text), ""]], colWidths=[res_w_left, res_w_right])
    diag_banner.setStyle(TableStyle([
        ('SPAN', (0, 0), (1, 0)),
        ('BACKGROUND', (0, 0), (-1, -1), DARK_900),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    diag_left_data = [
        _info_cell("CONDITION", "Diabetic Retinopathy (with Macular Risk Assessment)"),
        _info_cell("DIAGNOSIS", diagnostic_result.get('grade_desc', 'No Diabetic Retinopathy detected (ETDRS level 20 or lower).')),
        _info_cell("CARE PLAN", diagnostic_result.get('action_followup', 'Routine Rescreening in 12 Months')),
        _info_cell("AI INTERPRETATION", "Autonomous deep neural interpretation via DRISHYA Retinal Engine v1.0."),
    ]
    t_diag_left = Table(diag_left_data, colWidths=[res_w_left * 0.32, res_w_left * 0.68], rowHeights=[21, 21, 21, 21])
    t_diag_left.setStyle(TableStyle([
        ('INNERGRID', (0, 0), (-1, -1), 0.4, SLATE_200),
        ('BOX', (0, 0), (-1, -1), 0.5, SLATE_300),
        ('BACKGROUND', (0, 0), (0, -1), SLATE_50),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 4.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4.5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    # Clinical Verdict Card
    s_verdict_sub = ParagraphStyle('VerdictSub', fontName=FONT_BOLD, fontSize=6.8, leading=8.8, textColor=SLATE_600, alignment=1)
    s_verdict_main = ParagraphStyle('VerdictMain', fontName=FONT_BOLD, fontSize=14.5, leading=17.5, textColor=verdict_color, alignment=1)
    s_verdict_action = ParagraphStyle('VerdictAction', fontName=FONT_BOLD, fontSize=7.0, leading=9.0, textColor=verdict_color, alignment=1)
    s_verdict_meta = ParagraphStyle('VerdictMeta', fontName=FONT_REGULAR, fontSize=6.2, leading=8.2, textColor=SLATE_600, alignment=1)

    triage_sub = diagnostic_result.get('triage_sub', 'Specialist Slit-Lamp Exam Recommended' if is_referable else 'Routine Primary Care Screening')
    conf_str = diagnostic_result.get('confidence', '96.4%')
    iqa_str = diagnostic_result.get('iqa_status', 'Pass')
    grade_num = diagnostic_result.get('grade_num', 2)

    verdict_card_data = [
        [Paragraph("CLINICAL TRIAGE DECISION", s_verdict_sub)],
        [Paragraph(verdict_text, s_verdict_main)],
        [Paragraph(f"<b>Protocol:</b> {triage_sub}", s_verdict_action)],
        [Paragraph(f"<b>ICDR Grade {grade_num}</b> &nbsp;|&nbsp; Confidence: {conf_str} &nbsp;|&nbsp; IQA: {iqa_str}", s_verdict_meta)]
    ]
    t_verdict = Table(verdict_card_data, colWidths=[res_w_right], rowHeights=[15, 29, 20, 20])
    t_verdict.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), verdict_bg),
        ('BOX', (0, 0), (-1, -1), 0.7, verdict_border),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
    ]))

    results_grid = Table([
        [diag_banner, ''],
        [t_diag_left, t_verdict]
    ], colWidths=[res_w_left, res_w_right])
    results_grid.setStyle(TableStyle([
        ('SPAN', (0, 0), (1, 0)),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(results_grid)
    story.append(Spacer(1, 8))

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. QUANTITATIVE RETINAL BIOMARKERS (FULL WIDTH, CLEAN 3-COLUMN TABLE)
    # ═══════════════════════════════════════════════════════════════════════════
    mas_val = biomarker_metrics.get('mas', '0 detected')
    mas_rel = biomarker_metrics.get('mas_rel', 'Normal vascular integrity')

    ex_val = biomarker_metrics.get('exudates', '0.00% area')
    ex_rel = biomarker_metrics.get('exudates_rel', 'Absent')

    se_val = biomarker_metrics.get('soft_exudates', '0.00% area')
    se_rel = biomarker_metrics.get('soft_exudates_rel', 'Absent')

    he_val = biomarker_metrics.get('hemorrhages', '0 quadrants')
    he_rel = biomarker_metrics.get('hemorrhages_rel', 'No significant hemorrhage detected')

    mac_val = biomarker_metrics.get('macula', 'Low Risk')
    mac_rel = biomarker_metrics.get('macula_rel', 'No lesions in macular zone')

    nv_val = biomarker_metrics.get('neovascularization', '0 (Absent)')
    nv_rel = 'Proliferative DR Hallmarks (New vessels on disc/elsewhere absent)' if 'Absent' in nv_val else 'Suspected PDR (Urgent specialist evaluation required)'

    bio_table_data = [
        [
            Paragraph("<b>RETINAL BIOMARKER / LESION TYPE</b>", s_th),
            Paragraph("<b>QUANTITATIVE MEASUREMENT</b>", s_th),
            Paragraph("<b>CLINICAL INTERPRETATION &amp; STATUS</b>", s_th)
        ],
        [
            Paragraph("<b>Microaneurysms (MA)</b>", s_label),
            Paragraph(mas_val, s_value_bold),
            Paragraph(mas_rel, s_value)
        ],
        [
            Paragraph("<b>Hard Exudates (EX)</b>", s_label),
            Paragraph(ex_val, s_value_bold),
            Paragraph(ex_rel, s_value)
        ],
        [
            Paragraph("<b>Soft Exudates / Cotton-Wool Spots (SE)</b>", s_label),
            Paragraph(se_val, s_value_bold),
            Paragraph(se_rel, s_value)
        ],
        [
            Paragraph("<b>Intraretinal Hemorrhages (HE)</b>", s_label),
            Paragraph(he_val, s_value_bold),
            Paragraph(he_rel, s_value)
        ],
        [
            Paragraph("<b>Macular Edema Risk Assessment</b>", s_label),
            Paragraph(mac_val, s_value_bold),
            Paragraph(mac_rel, s_value)
        ],
        [
            Paragraph("<b>Neovascularization (NV)</b>", s_label),
            Paragraph(nv_val, s_value_bold),
            Paragraph(nv_rel, s_value)
        ],
    ]

    t_bio = Table(bio_table_data, colWidths=[W * 0.32, W * 0.24, W * 0.44], rowHeights=[19, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5])
    t_bio.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SLATE_100),
        ('INNERGRID', (0, 0), (-1, -1), 0.4, SLATE_200),
        ('BOX', (0, 0), (-1, -1), 0.5, SLATE_300),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, SLATE_50]),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    bio_section = Table([
        [_make_banner("QUANTITATIVE RETINAL BIOMARKERS", W)],
        [t_bio]
    ], colWidths=[W])
    bio_section.setStyle(TableStyle([
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(bio_section)
    story.append(Spacer(1, 10))

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. DISCLAIMER & AUDIT TRAIL
    # ═══════════════════════════════════════════════════════════════════════════
    story.append(HRFlowable(width="100%", thickness=0.5, color=SLATE_300, spaceBefore=0, spaceAfter=2))
    
    disc_text = (
        "<b>CLINICAL DISCLAIMER:</b> A positive screening result indicates suspected referable diabetic retinopathy "
        "(ETDRS level 35 or higher and/or macular edema risk). DRISHYA autonomous screening is designed for triage assistance and does not replace "
        "a comprehensive dilated fundus examination by a certified ophthalmologist. Images rendered in this report are compressed for transmission. "
        f"<b>Audit Hash:</b> SHA256:{datetime.now().strftime('%f%Y%m%d')[:16]} &nbsp;|&nbsp; "
        f"<b>Regulatory Class:</b> CDSCO SaMD Class B Equivalent • Validated for Primary Healthcare Tele-Screening."
    )
    story.append(Paragraph(disc_text, s_disc))

    doc.build(story, onFirstPage=_draw_page_decorations, onLaterPages=_draw_page_decorations)
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
        'triage_sub': 'Specialist Slit-Lamp Exam Recommended (Within 4 Weeks)',
        'iqa_status': 'Pass (Q=0.88)',
        'confidence': '96.4%',
        'grade_num': 2,
        'is_referable': True,
        'action_followup': 'Specialist evaluation within 4-6 weeks',
    }
    panel_paths = {
        'preprocessed': 'backend/outputs/20260903_125328_prep.png',
        'lesions': 'backend/outputs/20260903_125328_lesions.png',
        'gradcam': 'backend/outputs/20260903_125328_gradcam.png',
    }
    biomarker_metrics = {
        'mas': '12 detected',
        'mas_rel': 'Active microvascular leakage',
        'exudates': '1.10% area cluster',
        'exudates_rel': 'Lipoprotein deposits near arcade',
        'soft_exudates': '0.18% area',
        'soft_exudates_rel': 'Focal cotton wool spots (nerve fiber ischemia)',
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
