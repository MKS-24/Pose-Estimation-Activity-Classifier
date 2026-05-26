import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def build_pdf():
    pdf_filename = "report.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles for a premium academic look
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1A365D'), # Deep Blue
        spaceAfter=15,
        alignment=1 # Centered
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4A5568'), # Slate Gray
        spaceAfter=20,
        alignment=1 # Centered
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2B6CB0'), # Royal Blue
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2D3748'), # Charcoal
        spaceAfter=8
    )
    
    caption_style = ParagraphStyle(
        'Caption_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#718096'),
        alignment=1,
        spaceAfter=10
    )
    
    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#2D3748'),
        backColor=colors.HexColor('#EDF2F7'),
        borderColor=colors.HexColor('#CBD5E0'),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=10
    )
    
    story = []
    
    # --- PAGE 1: TITLE & PIPELINE DESIGN ---
    story.append(Paragraph("Computer Vision Assignment: Activity Classification Report", title_style))
    story.append(Paragraph("Topic: Pose Estimation, Joint Geometry & Rule-Based Activity Detection", subtitle_style))
    
    # Divider line
    divider = Table([[""]], colWidths=[540], rowHeights=[2])
    divider.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1A365D')),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(divider)
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("1. Introduction & Learning Objectives", h1_style))
    story.append(Paragraph(
        "This project develops a complete Computer Vision pipeline to detect human poses, smooth joint jitter, "
        "compute joint angles, and classify exercises in real-time. By leveraging Google MediaPipe's state-of-the-art "
        "Pose Landmarker API and a customized temporal filtering technique, the system extracts high-fidelity skeletal representations "
        "from video streams and performs rule-based activity classification.",
        body_style
    ))
    
    story.append(Paragraph("2. Pose Detection & Pre-processing (Smoothing Filter)", h1_style))
    story.append(Paragraph(
        "A pre-trained <b>MediaPipe Pose Landmarker (Heavy Model)</b> is utilized to extract 33 3D body keypoints per frame. "
        "Raw coordinate detections often contain high-frequency jitter due to video compression and sensor noise. "
        "To mitigate this, a temporal <b>Simple Moving Average (SMA)</b> filter was implemented with a window size of 5 frames. "
        "This filter smooths coordinate trajectories over time before angle calculations, preventing classification flickering while preserving motion trends.",
        body_style
    ))
    
    story.append(Paragraph("3. Joint Angle Computation Approach", h1_style))
    story.append(Paragraph(
        "Joint angles are calculated dynamically in 2D space. Given three consecutive joint coordinates: "
        "<b>A</b> (starting joint), <b>B</b> (vertex joint), and <b>C</b> (ending joint), the vectors <b>BA</b> and <b>BC</b> are computed. "
        "The angle θ at vertex B is calculated using the dot product and cosine similarity:",
        body_style
    ))
    
    story.append(Paragraph(
        "Vector BA = A - B &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; Vector BC = C - B<br/>"
        "cos(θ) = (BA · BC) / (||BA|| * ||BC||)<br/>"
        "θ = arccos(clip(cos(θ), -1.0, 1.0)) &times; (180 / π)",
        code_style
    ))
    
    story.append(Paragraph(
        "The pipeline tracks three major angles: <b>Knee Angle</b> (Hip-Knee-Ankle), <b>Hip Angle</b> (Shoulder-Hip-Knee), "
        "and <b>Elbow Angle</b> (Shoulder-Elbow-Wrist). Visibility scores are monitored to automatically select the side "
        "facing the camera (Right side was selected for this video with 96.8% visibility).",
        body_style
    ))
    
    story.append(Paragraph("4. Rule-Based Activity Classification Logic", h1_style))
    story.append(Paragraph(
        "A rule-based classifier is designed to distinguish between <b>Standing</b> and <b>Squatting</b> using joint angle thresholds. "
        "During a squat, both the knee and hip joints flex. The designed heuristic is defined as:",
        body_style
    ))
    story.append(Paragraph(
        "IF (Knee Angle < 140°) AND (Hip Angle < 140°):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;Activity = 'Squatting'<br/>"
        "ELSE:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;Activity = 'Standing'",
        code_style
    ))
    
    story.append(PageBreak())
    
    # --- PAGE 2: RESULTS, PLOTS, AND SCREENSHOTS ---
    story.append(Paragraph("5. Quantitative Evaluation & Results", h1_style))
    story.append(Paragraph(
        "The classification performance was evaluated frame-by-frame against a manually labeled ground truth. "
        "The video contains 559 frames (30 FPS) with 5 distinct repetitions of squats.",
        body_style
    ))
    
    # Performance Table
    data = [
        ['Metric', 'Value (%)', 'Interpretation'],
        ['Overall Accuracy', '82.29%', 'High overall temporal alignment'],
        ['Precision', '100.00%', 'Zero false positives (Standing never misclassified)'],
        ['Recall', '72.73%', 'Conservative transitions at start/end of squats'],
        ['F1-Score', '84.21%', 'Robust overall model representation'],
    ]
    t = Table(data, colWidths=[130, 90, 320])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F7FAFC')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,1), (-1,-1), 4),
        ('TOPPADDING', (0,1), (-1,-1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("6. Analysis of Joint Tracking & Transitions", h1_style))
    
    # Plot Image
    plot_img_path = "angle_tracking.png"
    if os.path.exists(plot_img_path):
        img_w = 480
        img_h = 240
        story.append(Image(plot_img_path, width=img_w, height=img_h))
        story.append(Paragraph("Figure 1: Joint Angles over Time and Activity Classification (Ground Truth vs Prediction).", caption_style))
    else:
        story.append(Paragraph("[Missing tracking plot image]", body_style))
        
    story.append(Paragraph(
        "<b>Observations:</b> The smoothing filter effectively eliminated coordinate noise, producing smooth trajectories "
        "for the knee and hip angles. The 100% precision score indicates that the threshold of 140° is highly specific "
        "to full squatting states. The lower recall (72.73%) is due to the transitional periods where the subject is "
        "partially bent (between 140° and 150°), which are labeled as Squatting in the ground truth but classified as Standing "
        "by the conservative threshold.",
        body_style
    ))
    
    # Screenshots
    story.append(Paragraph("7. Pipeline Visual Outputs (Demo Screenshots)", h1_style))
    
    screenshot_1 = "screenshot_20.png"
    screenshot_2 = "screenshot_90.png"
    
    if os.path.exists(screenshot_1) and os.path.exists(screenshot_2):
        # Resize screenshots to fit side-by-side
        img_w_sc = 240
        img_h_sc = 135
        sc_data = [
            [Image(screenshot_1, width=img_w_sc, height=img_h_sc), Image(screenshot_2, width=img_w_sc, height=img_h_sc)],
            [Paragraph("Frame 20: Classified as Standing", caption_style), Paragraph("Frame 90: Classified as Squatting", caption_style)]
        ]
        sc_table = Table(sc_data, colWidths=[260, 260])
        sc_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(sc_table)
    else:
        story.append(Paragraph("[Missing frame screenshots]", body_style))
        
    doc.build(story)
    print("Report PDF generated successfully!")

if __name__ == "__main__":
    build_pdf()
