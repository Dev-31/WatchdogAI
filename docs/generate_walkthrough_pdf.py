"""
Generate PDF Documentation for Watchdog AI
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from datetime import datetime

def create_pdf():
    doc = SimpleDocTemplate(
        "Watchdog_AI_Technical_Walkthrough.pdf",
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#1a365d'),
        alignment=TA_CENTER
    )
    
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.HexColor('#2c5282')
    )
    
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading2'],
        fontSize=13,
        spaceAfter=8,
        spaceBefore=14,
        textColor=colors.HexColor('#2b6cb0')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        leading=14
    )
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Normal'],
        fontSize=9,
        fontName='Courier',
        backColor=colors.HexColor('#f7fafc'),
        spaceAfter=8,
        leftIndent=10
    )
    
    key_point_style = ParagraphStyle(
        'KeyPoint',
        parent=styles['Normal'],
        fontSize=10,
        backColor=colors.HexColor('#ebf8ff'),
        borderColor=colors.HexColor('#3182ce'),
        borderWidth=1,
        borderPadding=8,
        spaceAfter=12,
        leftIndent=10
    )
    
    story = []
    
    # Title
    story.append(Paragraph("🛡️ Watchdog AI", title_style))
    story.append(Paragraph("Technical Code Walkthrough", styles['Heading2']))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 30))
    
    # Overview
    story.append(Paragraph("Overview", h1_style))
    story.append(Paragraph(
        "Watchdog AI is a <b>data quality validation and curation pipeline</b> designed to clean AI training datasets. "
        "It uses <b>rule-based NLP detection</b> (no external LLM dependencies) and integrates with the "
        "<b>Climatiq API</b> for carbon footprint tracking.",
        body_style
    ))
    story.append(Spacer(1, 15))
    
    # Architecture overview
    story.append(Paragraph("The 4-Step Pipeline:", h2_style))
    pipeline_data = [
        ['Step', 'Module', 'Purpose'],
        ['1', 'Misinformation Detector', 'Remove high-risk/misleading content'],
        ['2', 'Quality Scorer', 'Filter low-quality items'],
        ['3', 'Redundancy Detector', 'Remove duplicates (exact + semantic)'],
        ['4', 'Sustainability Tracker', 'Calculate carbon footprint savings'],
    ]
    pipeline_table = Table(pipeline_data, colWidths=[0.5*inch, 2*inch, 3*inch])
    pipeline_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(pipeline_table)
    story.append(PageBreak())
    
    # Module 1: Misinformation Detector
    story.append(Paragraph("1. Misinformation Detector", h1_style))
    story.append(Paragraph("<b>File:</b> src/misinformation_detector.py", body_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>🔑 KEY POINT:</b> This module uses regex-based pattern matching — NOT a machine learning model or LLM. "
        "It's fast, deterministic, and has no API costs.",
        key_point_style
    ))
    
    story.append(Paragraph("Core Functions:", h2_style))
    misinfo_funcs = [
        ['Function', 'Purpose'],
        ['detect_generic_content()', 'Calculates "vagueness score" using filler words count'],
        ['detect_clickbait()', 'Scans for phrases like "you won\'t believe"'],
        ['detect_excessive_caps()', 'Calculates uppercase ratio (>60% = high risk)'],
        ['detect_excessive_punctuation()', 'Counts !!! and ??? patterns'],
        ['check_source_credibility()', 'Boosts .gov/.edu, penalizes blog domains'],
        ['analyze_text()', 'Main entry: combines all signals with weighted scoring'],
        ['batch_analyze()', 'Processes multiple texts in a loop'],
    ]
    func_table = Table(misinfo_funcs, colWidths=[2.2*inch, 4*inch])
    func_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(func_table)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Weighted Scoring System:", h2_style))
    weights_data = [
        ['Signal', 'Weight', 'Description'],
        ['Toxicity', '40%', 'Highest weight - triggers high risk immediately'],
        ['Pattern Match', '35%', 'Conspiracy, misinformation phrases'],
        ['Clickbait', '25%', 'Sensationalist language'],
        ['Caps/Punctuation', '10% each', 'Formatting red flags'],
        ['Generic Content', '10%', 'Vague, low-value text'],
        ['Source', '10%', 'Domain credibility'],
    ]
    weights_table = Table(weights_data, colWidths=[1.5*inch, 0.8*inch, 3.5*inch])
    weights_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#38a169')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0fff4')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(weights_table)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Risk Level Thresholds:", h2_style))
    story.append(Paragraph("• <b>High Risk:</b> Score ≥0.60 OR ≥3 flags OR toxicity detected → FILTERED OUT", body_style))
    story.append(Paragraph("• <b>Medium Risk:</b> Score ≥0.35 OR ≥1 flag", body_style))
    story.append(Paragraph("• <b>Low Risk:</b> Clean content - kept in dataset", body_style))
    story.append(PageBreak())
    
    # Module 2: Quality Scorer
    story.append(Paragraph("2. Data Quality Scorer", h1_style))
    story.append(Paragraph("<b>File:</b> src/quality_scorer.py", body_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>🔑 KEY POINT:</b> Evaluates text structural quality — not meaning. Uses statistical metrics "
        "without any ML model to determine if data is well-formed for training.",
        key_point_style
    ))
    
    story.append(Paragraph("Core Functions:", h2_style))
    quality_funcs = [
        ['Function', 'Purpose'],
        ['_calculate_text_completeness()', 'Checks required (text) + optional fields (title, source, etc.)'],
        ['_calculate_text_length_score()', 'Scores by char count: <20=0.2, 200+=1.0'],
        ['_calculate_word_count_score()', 'Optimal range: 50-500 words = 1.0'],
        ['_calculate_language_quality()', 'Capitalization, punctuation, sentence structure'],
        ['_calculate_information_density()', 'KEY: Lexical diversity (unique/total words)'],
        ['_calculate_spam_indicators()', 'Detects "SHOCKING", "FREE", "CLICK HERE"'],
        ['score_data()', 'Main entry: returns overall_score, quality_level, recommendation'],
    ]
    q_table = Table(quality_funcs, colWidths=[2.5*inch, 3.7*inch])
    q_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(q_table)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Quality Scoring Weights:", h2_style))
    q_weights = [
        ['Metric', 'Weight'],
        ['Information Density', '28% (most important)'],
        ['Language Quality', '25%'],
        ['Word Count', '13%'],
        ['Completeness', '12%'],
        ['Spam Check', '12%'],
        ['Text Length', '10%'],
    ]
    qw_table = Table(q_weights, colWidths=[2*inch, 2.5*inch])
    qw_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#805ad5')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#faf5ff')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(qw_table)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Quality Levels & Recommendations:", h2_style))
    story.append(Paragraph("• <b>High (≥0.80):</b> \"Excellent quality - keep\"", body_style))
    story.append(Paragraph("• <b>Medium (≥0.60):</b> \"Good quality - minor review\"", body_style))
    story.append(Paragraph("• <b>Low (≥0.40):</b> \"Fair quality - review before keeping\"", body_style))
    story.append(Paragraph("• <b>Very Low (<0.40):</b> \"Poor quality - consider removing\" → FILTERED OUT", body_style))
    story.append(PageBreak())
    
    # Module 3: Redundancy Detector
    story.append(Paragraph("3. Redundancy Detector", h1_style))
    story.append(Paragraph("<b>File:</b> src/redundancy_detector.py", body_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>🔑 KEY POINT:</b> Uses TF-IDF vectorization + cosine similarity from scikit-learn for semantic "
        "duplicate detection. Also does exact-match via MD5 hashing. No LLM required.",
        key_point_style
    ))
    
    story.append(Paragraph("Core Functions:", h2_style))
    dup_funcs = [
        ['Function', 'Purpose'],
        ['_normalize_text()', 'Lowercase, strip whitespace, remove punctuation'],
        ['_compute_hash()', 'MD5 hash for O(1) exact duplicate lookup'],
        ['find_exact_duplicates()', 'Hash-based matching, groups duplicates'],
        ['find_semantic_duplicates()', 'TF-IDF + Cosine Similarity (scikit-learn)'],
        ['find_duplicates()', 'Main entry: supports exact, semantic, or both'],
    ]
    dup_table = Table(dup_funcs, colWidths=[2.2*inch, 4*inch])
    dup_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(dup_table)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("TF-IDF Configuration:", h2_style))
    story.append(Paragraph("TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))", code_style))
    story.append(Paragraph("• <b>max_features=1000:</b> Cap vocabulary size for efficiency", body_style))
    story.append(Paragraph("• <b>stop_words='english':</b> Remove common words (the, is, at)", body_style))
    story.append(Paragraph("• <b>ngram_range=(1,2):</b> Word + bigram features for context", body_style))
    story.append(Paragraph("• <b>Default threshold:</b> 85% similarity = duplicate", body_style))
    story.append(PageBreak())
    
    # Module 4: Sustainability Tracker
    story.append(Paragraph("4. Sustainability Tracker", h1_style))
    story.append(Paragraph("<b>File:</b> src/sustainability_tracker.py", body_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>🔑 KEY POINT:</b> Integrates with the Climatiq API to calculate real carbon emissions. "
        "Falls back to regional averages if API unavailable.",
        key_point_style
    ))
    
    story.append(Paragraph("Core Functions:", h2_style))
    sus_funcs = [
        ['Function', 'Purpose'],
        ['_get_carbon_intensity_from_climatiq()', 'API call to Climatiq /estimate endpoint'],
        ['calculate_data_size_mb()', 'Recursively calculates UTF-8 byte size'],
        ['track_operation()', 'Logs energy/carbon for text processing, inference, training'],
        ['calculate_savings()', 'KEY: Computes environmental savings from data reduction'],
        ['get_session_summary()', 'Returns totals + equivalencies (trees, car miles)'],
    ]
    sus_table = Table(sus_funcs, colWidths=[2.8*inch, 3.4*inch])
    sus_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(sus_table)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Energy Metrics:", h2_style))
    energy_data = [
        ['Operation', 'Energy (kWh)'],
        ['Text Processing (per MB)', '0.00025'],
        ['Inference (per 1K tokens)', '0.000004'],
        ['Model Training (per epoch)', '0.5'],
    ]
    energy_table = Table(energy_data, colWidths=[2.5*inch, 2*inch])
    energy_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#38a169')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0fff4')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(energy_table)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Carbon Intensity Fallbacks:", h2_style))
    carbon_data = [
        ['Region', 'kg CO₂/kWh'],
        ['Global Average', '0.475'],
        ['US Average', '0.386'],
        ['EU Average', '0.295'],
        ['Renewable', '0.05'],
    ]
    carbon_table = Table(carbon_data, colWidths=[2*inch, 1.5*inch])
    carbon_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dd6b20')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fffaf0')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(carbon_table)
    story.append(PageBreak())
    
    # Module 5: Dataset Processor
    story.append(Paragraph("5. Dataset Processor (Orchestration)", h1_style))
    story.append(Paragraph("<b>File:</b> src/dataset_processor.py", body_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>🔑 KEY POINT:</b> This is the orchestration layer that runs the full 4-step pipeline "
        "on entire datasets (CSV, JSON, JSONL).",
        key_point_style
    ))
    
    story.append(Paragraph("Core Functions:", h2_style))
    proc_funcs = [
        ['Function', 'Purpose'],
        ['process_dataframe()', 'Main pipeline: runs all 4 steps, returns cleaned DataFrame'],
        ['process_csv()', 'Load CSV and process'],
        ['process_json()', 'Handle JSON array or single object'],
        ['process_jsonl()', 'Load newline-delimited JSON'],
        ['save_results()', 'Export as CSV/JSON/JSONL + statistics'],
    ]
    proc_table = Table(proc_funcs, colWidths=[2*inch, 4.2*inch])
    proc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(proc_table)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Default Parameters:", h2_style))
    story.append(Paragraph("• <b>quality_threshold = 0.5:</b> Minimum quality score to keep", body_style))
    story.append(Paragraph("• <b>remove_high_risk = True:</b> Filter misinformation", body_style))
    story.append(Paragraph("• <b>remove_duplicates = True:</b> Remove redundant entries", body_style))
    story.append(Paragraph("• <b>similarity_threshold = 0.85:</b> 85% similarity = duplicate", body_style))
    story.append(Spacer(1, 15))
    
    # Module 6: REST API
    story.append(Paragraph("6. Flask REST API", h1_style))
    story.append(Paragraph("<b>File:</b> api/app.py", body_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>🔑 KEY POINT:</b> Production-ready REST API with CORS support, exposing all modules as HTTP endpoints.",
        key_point_style
    ))
    
    story.append(Paragraph("API Endpoints:", h2_style))
    api_data = [
        ['Endpoint', 'Method', 'Purpose'],
        ['/', 'GET', 'API info and version'],
        ['/health', 'GET', 'Health check for load balancers'],
        ['/analyze', 'POST', 'Single text analysis'],
        ['/analyze/batch', 'POST', 'Batch text analysis'],
        ['/quality', 'POST', 'Quality scoring only'],
        ['/duplicates', 'POST', 'Find duplicates in text list'],
        ['/process', 'POST', 'Full pipeline on dataset'],
        ['/sustainability', 'POST', 'Calculate carbon savings'],
    ]
    api_table = Table(api_data, colWidths=[1.5*inch, 0.8*inch, 3.5*inch])
    api_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(api_table)
    story.append(PageBreak())
    
    # Key Selling Points
    story.append(Paragraph("Key Selling Points", h1_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("1. No External LLM Dependencies", h2_style))
    story.append(Paragraph("• All detection uses rule-based regex patterns and scikit-learn TF-IDF", body_style))
    story.append(Paragraph("• Zero API costs for core analysis (only Climatiq for sustainability)", body_style))
    story.append(Paragraph("• Deterministic, explainable results", body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("2. Production-Ready Architecture", h2_style))
    story.append(Paragraph("• Modular components usable standalone or together", body_style))
    story.append(Paragraph("• REST API with CORS for frontend integration", body_style))
    story.append(Paragraph("• Supports CSV, JSON, JSONL dataset formats", body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("3. Explainable AI", h2_style))
    story.append(Paragraph("• Every detection provides human-readable explanations", body_style))
    story.append(Paragraph("• Transparent scoring with visible weights", body_style))
    story.append(Paragraph("• Actionable recommendations (keep, review, remove)", body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("4. Sustainability Tracking", h2_style))
    story.append(Paragraph("• Real carbon footprint calculations via Climatiq API", body_style))
    story.append(Paragraph("• Quantifies environmental impact of data reduction", body_style))
    story.append(Paragraph("• Converts to relatable metrics (trees planted, car miles)", body_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("5. Computational Efficiency", h2_style))
    story.append(Paragraph("• TF-IDF vectorization is O(n) for document count", body_style))
    story.append(Paragraph("• MD5 hashing provides O(1) exact duplicate lookup", body_style))
    story.append(Paragraph("• Processes 1000s of items per second on standard hardware", body_style))
    
    # Build PDF
    doc.build(story)
    print("✅ PDF generated: Watchdog_AI_Technical_Walkthrough.pdf")

if __name__ == "__main__":
    create_pdf()
