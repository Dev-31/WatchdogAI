"""
Generate PDF with 10 Demo Inputs for Watchdog AI Demonstration
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from datetime import datetime

def create_demo_pdf():
    doc = SimpleDocTemplate(
        "Watchdog_AI_Demo_Inputs.pdf",
        pagesize=A4,
        rightMargin=0.6*inch,
        leftMargin=0.6*inch,
        topMargin=0.6*inch,
        bottomMargin=0.6*inch
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=22,
        spaceAfter=20,
        textColor=colors.HexColor('#1a365d'),
        alignment=TA_CENTER
    )
    
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=15,
        textColor=colors.HexColor('#2c5282')
    )
    
    input_style = ParagraphStyle(
        'Input',
        parent=styles['Normal'],
        fontSize=9,
        fontName='Courier',
        backColor=colors.HexColor('#1a202c'),
        textColor=colors.white,
        spaceAfter=10,
        leftIndent=8,
        rightIndent=8,
        borderPadding=10,
        leading=13
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        leading=13
    )
    
    expected_style = ParagraphStyle(
        'Expected',
        parent=styles['Normal'],
        fontSize=9,
        backColor=colors.HexColor('#f0fff4'),
        borderColor=colors.HexColor('#38a169'),
        spaceAfter=12,
        leftIndent=8,
        borderPadding=8
    )
    
    story = []
    
    # Title
    story.append(Paragraph("🛡️ Watchdog AI - Demo Inputs", title_style))
    story.append(Paragraph("10 Test Cases for Professional Demonstrations", styles['Heading3']))
    story.append(Spacer(1, 15))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph(
        "Use these carefully crafted inputs to demonstrate each detection capability of Watchdog AI. "
        "Each example is designed to trigger specific flags and showcase how the system identifies and explains issues.",
        body_style
    ))
    story.append(Spacer(1, 15))
    
    # Demo 1: High-Quality Clean Content
    story.append(Paragraph("Demo 1: High-Quality Clean Content (SHOULD PASS)", h1_style))
    story.append(Paragraph("<b>Purpose:</b> Show that good content is correctly identified and kept.", body_style))
    story.append(Paragraph(
        "The development of renewable energy sources has accelerated significantly over the past decade. "
        "Solar panel efficiency has improved from 15% to over 22% in commercial applications, while costs "
        "have decreased by approximately 89% since 2010. Wind power capacity has grown at an annual rate "
        "of 12%, with offshore installations showing particular promise. According to the International "
        "Energy Agency, renewable sources now account for 29% of global electricity generation.",
        input_style
    ))
    story.append(Paragraph(
        "<b>Expected Result:</b> Risk Level: LOW | Quality: HIGH (0.85+) | No flags | Recommendation: Keep",
        expected_style
    ))
    
    # Demo 2: Clickbait Misinformation
    story.append(Paragraph("Demo 2: Clickbait Misinformation (HIGH RISK)", h1_style))
    story.append(Paragraph("<b>Purpose:</b> Detect sensationalist, misleading content.", body_style))
    story.append(Paragraph(
        "You WON'T BELIEVE what doctors don't want you to know!!! This MIRACLE cure will SHOCK you! "
        "Big pharma has been hiding this SECRET for years. Number 5 will blow your mind! Scientists "
        "EXPOSED the truth they tried to cover up. Share this before they delete it!!!",
        input_style
    ))
    story.append(Paragraph(
        "<b>Expected Result:</b> Risk Level: HIGH | Flags: clickbait, suspicious_patterns, excessive_punctuation | Filtered OUT",
        expected_style
    ))
    
    # Demo 3: Toxic Content
    story.append(Paragraph("Demo 3: Toxic Language Detection (HIGH RISK)", h1_style))
    story.append(Paragraph("<b>Purpose:</b> Show toxicity detection capability.", body_style))
    story.append(Paragraph(
        "This product is a complete disaster and anyone who buys it is an idiot. The company is a fraud "
        "and the developers are pathetic losers. I hate everything about this stupid project. "
        "Nobody cares about your opinions, just shut up already.",
        input_style
    ))
    story.append(Paragraph(
        "<b>Expected Result:</b> Risk Level: HIGH | Flags: toxicity | Misinformation Score: 0.70+ | Filtered OUT",
        expected_style
    ))
    
    # Demo 4: Conspiracy Theory
    story.append(Paragraph("Demo 4: Conspiracy Theory Content (HIGH RISK)", h1_style))
    story.append(Paragraph("<b>Purpose:</b> Detect conspiracy-related patterns.", body_style))
    story.append(Paragraph(
        "The deep state doesn't want you to see this. The illuminati and new world order have been "
        "controlling the mainstream media for decades. Wake up sheeple! The truth about the global "
        "conspiracy is being suppressed. They control everything from behind the scenes.",
        input_style
    ))
    story.append(Paragraph(
        "<b>Expected Result:</b> Risk Level: HIGH | Flags: suspicious_patterns (conspiracy keywords) | Filtered OUT",
        expected_style
    ))
    story.append(PageBreak())
    
    # Demo 5: Low Quality / Vague Content
    story.append(Paragraph("Demo 5: Low Quality / Vague Content (LOW QUALITY)", h1_style))
    story.append(Paragraph("<b>Purpose:</b> Show quality scoring for low-information text.", body_style))
    story.append(Paragraph(
        "This thing is really good. It does some stuff that is very nice. Many people like it a lot. "
        "It's quite great actually. Some things about it are really fine. Very good stuff indeed.",
        input_style
    ))
    story.append(Paragraph(
        "<b>Expected Result:</b> Quality: VERY LOW (0.30-0.40) | Issues: Low information density, Generic content | Filtered OUT",
        expected_style
    ))
    
    # Demo 6: Spam Content
    story.append(Paragraph("Demo 6: Spam / Promotional Content (MEDIUM-HIGH RISK)", h1_style))
    story.append(Paragraph("<b>Purpose:</b> Detect spam keywords and patterns.", body_style))
    story.append(Paragraph(
        "CONGRATULATIONS!!! You are our EXCLUSIVE WINNER! CLAIM NOW your FREE prize worth $10,000! "
        "ACT NOW - LIMITED TIME OFFER! CLICK HERE to receive your GUARANTEED reward! This is URGENT! "
        "100% REAL! Don't miss this SHOCKING opportunity!!!",
        input_style
    ))
    story.append(Paragraph(
        "<b>Expected Result:</b> Risk Level: HIGH | Flags: clickbait, excessive_caps, excessive_punctuation, spam | Filtered OUT",
        expected_style
    ))
    
    # Demo 7: Excessive Capitalization
    story.append(Paragraph("Demo 7: Excessive Capitalization (MEDIUM RISK)", h1_style))
    story.append(Paragraph("<b>Purpose:</b> Detect unprofessional formatting.", body_style))
    story.append(Paragraph(
        "THIS IS A VERY IMPORTANT MESSAGE THAT EVERYONE NEEDS TO READ RIGHT NOW. THE INFORMATION "
        "CONTAINED HERE IS CRITICAL AND YOU MUST PAY ATTENTION. PLEASE SHARE WITH EVERYONE YOU KNOW.",
        input_style
    ))
    story.append(Paragraph(
        "<b>Expected Result:</b> Risk Level: MEDIUM | Flags: excessive_caps (80%+) | Quality penalized",
        expected_style
    ))
    
    # Demo 8: Duplicate Detection Test
    story.append(Paragraph("Demo 8: Duplicate Detection (Batch Test)", h1_style))
    story.append(Paragraph("<b>Purpose:</b> Show exact and semantic duplicate detection.", body_style))
    story.append(Paragraph("<b>Input as JSON array for /duplicates endpoint:</b>", body_style))
    story.append(Paragraph(
        '{"texts": [<br/>'
        '  "Machine learning models require large datasets for training.",<br/>'
        '  "Machine learning models require large datasets for training.",<br/>'
        '  "ML models need big data sets to train effectively.",<br/>'
        '  "The weather today is sunny with clear skies.",<br/>'
        '  "Deep learning requires substantial training data."<br/>'
        ']}',
        input_style
    ))
    story.append(Paragraph(
        "<b>Expected Result:</b> 2 duplicates found (1 exact: items 0,1 | 1 semantic: items 0,2,4 similar) | 3 unique remain",
        expected_style
    ))
    story.append(PageBreak())
    
    # Demo 9: Source Credibility
    story.append(Paragraph("Demo 9: Source Credibility Testing", h1_style))
    story.append(Paragraph("<b>Purpose:</b> Show how source affects risk assessment.", body_style))
    story.append(Paragraph("<b>Same text with different sources:</b>", body_style))
    story.append(Paragraph(
        '{"text": "New research shows promising results in cancer treatment studies.",<br/>'
        ' "source": "nature.com"}  → Expected: Source boost (+0.25 credibility)<br/><br/>'
        '{"text": "New research shows promising results in cancer treatment studies.",<br/>'
        ' "source": "breaking-real-truth-news.blog"}  → Expected: Source penalty (-0.2 credibility)',
        input_style
    ))
    story.append(Paragraph(
        "<b>Expected Result:</b> Same content, different risk levels based on source credibility scoring",
        expected_style
    ))
    
    # Demo 10: Full Pipeline Dataset
    story.append(Paragraph("Demo 10: Full Pipeline Processing (Batch)", h1_style))
    story.append(Paragraph("<b>Purpose:</b> Demonstrate complete 4-step pipeline with sustainability metrics.", body_style))
    story.append(Paragraph("<b>Input for /process endpoint:</b>", body_style))
    story.append(Paragraph(
        '{"data": [<br/>'
        '  {"text": "Artificial intelligence continues to transform healthcare diagnostics with improved accuracy.", "source": "ieee.org"},<br/>'
        '  {"text": "SHOCKING!!! Doctors HATE this miracle cure!! Share before deleted!!!", "source": "fake-news.blog"},<br/>'
        '  {"text": "Good stuff. Very nice. Really great things happening.", "source": null},<br/>'
        '  {"text": "Renewable energy investments reached $500 billion globally in 2023.", "source": "reuters.com"},<br/>'
        '  {"text": "The deep state conspiracy is controlling everything wake up sheeple!", "source": null},<br/>'
        '  {"text": "Renewable energy investments hit $500B worldwide last year.", "source": "bbc.com"},<br/>'
        '  {"text": "This idiot product is a complete disaster and fraud.", "source": null}<br/>'
        '], "text_column": "text", "source_column": "source"}',
        input_style
    ))
    story.append(Paragraph(
        "<b>Expected Result:</b><br/>"
        "• Original: 7 items<br/>"
        "• After Misinformation Filter: 4 (removed: clickbait, conspiracy, toxic)<br/>"
        "• After Quality Filter: 3 (removed: vague content)<br/>"
        "• After Duplicate Filter: 2 (removed: semantic duplicate about renewable energy)<br/>"
        "• Sustainability: Shows data reduction %, energy saved, CO₂ saved",
        expected_style
    ))
    story.append(Spacer(1, 20))
    
    # Quick Reference Table
    story.append(Paragraph("Quick Reference: Detection Triggers", h1_style))
    ref_data = [
        ['Detection Type', 'Key Triggers'],
        ['Clickbait', '"you won\'t believe", "shocking", "number X will..."'],
        ['Conspiracy', '"deep state", "illuminati", "wake up sheeple"'],
        ['Toxicity', '"idiot", "stupid", "hate", "disaster", "fraud"'],
        ['Spam', '"FREE", "CLICK HERE", "GUARANTEED", "ACT NOW"'],
        ['Excessive Caps', '>40% uppercase letters in text'],
        ['Excessive Punctuation', '3+ exclamation marks, "!!!" or "???"'],
        ['Low Quality', 'Filler words: "good", "stuff", "very", "really"'],
        ['Low Credibility Source', '.blog, "breaking-news", "secret", "leaked"'],
        ['High Credibility Source', '.gov, .edu, reuters, bbc, nature, ieee'],
    ]
    ref_table = Table(ref_data, colWidths=[1.8*inch, 4.5*inch])
    ref_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(ref_table)
    story.append(PageBreak())
    
    # API Usage Examples
    story.append(Paragraph("API Usage Examples", h1_style))
    story.append(Paragraph("<b>Single Text Analysis (curl):</b>", body_style))
    story.append(Paragraph(
        'curl -X POST http://localhost:5000/analyze \\<br/>'
        '  -H "Content-Type: application/json" \\<br/>'
        '  -d \'{"text": "Your test text here", "source": "example.com"}\'',
        input_style
    ))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("<b>Python Example:</b>", body_style))
    story.append(Paragraph(
        'import requests<br/><br/>'
        'response = requests.post(<br/>'
        '    "http://localhost:5000/analyze",<br/>'
        '    json={"text": "Your test text", "source": "example.com"}<br/>'
        ')<br/>'
        'print(response.json())',
        input_style
    ))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("<b>Frontend Integration (JavaScript):</b>", body_style))
    story.append(Paragraph(
        'fetch("http://localhost:5000/analyze", {<br/>'
        '  method: "POST",<br/>'
        '  headers: {"Content-Type": "application/json"},<br/>'
        '  body: JSON.stringify({text: "Test text", source: "example.com"})<br/>'
        '})<br/>'
        '.then(res => res.json())<br/>'
        '.then(data => console.log(data));',
        input_style
    ))
    
    # Build PDF
    doc.build(story)
    print("✅ PDF generated: Watchdog_AI_Demo_Inputs.pdf")

if __name__ == "__main__":
    create_demo_pdf()
