import io
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# SMTP Config
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

def generate_checkup_pdf(report_data: dict) -> bytes:
    """Generates a PDF report from the checkup data."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    styles.add(ParagraphStyle(name='SectionHeader', parent=styles['Heading2'], spaceAfter=12, textColor=colors.darkblue))
    styles.add(ParagraphStyle(name='MetricLabel', parent=styles['Normal'], fontName='Helvetica-Bold'))
    
    story = []
    
    # Title
    title = f"Business Checkup Report: {report_data['business_info'].get('name', 'Business')}"
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 12))
    
    # calculated_scores Section
    scores = report_data.get('calculated_scores', {})
    score_data = [
        ['Metric', 'Value'],
        ['Visibility Score', f"{scores.get('visibility_score', 0)}/100"],
        ['Est. Monthly Revenue Leakage', f"${scores.get('estimated_monthly_revenue_leakage', 0):,}"],
        ['Est. Loss Percentage', f"{scores.get('estimated_loss_percentage', 0)}%"]
    ]
    t = Table(score_data, colWidths=[3*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(Paragraph("Executive Summary", styles['SectionHeader']))
    story.append(t)
    story.append(Spacer(1, 24))

    # google_metrics Section
    g_metrics = report_data.get('google_metrics', {})
    story.append(Paragraph("Google Business Profile Performance", styles['SectionHeader']))
    
    g_data = [
        ['Metric', 'Value'],
        ['Rating', f"{g_metrics.get('rating', 'N/A')} ({g_metrics.get('review_count', 0)} reviews)"],
        ['Category', g_metrics.get('category', 'N/A')],
        ['Verification Status', g_metrics.get('verification_status', 'N/A')],
        ['Has Photos', 'Yes' if g_metrics.get('has_photos') else 'No']
    ]
    
    # Profile Completeness
    completeness = g_metrics.get('profile_completeness', {})
    completeness_str = ", ".join([k.replace('has_', '').title() for k, v in completeness.items() if v])
    g_data.append(['Profile Completeness', completeness_str if completeness_str else "None"])

    t_google = Table(g_data, colWidths=[3*inch, 3*inch])
    t_google.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    story.append(t_google)
    story.append(Spacer(1, 24))

    # website_metrics Section
    w_metrics = report_data.get('website_metrics', {})
    story.append(Paragraph("Website Performance", styles['SectionHeader']))
    
    w_data = [
        ['Metric', 'Score'],
        ['Mobile Performance', f"{w_metrics.get('mobile_score', 'N/A')}/100"],
        ['Desktop Performance', f"{w_metrics.get('desktop_score', 'N/A')}/100"],
        ['Core Web Vitals', w_metrics.get('core_web_vitals_summary', 'N/A')],
        ['HTTPS Secured', 'Yes' if w_metrics.get('https_enabled') else 'No']
    ]

    t_website = Table(w_data, colWidths=[3*inch, 2*inch])
    t_website.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    story.append(t_website)
    story.append(Spacer(1, 24))
    
    # Business Info Footer
    b_info = report_data.get('business_info', {})
    footer_text = f"Report generated for: {b_info.get('name')}<br/>Address: {b_info.get('address')}<br/>Website: {b_info.get('website')}"
    story.append(Paragraph(footer_text, styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def send_checkup_email_report(to_email: str, report_data: dict):
    """Generates PDF and sends it via email."""
    # Reload env vars to ensure we have the latest credentials
    from dotenv import load_dotenv
    load_dotenv()
    
    SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    PORT = int(os.getenv("SMTP_PORT", 587))
    USER = os.getenv("SMTP_USER")
    PASSWORD = os.getenv("SMTP_PASSWORD")

    if not USER or not PASSWORD:
        print(f"SMTP credentials not configured (User: {USER}). Skipping email.")
        return

    try:
        pdf_bytes = generate_checkup_pdf(report_data)
        business_name = report_data['business_info'].get('name', 'Business')
        
        msg = MIMEMultipart()
        msg['From'] = USER
        msg['To'] = to_email
        msg['Subject'] = f"Your Business Checkup Report for {business_name}"
        
        body = f"""Hello,

Please find attached the detailed Business Checkup Report for {business_name}.

Summary:
- Visibility Score: {report_data['calculated_scores'].get('visibility_score')}/100
- Est. Monthly Revenue Leakage: ${report_data['calculated_scores'].get('estimated_monthly_revenue_leakage', 0):,}

Best regards,
Your Business Checkup Team
"""
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF
        filename = f"{business_name.replace(' ', '_')}_Report.pdf"
        part = MIMEApplication(pdf_bytes, Name=filename)
        part['Content-Disposition'] = f'attachment; filename="{filename}"'
        msg.attach(part)
        
        # Send
        with smtplib.SMTP(SERVER, PORT) as server:
            server.starttls()
            server.login(USER, PASSWORD)
            server.send_message(msg)
            
        print(f"Email sent successfully to {to_email}")
        
    except Exception as e:
        print(f"Failed to send email: {e}")
