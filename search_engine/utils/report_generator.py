import os
import io
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.units import inch
from datetime import datetime

class ReportGenerator:
    """
    Generates professional PDF reports for Task 2 Classification results.
    """
    
    def __init__(self, output_path):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self._set_custom_styles()

    def _set_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name='CenterTitle',
            parent=self.styles['Heading1'],
            alignment=1,
            spaceAfter=20,
            textColor=colors.HexColor('#4f46e5')
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHead',
            parent=self.styles['Heading2'],
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.HexColor('#1e293b')
        ))

    def generate_full_report(self, metrics, dataset_stats):
        """Build the complete PDF report."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []

        # 1. Title Page
        elements.append(Spacer(1, 2 * inch))
        elements.append(Paragraph("Machine Learning Portfolio Report", self.styles['CenterTitle']))
        elements.append(Paragraph("Task 2: Multi-label Document Classification", self.styles['Heading3']))
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y')}", self.styles['Normal']))
        elements.append(Paragraph("Research Centre for Computational Science", self.styles['Normal']))
        elements.append(PageBreak())

        # 2. Executive Summary
        elements.append(Paragraph("1. Executive Summary", self.styles['SectionHead']))
        summary_text = (
            "This report summarizes the performance of the Multinomial Naive Bayes classifier "
            "on the expanded news dataset. The system achieves a balanced performance across "
            "Business, Entertainment, and Health categories."
        )
        elements.append(Paragraph(summary_text, self.styles['Normal']))
        elements.append(Spacer(1, 0.2 * inch))

        # 3. Model Performance Table
        elements.append(Paragraph("2. Performance Metrics", self.styles['SectionHead']))
        table_data = [['Category', 'Precision', 'Recall', 'F1-Score']]
        
        # Add per-class metrics
        class_metrics = metrics.get('classification_report', {})
        for label, scores in class_metrics.items():
            if isinstance(scores, dict):
                table_data.append([
                    label.capitalize(),
                    f"{scores.get('precision', 0):.2f}",
                    f"{scores.get('recall', 0):.2f}",
                    f"{scores.get('f1-score', 0):.2f}"
                ])
        
        # Add averages
        table_data.append([
            'Weighted Average',
            f"{metrics.get('weighted_avg_precision', 0):.2f}",
            f"{metrics.get('weighted_avg_recall', 0):.2f}",
            f"{metrics.get('weighted_avg_f1', 0):.2f}"
        ])

        t = Table(table_data, colWidths=[2 * inch] * 4)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.4 * inch))

        # 4. Dataset Statistics
        elements.append(Paragraph("3. Dataset Statistics", self.styles['SectionHead']))
        stats_data = [
            ['Metric', 'Value'],
            ['Total Documents', str(dataset_stats.get('total_documents', ''))],
            ['Unique Labels', str(dataset_stats.get('unique_labels_count', ''))],
            ['Avg Labels / Doc', str(dataset_stats.get('avg_labels_per_doc', ''))],
            ['Collection Period', dataset_stats.get('date_range', '')]
        ]
        st = Table(stats_data, colWidths=[3 * inch, 2 * inch])
        st.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
        ]))
        elements.append(st)
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        
        with open(self.output_path, 'wb') as f:
            f.write(buffer.read())
            
        return self.output_path

    @staticmethod
    def create_confusion_matrix_plot(cm, label, save_path):
        """Generate confusion matrix image using matplotlib."""
        plt.figure(figsize=(4, 4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix: {label}')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return save_path
    @staticmethod
    def generate_metrics_csv(metrics):
        """Generate a CSV string from per-label metrics."""
        import csv
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['Category', 'Precision', 'Recall', 'F1-Score', 'Support'])
        
        # Data rows
        per_label = metrics.get('per_label', {})
        for label, scores in per_label.items():
            writer.writerow([
                label,
                round(scores.get('precision', 0), 4),
                round(scores.get('recall', 0), 4),
                round(scores.get('f1', 0), 4),
                scores.get('support', 0)
            ])
            
        return output.getvalue()
