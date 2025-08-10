import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from datetime import datetime, date
import base64
from io import BytesIO

class OfflineAnomalyReportGenerator:
    def __init__(self, data_source, total_records=100000):
        """
        Initialize the report generator
        data_source: pandas DataFrame or CSV file path
        total_records: Total number of records processed (for percentage calculation)
        """
        if isinstance(data_source, str):
            self.df = pd.read_csv(data_source)
        else:
            self.df = data_source.copy()
        
        self.report_date = datetime.now().strftime("%Y-%m-%d")
        self.total_records = total_records
        
    def preprocess_data(self):
        """Clean and prepare data for analysis"""
        # Handle missing values
        self.df = self.df.fillna('Unknown')
        
        # Standardize column names (adjust based on your actual column names)
        column_mapping = {
            'status_work': 'Status_Work',
            'subscription_status': 'Subscription_Status', 
            'activation_status': 'Activation_Status',
            'billing_status': 'Billing_Status',
            'vendor_status': 'Vendor_Status',
            'product_name': 'Product_Name',
            'anomaly_type': 'Anomaly_Type',
            'created_date': 'Created_Date'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in self.df.columns:
                self.df.rename(columns={old_name: new_name}, inplace=True)
                
        print(f"Data preprocessed: {len(self.df)} anomaly records loaded")
    
    def calculate_summary_metrics(self):
        """Calculate key metrics for the summary report"""
        anomaly_count = len(self.df)
        
        # Identify critical anomalies (adjust logic based on your criteria)
        critical_conditions = []
        if 'Anomaly_Type' in self.df.columns:
            critical_conditions.append(
                self.df['Anomaly_Type'].str.contains('Critical|High|Direct Anomaly', case=False, na=False)
            )
        
        # Add more critical conditions based on status columns
        status_columns = ['Status_Work', 'Billing_Status', 'Activation_Status']
        for col in status_columns:
            if col in self.df.columns:
                critical_conditions.append(
                    self.df[col].str.contains('Failed|Error|Critical', case=False, na=False)
                )
        
        if critical_conditions:
            critical_mask = pd.concat(critical_conditions, axis=1).any(axis=1)
            critical_count = critical_mask.sum()
        else:
            critical_count = 0
            
        metrics = {
            'total_records': self.total_records,
            'anomaly_count': anomaly_count,
            'anomaly_percentage': round((anomaly_count / self.total_records) * 100, 2) if self.total_records > 0 else 0,
            'critical_anomalies': critical_count,
            'critical_percentage': round((critical_count / anomaly_count) * 100, 2) if anomaly_count > 0 else 0
        }
        
        return metrics
    
    def create_status_work_chart(self):
        """Create chart for Status Work anomalies"""
        if 'Status_Work' not in self.df.columns:
            return None
            
        status_counts = self.df['Status_Work'].value_counts().head(10)
        
        fig = go.Figure(data=[
            go.Bar(
                y=status_counts.index,
                x=status_counts.values,
                orientation='h',
                marker=dict(
                    color=['#e74c3c', '#f39c12', '#e67e22', '#d35400', '#c0392b'] * 2,
                    line=dict(color='rgba(58, 71, 80, 1.0)', width=1)
                ),
                text=status_counts.values,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title={
                'text': 'Anomalies by Status Work Category',
                'x': 0.5,
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            xaxis_title='Count',
            yaxis_title='Status Work Category',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            margin=dict(l=150, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_product_category_chart(self):
        """Create chart for Product Category failures"""
        if 'Product_Name' not in self.df.columns:
            return None
            
        product_counts = self.df['Product_Name'].value_counts().head(10)
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', 
                  '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']
        
        fig = go.Figure(data=[
            go.Pie(
                labels=product_counts.index,
                values=product_counts.values,
                hole=0.3,
                marker=dict(colors=colors),
                textinfo='percent+label',
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title={
                'text': 'Top 10 Product Categories with Anomalies',
                'x': 0.5,
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        
        return fig
    
    def create_anomaly_type_chart(self):
        """Create chart for Anomaly Types"""
        if 'Anomaly_Type' not in self.df.columns:
            return None
            
        anomaly_counts = self.df['Anomaly_Type'].value_counts()
        
        colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6']
        
        fig = go.Figure(data=[
            go.Pie(
                labels=anomaly_counts.index,
                values=anomaly_counts.values,
                hole=0.4,
                marker=dict(colors=colors),
                textinfo='percent+label',
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title={
                'text': 'Distribution by Anomaly Type',
                'x': 0.5,
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        
        return fig
    
    def create_journey_analysis_chart(self):
        """Create chart showing failure points in customer journey"""
        journey_data = []
        
        # Define journey stages and their failure indicators
        journey_stages = {
            'Status Work': 'Status_Work',
            'Subscription': 'Subscription_Status',
            'Activation': 'Activation_Status', 
            'Billing': 'Billing_Status',
            'Vendor Process': 'Vendor_Status'
        }
        
        for stage_name, col_name in journey_stages.items():
            if col_name in self.df.columns:
                # Count failures (adjust these keywords based on your data)
                failure_keywords = ['fail', 'error', 'issue', 'problem', 'inactive', 'cancelled', 'suspended']
                failure_count = 0
                
                for keyword in failure_keywords:
                    failure_count += len(self.df[self.df[col_name].str.contains(keyword, case=False, na=False)])
                
                journey_data.append({
                    'Stage': stage_name,
                    'Failures': failure_count,
                    'Success_Rate': round(((len(self.df) - failure_count) / len(self.df)) * 100, 1) if len(self.df) > 0 else 0
                })
        
        if not journey_data:
            return None
            
        journey_df = pd.DataFrame(journey_data)
        
        fig = go.Figure()
        
        # Add failure bars
        fig.add_trace(go.Bar(
            name='Failures',
            x=journey_df['Stage'],
            y=journey_df['Failures'],
            marker_color='#e74c3c',
            text=journey_df['Failures'],
            textposition='auto',
        ))
        
        fig.update_layout(
            title={
                'text': 'Failure Distribution Across Customer Journey',
                'x': 0.5,
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            xaxis_title='Journey Stage',
            yaxis_title='Number of Failures',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            showlegend=False
        )
        
        return fig
    
    def create_trend_analysis(self):
        """Create trend analysis if date column exists"""
        date_columns = ['Created_Date', 'created_date', 'date', 'Date']
        date_col = None
        
        for col in date_columns:
            if col in self.df.columns:
                date_col = col
                break
        
        if not date_col:
            return None
            
        try:
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
            self.df = self.df.dropna(subset=[date_col])
            
            # Group by date and count anomalies
            daily_counts = self.df.groupby(self.df[date_col].dt.date).size().reset_index()
            daily_counts.columns = ['Date', 'Count']
            daily_counts = daily_counts.sort_values('Date').tail(30)  # Last 30 days
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=daily_counts['Date'],
                y=daily_counts['Count'],
                mode='lines+markers',
                name='Daily Anomalies',
                line=dict(color='#3498db', width=3),
                marker=dict(size=8, color='#e74c3c')
            ))
            
            fig.update_layout(
                title={
                    'text': 'Daily Anomaly Trends (Last 30 Days)',
                    'x': 0.5,
                    'font': {'size': 18, 'color': '#2c3e50'}
                },
                xaxis_title='Date',
                yaxis_title='Anomaly Count',
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating trend analysis: {e}")
            return None
    
    def create_summary_table_html(self):
        """Create summary statistics as HTML table"""
        summary_stats = []
        metrics = self.calculate_summary_metrics()
        
        # Overall statistics
        summary_stats.append({
            'Metric': 'Total Records Processed',
            'Value': f"{metrics['total_records']:,}",
            'Percentage': '100.0%'
        })
        
        summary_stats.append({
            'Metric': 'Total Anomalies Found',
            'Value': f"{metrics['anomaly_count']:,}",
            'Percentage': f"{metrics['anomaly_percentage']}%"
        })
        
        summary_stats.append({
            'Metric': 'Critical Anomalies',
            'Value': f"{metrics['critical_anomalies']:,}",
            'Percentage': f"{metrics['critical_percentage']}%"
        })
        
        # Status-wise breakdown
        status_columns = {
            'Status_Work': 'Status Work Issues',
            'Subscription_Status': 'Subscription Issues', 
            'Activation_Status': 'Activation Issues',
            'Billing_Status': 'Billing Issues',
            'Vendor_Status': 'Vendor Issues'
        }
        
        for col_name, display_name in status_columns.items():
            if col_name in self.df.columns:
                # Count issues in this column
                issue_keywords = ['fail', 'error', 'issue', 'problem', 'inactive', 'cancelled', 'suspended']
                issue_count = 0
                for keyword in issue_keywords:
                    issue_count += len(self.df[self.df[col_name].str.contains(keyword, case=False, na=False)])
                
                percentage = round((issue_count / metrics['anomaly_count']) * 100, 1) if metrics['anomaly_count'] > 0 else 0
                
                summary_stats.append({
                    'Metric': display_name,
                    'Value': f"{issue_count:,}",
                    'Percentage': f"{percentage}%"
                })
        
        # Generate HTML table
        table_html = """
        <table style="width: 100%; border-collapse: collapse; margin: 20px 0; font-family: Arial, sans-serif;">
            <thead>
                <tr style="background-color: #3498db; color: white;">
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Metric</th>
                    <th style="padding: 12px; text-align: right; border: 1px solid #ddd;">Value</th>
                    <th style="padding: 12px; text-align: right; border: 1px solid #ddd;">Percentage</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for i, row in enumerate(summary_stats):
            bg_color = "#f8f9fa" if i % 2 == 0 else "#ffffff"
            table_html += f"""
                <tr style="background-color: {bg_color};">
                    <td style="padding: 10px; border: 1px solid #ddd; font-weight: {'bold' if 'Total' in row['Metric'] else 'normal'};">{row['Metric']}</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: right; font-weight: {'bold' if 'Total' in row['Metric'] else 'normal'};">{row['Value']}</td>
                    <td style="padding: 10px; border: 1px solid #ddd; text-align: right; font-weight: {'bold' if 'Total' in row['Metric'] else 'normal'};">{row['Percentage']}</td>
                </tr>
            """
        
        table_html += """
            </tbody>
        </table>
        """
        
        return table_html
    
    def generate_html_report(self, output_file=None):
        """Generate complete HTML report"""
        if output_file is None:
            output_file = f'anomaly_report_{self.report_date}.html'
        
        # Preprocess data
        self.preprocess_data()
        
        # Calculate metrics
        metrics = self.calculate_summary_metrics()
        
        # Create all charts
        charts = {
            'status_work': self.create_status_work_chart(),
            'product': self.create_product_category_chart(), 
            'anomaly_type': self.create_anomaly_type_chart(),
            'journey': self.create_journey_analysis_chart(),
            'trend': self.create_trend_analysis()
        }
        
        # Generate chart HTML with embedded Plotly
        chart_htmls = {}
        for chart_name, chart in charts.items():
            if chart:
                chart_htmls[chart_name] = pio.to_html(
                    chart, 
                    include_plotlyjs='inline',
                    div_id=f"{chart_name}_chart",
                    config={'displayModeBar': False}
                )
            else:
                chart_htmls[chart_name] = f'<div style="text-align: center; padding: 50px; color: #7f8c8d; font-style: italic;">No data available for {chart_name.replace("_", " ").title()} analysis</div>'
        
        # Get summary table HTML
        summary_table = self.create_summary_table_html()
        
        # Generate complete HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Anomaly Report - {self.report_date}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
            color: #2c3e50;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 3px solid #3498db;
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin: 0 0 10px 0;
            font-weight: 300;
        }}
        
        .header h2 {{
            color: #7f8c8d;
            font-size: 1.2em;
            margin: 0;
            font-weight: 400;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        
        .metric-card.total {{ background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); }}
        .metric-card.anomaly {{ background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }}
        .metric-card.percentage {{ background: linear-gradient(135deg, #f39c12 0%, #d35400 100%); }}
        .metric-card.critical {{ background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%); }}
        
        .metric-number {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .metric-label {{
            font-size: 1.1em;
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        
        .section {{
            margin: 40px 0;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ecf0f1;
        }}
        
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        
        .full-width {{
            grid-column: 1 / -1;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>📊 Daily Anomaly Summary Report</h1>
            <h2>Report Date: {self.report_date}</h2>
        </div>
        
        <!-- Key Metrics -->
        <div class="metrics-grid">
            <div class="metric-card total">
                <div class="metric-number">{metrics['total_records']:,}</div>
                <div class="metric-label">Total Records</div>
            </div>
            <div class="metric-card anomaly">
                <div class="metric-number">{metrics['anomaly_count']:,}</div>
                <div class="metric-label">Anomalies Found</div>
            </div>
            <div class="metric-card percentage">
                <div class="metric-number">{metrics['anomaly_percentage']}%</div>
                <div class="metric-label">Anomaly Rate</div>
            </div>
            <div class="metric-card critical">
                <div class="metric-number">{metrics['critical_anomalies']:,}</div>
                <div class="metric-label">Critical Issues</div>
            </div>
        </div>
        
        <!-- Summary Statistics Table -->
        <div class="section">
            <div class="section-title">📈 Summary Statistics</div>
            {summary_table}
        </div>
        
        <!-- Charts Section -->
        <div class="section">
            <div class="section-title">📊 Analysis Charts</div>
            
            <div class="charts-grid">
                <div class="chart-container">
                    {chart_htmls['status_work']}
                </div>
                <div class="chart-container">
                    {chart_htmls['product']}
                </div>
            </div>
            
            <div class="charts-grid">
                <div class="chart-container">
                    {chart_htmls['anomaly_type']}
                </div>
                <div class="chart-container">
                    {chart_htmls['journey']}
                </div>
            </div>
            
            <div class="charts-grid">
                <div class="chart-container full-width">
                    {chart_htmls['trend']}
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>
                🕐 Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} <br>
                📊 Data source: {len(self.df):,} anomaly records out of {self.total_records:,} total records
            </p>
        </div>
    </div>
</body>
</html>
        """
        
        # Save HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML report generated successfully: {output_file}")
        print(f"📊 Report contains {metrics['anomaly_count']:,} anomaly records")
        print(f"📧 File ready for email attachment: {round(len(html_content)/1024/1024, 2)} MB")
        
        return output_file

# Database integration functions
def load_data_from_database(connection_string, query=None):
    """
    Load anomaly data from database
    """
    import sqlalchemy as sa
    
    if query is None:
        query = """
        SELECT 
            status_work,
            subscription_status, 
            activation_status,
            billing_status,
            vendor_status,
            product_name,
            anomaly_type,
            created_date
        FROM anomaly_table 
        WHERE DATE(created_date) >= CURRENT_DATE - INTERVAL '1 day'
        """
    
    try:
        engine = sa.create_engine(connection_string)
        df = pd.read_sql(query, engine)
        print(f"✅ Loaded {len(df)} records from database")
        return df
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None

def schedule_daily_report(data_source, total_records=100000, output_dir="./reports/"):
    """
    Main function to generate daily report - can be called by cron job
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    if isinstance(data_source, str) and data_source.startswith(('postgresql://', 'mysql://', 'sqlite://')):
        # Database connection string
        df = load_data_from_database(data_source)
        if df is None:
            return None
    else:
        # CSV file or DataFrame
        df = data_source
    
    # Generate report
    report_generator = OfflineAnomalyReportGenerator(df, total_records)
    
    # Generate HTML file
    output_file = os.path.join(output_dir, f'daily_anomaly_report_{date.today()}.html')
    report_file = report_generator.generate_html_report(output_file)
    
    return report_file

def create_sample_data(num_records=4000):
    """Create sample anomaly data for testing"""
    import random
    
    sample_data = {
        'status_work': random.choices(['Failed', 'Error', 'Timeout', 'Pending', 'Incomplete'], k=num_records),
        'subscription_status': random.choices(['Active', 'Inactive', 'Suspended', 'Cancelled', 'Expired'], k=num_records),
        'activation_status': random.choices(['Activated', 'Failed', 'Pending', 'Error', 'Timeout'], k=num_records),
        'billing_status': random.choices(['Paid', 'Failed', 'Pending', 'Declined', 'Expired'], k=num_records),
        'vendor_status': random.choices(['Active', 'Inactive', 'Error', 'Unavailable', 'Timeout'], k=num_records),
        'product_name': random.choices(['Premium Service', 'Basic Plan', 'Enterprise', 'Starter', 'Professional', 'Ultimate', 'Standard', 'Lite'], k=num_records),
        'anomaly_type': random.choices(['Direct Anomaly', 'Pending with Billing', 'Pending with Activation', 'Critical Issue', 'System Error'], k=num_records),
        'created_date': pd.date_range(start='2024-07-01', end='2024-08-10', periods=num_records)
    }
    
    return pd.DataFrame(sample_data)

# Example usage
if __name__ == "__main__":
    print("🚀 Starting Anomaly Report Generation...")
    
    # Create sample data (replace with your actual data loading)
    df = create_sample_data(4000)
    print(f"📊 Sample data created: {len(df)} records")
    
    # Generate report
    report_file = schedule_daily_report(
        data_source=df,
        total_records=100000,
        output_dir="./reports/"
    )
    
    if report_file:
        print(f"✅ Report generation completed!")
        print(f"📧 Ready for email attachment: {report_file}")
    else:
        print("❌ Report generation failed!")

    # Example for database integration:
    # connection_string = "postgresql://username:password@host:port/database"
    # report_file = schedule_daily_report(connection_string, total_records=100000)
