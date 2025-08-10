import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
from datetime import datetime, date
import numpy as np

class AnomalyReportGenerator:
    def __init__(self, data_source):
        """
        Initialize the report generator
        data_source: pandas DataFrame or CSV file path
        """
        if isinstance(data_source, str):
            self.df = pd.read_csv(data_source)
        else:
            self.df = data_source.copy()
        
        self.report_date = datetime.now().strftime("%Y-%m-%d")
        self.total_records = len(self.df)
        
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
            'anomaly_type': 'Anomaly_Type'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in self.df.columns:
                self.df.rename(columns={old_name: new_name}, inplace=True)
    
    def calculate_summary_metrics(self):
        """Calculate key metrics for the summary report"""
        metrics = {
            'total_records': self.total_records,
            'anomaly_count': len(self.df),
            'anomaly_percentage': round((len(self.df) / self.total_records) * 100, 2),
            'critical_anomalies': len(self.df[self.df['Anomaly_Type'].str.contains('Critical|High', case=False, na=False)]),
        }
        
        return metrics
    
    def create_status_work_chart(self):
        """Create chart for Status Work anomalies"""
        if 'Status_Work' in self.df.columns:
            status_counts = self.df['Status_Work'].value_counts()
            
            fig = px.bar(
                x=status_counts.values,
                y=status_counts.index,
                orientation='h',
                title='Anomalies by Status Work Category',
                labels={'x': 'Count', 'y': 'Status Work'},
                color=status_counts.values,
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                title_font_size=16,
                font_size=12
            )
            
            return fig
        return None
    
    def create_product_category_chart(self):
        """Create chart for Product Category failures"""
        if 'Product_Name' in self.df.columns:
            product_counts = self.df['Product_Name'].value_counts().head(10)
            
            fig = px.pie(
                values=product_counts.values,
                names=product_counts.index,
                title='Top 10 Product Categories with Anomalies'
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400, title_font_size=16)
            
            return fig
        return None
    
    def create_anomaly_type_chart(self):
        """Create chart for Anomaly Types"""
        if 'Anomaly_Type' in self.df.columns:
            anomaly_counts = self.df['Anomaly_Type'].value_counts()
            
            fig = px.donut(
                values=anomaly_counts.values,
                names=anomaly_counts.index,
                title='Distribution by Anomaly Type',
                hole=0.4
            )
            
            fig.update_layout(height=400, title_font_size=16)
            return fig
        return None
    
    def create_journey_analysis_chart(self):
        """Create chart showing failure points in customer journey"""
        # Create a journey analysis based on different status columns
        journey_data = []
        
        status_columns = ['Status_Work', 'Subscription_Status', 'Activation_Status', 'Billing_Status', 'Vendor_Status']
        
        for col in status_columns:
            if col in self.df.columns:
                failure_count = len(self.df[self.df[col].str.contains('fail|error|issue', case=False, na=False)])
                journey_data.append({'Stage': col.replace('_', ' '), 'Failures': failure_count})
        
        if journey_data:
            journey_df = pd.DataFrame(journey_data)
            
            fig = px.funnel(
                journey_df,
                x='Failures',
                y='Stage',
                title='Failure Distribution Across Customer Journey'
            )
            
            fig.update_layout(height=400, title_font_size=16)
            return fig
        return None
    
    def create_trend_analysis(self):
        """Create trend analysis if date column exists"""
        # This assumes you have a date column for trend analysis
        if 'created_date' in self.df.columns or 'date' in self.df.columns:
            date_col = 'created_date' if 'created_date' in self.df.columns else 'date'
            
            try:
                self.df[date_col] = pd.to_datetime(self.df[date_col])
                daily_counts = self.df.groupby(self.df[date_col].dt.date).size()
                
                fig = px.line(
                    x=daily_counts.index,
                    y=daily_counts.values,
                    title='Daily Anomaly Trends (Last 30 Days)'
                )
                
                fig.update_layout(
                    height=300,
                    xaxis_title='Date',
                    yaxis_title='Anomaly Count',
                    title_font_size=16
                )
                
                return fig
            except:
                pass
        
        return None
    
    def create_summary_table(self):
        """Create summary statistics table"""
        summary_stats = []
        
        # Overall statistics
        metrics = self.calculate_summary_metrics()
        summary_stats.append({
            'Metric': 'Total Records Processed',
            'Value': f"{metrics['total_records']:,}",
            'Percentage': '100%'
        })
        
        summary_stats.append({
            'Metric': 'Total Anomalies Found',
            'Value': f"{metrics['anomaly_count']:,}",
            'Percentage': f"{metrics['anomaly_percentage']}%"
        })
        
        # Status-wise breakdown
        for col in ['Status_Work', 'Subscription_Status', 'Activation_Status', 'Billing_Status']:
            if col in self.df.columns:
                col_anomalies = self.df[col].value_counts().sum()
                percentage = round((col_anomalies / metrics['anomaly_count']) * 100, 1) if metrics['anomaly_count'] > 0 else 0
                
                summary_stats.append({
                    'Metric': f'{col.replace("_", " ")} Anomalies',
                    'Value': f"{col_anomalies:,}",
                    'Percentage': f"{percentage}%"
                })
        
        return pd.DataFrame(summary_stats)
    
    def create_dash_app(self):
        """Create the Dash application"""
        app = dash.Dash(__name__)
        
        # Preprocess data
        self.preprocess_data()
        
        # Calculate metrics
        metrics = self.calculate_summary_metrics()
        summary_table_df = self.create_summary_table()
        
        # Create charts
        status_work_chart = self.create_status_work_chart()
        product_chart = self.create_product_category_chart()
        anomaly_type_chart = self.create_anomaly_type_chart()
        journey_chart = self.create_journey_analysis_chart()
        trend_chart = self.create_trend_analysis()
        
        # Define the layout
        app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Daily Anomaly Summary Report", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                html.H3(f"Report Date: {self.report_date}", 
                       style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'}),
                
                # Key Metrics Cards
                html.Div([
                    html.Div([
                        html.H2(f"{metrics['total_records']:,}", style={'color': '#3498db', 'margin': '0'}),
                        html.P("Total Records", style={'color': '#7f8c8d', 'margin': '0'})
                    ], className='metric-card', style={
                        'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px',
                        'textAlign': 'center', 'margin': '10px', 'flex': '1'
                    }),
                    
                    html.Div([
                        html.H2(f"{metrics['anomaly_count']:,}", style={'color': '#e74c3c', 'margin': '0'}),
                        html.P("Anomalies Found", style={'color': '#7f8c8d', 'margin': '0'})
                    ], className='metric-card', style={
                        'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px',
                        'textAlign': 'center', 'margin': '10px', 'flex': '1'
                    }),
                    
                    html.Div([
                        html.H2(f"{metrics['anomaly_percentage']}%", style={'color': '#f39c12', 'margin': '0'}),
                        html.P("Anomaly Rate", style={'color': '#7f8c8d', 'margin': '0'})
                    ], className='metric-card', style={
                        'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px',
                        'textAlign': 'center', 'margin': '10px', 'flex': '1'
                    }),
                    
                    html.Div([
                        html.H2(f"{metrics.get('critical_anomalies', 0):,}", style={'color': '#9b59b6', 'margin': '0'}),
                        html.P("Critical Issues", style={'color': '#7f8c8d', 'margin': '0'})
                    ], className='metric-card', style={
                        'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px',
                        'textAlign': 'center', 'margin': '10px', 'flex': '1'
                    })
                ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'})
            ], style={'margin': '20px'}),
            
            # Summary Table
            html.Div([
                html.H3("Summary Statistics", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                dash_table.DataTable(
                    data=summary_table_df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in summary_table_df.columns],
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
                    style_data={'backgroundColor': '#ecf0f1'},
                )
            ], style={'margin': '20px'}),
            
            # Charts Section
            html.Div([
                # Row 1: Status Work and Product Categories
                html.Div([
                    html.Div([
                        dcc.Graph(figure=status_work_chart) if status_work_chart else html.P("Status Work data not available")
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(figure=product_chart) if product_chart else html.P("Product category data not available")
                    ], style={'width': '50%', 'display': 'inline-block'})
                ]),
                
                # Row 2: Anomaly Types and Journey Analysis
                html.Div([
                    html.Div([
                        dcc.Graph(figure=anomaly_type_chart) if anomaly_type_chart else html.P("Anomaly type data not available")
                    ], style={'width': '50%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Graph(figure=journey_chart) if journey_chart else html.P("Journey analysis data not available")
                    ], style={'width': '50%', 'display': 'inline-block'})
                ]),
                
                # Row 3: Trend Analysis (if available)
                html.Div([
                    dcc.Graph(figure=trend_chart) if trend_chart else html.P("Trend analysis not available - requires date column")
                ], style={'width': '100%', 'marginTop': '20px'}) if trend_chart else html.Div()
                
            ], style={'margin': '20px'}),
            
            # Footer
            html.Div([
                html.Hr(),
                html.P(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                      f"Data source: {len(self.df)} anomaly records out of {self.total_records} total records",
                      style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '12px'})
            ])
        ])
        
        return app
    
    def generate_static_html(self, output_file='anomaly_report.html'):
        """Generate static HTML file for email attachment"""
        app = self.create_dash_app()
        
        # You can use plotly's offline plotting to create static HTML
        from plotly.offline import plot
        import plotly.io as pio
        
        # Create all charts
        self.preprocess_data()
        metrics = self.calculate_summary_metrics()
        
        charts = {
            'status_work': self.create_status_work_chart(),
            'product': self.create_product_category_chart(),
            'anomaly_type': self.create_anomaly_type_chart(),
            'journey': self.create_journey_analysis_chart(),
            'trend': self.create_trend_analysis()
        }
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Daily Anomaly Report - {self.report_date}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .metrics {{ display: flex; justify-content: space-around; margin-bottom: 30px; }}
                .metric-card {{ background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; }}
                .chart-container {{ margin: 20px 0; }}
                .footer {{ text-align: center; color: #666; font-size: 12px; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Daily Anomaly Summary Report</h1>
                <h3>Report Date: {self.report_date}</h3>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <h2 style="color: #3498db; margin: 0;">{metrics['total_records']:,}</h2>
                    <p style="margin: 0;">Total Records</p>
                </div>
                <div class="metric-card">
                    <h2 style="color: #e74c3c; margin: 0;">{metrics['anomaly_count']:,}</h2>
                    <p style="margin: 0;">Anomalies Found</p>
                </div>
                <div class="metric-card">
                    <h2 style="color: #f39c12; margin: 0;">{metrics['anomaly_percentage']}%</h2>
                    <p style="margin: 0;">Anomaly Rate</p>
                </div>
            </div>
        """
        
        # Add charts to HTML
        for chart_name, chart in charts.items():
            if chart:
                chart_html = pio.to_html(chart, include_plotlyjs='inline', div_id=f"{chart_name}_chart")
                html_content += f'<div class="chart-container">{chart_html}</div>'
        
        html_content += f"""
            <div class="footer">
                <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                Data source: {len(self.df)} anomaly records out of {self.total_records} total records</p>
            </div>
        </body>
        </html>
        """
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Static HTML report generated: {output_file}")
        return output_file

# Example usage and database integration
def load_data_from_db(connection_string, query):
    """
    Load data from database instead of CSV
    """
    import sqlalchemy as sa
    
    engine = sa.create_engine(connection_string)
    df = pd.read_sql(query, engine)
    return df

def schedule_daily_report():
    """
    Function to be called by scheduler (cron job, etc.)
    """
    # Option 1: Load from CSV
    # df = pd.read_csv('anomaly_data.csv')
    
    # Option 2: Load from database
    # connection_string = "postgresql://user:password@host:port/database"
    # query = """
    # SELECT status_work, subscription_status, activation_status, 
    #        billing_status, vendor_status, product_name, anomaly_type,
    #        created_date
    # FROM anomaly_table 
    # WHERE DATE(created_date) = CURRENT_DATE - INTERVAL '1 day'
    # """
    # df = load_data_from_db(connection_string, query)
    
    # For demo purposes, create sample data
    sample_data = {
        'Status_Work': ['Failed', 'Pending', 'Error', 'Failed', 'Pending'] * 800,
        'Subscription_Status': ['Active', 'Inactive', 'Suspended', 'Cancelled', 'Active'] * 800,
        'Activation_Status': ['Activated', 'Failed', 'Pending', 'Activated', 'Failed'] * 800,
        'Billing_Status': ['Paid', 'Failed', 'Pending', 'Paid', 'Failed'] * 800,
        'Vendor_Status': ['Active', 'Inactive', 'Active', 'Active', 'Inactive'] * 800,
        'Product_Name': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'] * 800,
        'Anomaly_Type': ['Direct Anomaly', 'Pending with Billing', 'Pending with Activation', 'Critical Issue', 'Direct Anomaly'] * 800,
        'created_date': pd.date_range(start='2024-01-01', periods=4000, freq='H')
    }
    df = pd.DataFrame(sample_data)
    
    # Generate report
    report_generator = AnomalyReportGenerator(df)
    report_generator.total_records = 100000  # Set total records for percentage calculation
    
    # Generate static HTML for email
    html_file = report_generator.generate_static_html(f'daily_anomaly_report_{date.today()}.html')
    
    # Send email (implement your email logic here)
    # send_email_with_attachment(html_file)
    
    return html_file

# For development/testing - run interactive Dash app
def run_interactive_app(data_source):
    """Run interactive Dash app for development"""
    report_generator = AnomalyReportGenerator(data_source)
    report_generator.total_records = 100000  # Set your actual total
    app = report_generator.create_dash_app()
    app.run_server(debug=True, host='0.0.0.0', port=8050)

if __name__ == "__main__":
    # Example: Generate daily report
    report_file = schedule_daily_report()
    print(f"Daily report generated: {report_file}")
    
    # Example: Run interactive app (comment out for production)
    # run_interactive_app('your_anomaly_data.csv')
