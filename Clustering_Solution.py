import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class AnomalyPatternAnalyzer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.cluster_model = None
        self.patterns = {}
        self.feature_columns = []
        
    def preprocess_data(self, df):
        """
        Preprocess categorical data for clustering
        """
        processed_df = df.copy()
        
        # Identify categorical columns (excluding ID columns)
        categorical_cols = [col for col in df.columns if col not in ['CategoryID']]
        self.feature_columns = categorical_cols
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            processed_df[col] = self.label_encoders[col].fit_transform(processed_df[col].astype(str))
        
        return processed_df[categorical_cols]
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """
        Find optimal number of clusters using silhouette score
        """
        silhouette_scores = []
        K_range = range(2, min(max_clusters + 1, len(X)))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        
        optimal_k = K_range[np.argmax(silhouette_scores)]
        return optimal_k, silhouette_scores
    
    def perform_clustering(self, X, method='kmeans', n_clusters=None):
        """
        Perform clustering using specified method
        """
        if method == 'kmeans':
            if n_clusters is None:
                n_clusters, _ = self.find_optimal_clusters(X)
            
            self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = self.cluster_model.fit_predict(X)
            
        elif method == 'dbscan':
            self.cluster_model = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = self.cluster_model.fit_predict(X)
            
        return cluster_labels
    
    def analyze_cluster_patterns(self, df, cluster_labels):
        """
        Analyze patterns within each cluster
        """
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = cluster_labels
        
        cluster_patterns = {}
        
        for cluster_id in df_with_clusters['Cluster'].unique():
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
                
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
            
            # Analyze patterns for this cluster
            patterns = {}
            
            # System-wise anomaly patterns
            systems = ['CJCM', 'CASSANDRA', 'Product', 'Billing']
            for system in systems:
                system_cols = [col for col in df.columns if system in col]
                if system_cols:
                    system_patterns = {}
                    for col in system_cols:
                        if col in cluster_data.columns:
                            value_counts = cluster_data[col].value_counts()
                            system_patterns[col] = value_counts.to_dict()
                    patterns[system] = system_patterns
            
            # Calculate cluster statistics
            cluster_size = len(cluster_data)
            cluster_percentage = (cluster_size / len(df)) * 100
            
            cluster_patterns[cluster_id] = {
                'size': cluster_size,
                'percentage': cluster_percentage,
                'patterns': patterns,
                'sample_records': cluster_data.head(3).to_dict('records')
            }
        
        return cluster_patterns
    
    def find_frequent_patterns(self, df, min_support=0.05):
        """
        Find frequent patterns using association rule mining approach
        """
        # Convert to binary matrix for pattern mining
        binary_matrix = pd.get_dummies(df[self.feature_columns])
        
        # Find frequent itemsets
        frequent_patterns = {}
        n_transactions = len(binary_matrix)
        
        # Single items
        for col in binary_matrix.columns:
            support = binary_matrix[col].sum() / n_transactions
            if support >= min_support:
                frequent_patterns[col] = {
                    'support': support,
                    'count': binary_matrix[col].sum(),
                    'items': [col]
                }
        
        # Pairs of items
        for col1, col2 in combinations(binary_matrix.columns, 2):
            support = (binary_matrix[col1] & binary_matrix[col2]).sum() / n_transactions
            if support >= min_support:
                pattern_name = f"{col1} & {col2}"
                frequent_patterns[pattern_name] = {
                    'support': support,
                    'count': (binary_matrix[col1] & binary_matrix[col2]).sum(),
                    'items': [col1, col2]
                }
        
        # Sort by support
        sorted_patterns = dict(sorted(frequent_patterns.items(), 
                                    key=lambda x: x[1]['support'], 
                                    reverse=True))
        
        return sorted_patterns
    
    def generate_insights(self, cluster_patterns, frequent_patterns):
        """
        Generate actionable insights from patterns
        """
        insights = {
            'top_clusters': [],
            'system_specific_issues': defaultdict(list),
            'cross_system_issues': [],
            'frequent_combinations': []
        }
        
        # Top clusters by size
        sorted_clusters = sorted(cluster_patterns.items(), 
                               key=lambda x: x[1]['size'], 
                               reverse=True)
        
        for cluster_id, cluster_info in sorted_clusters[:5]:
            insights['top_clusters'].append({
                'cluster_id': cluster_id,
                'size': cluster_info['size'],
                'percentage': cluster_info['percentage'],
                'description': self._describe_cluster(cluster_info['patterns'])
            })
        
        # System-specific issues
        for cluster_id, cluster_info in cluster_patterns.items():
            for system, system_patterns in cluster_info['patterns'].items():
                if any('Anomaly_Yes' in str(pattern) for pattern in system_patterns.values()):
                    insights['system_specific_issues'][system].append({
                        'cluster_id': cluster_id,
                        'size': cluster_info['size'],
                        'patterns': system_patterns
                    })
        
        # Frequent patterns
        for pattern_name, pattern_info in list(frequent_patterns.items())[:10]:
            insights['frequent_combinations'].append({
                'pattern': pattern_name,
                'support': pattern_info['support'],
                'count': pattern_info['count'],
                'impact': 'High' if pattern_info['support'] > 0.2 else 'Medium' if pattern_info['support'] > 0.1 else 'Low'
            })
        
        return insights
    
    def _describe_cluster(self, patterns):
        """
        Generate human-readable description of cluster patterns
        """
        descriptions = []
        
        for system, system_patterns in patterns.items():
            anomaly_found = False
            system_desc = []
            
            for attr, values in system_patterns.items():
                if 'Anomaly' in attr and any('Yes' in str(k) for k in values.keys()):
                    anomaly_found = True
                    # Find most common non-anomaly attribute
                    other_attrs = {k: v for k, v in system_patterns.items() if 'Anomaly' not in k}
                    if other_attrs:
                        most_common_attr = max(other_attrs.items(), key=lambda x: max(x[1].values()) if x[1] else 0)
                        system_desc.append(f"{most_common_attr[0]}: {max(most_common_attr[1].keys(), key=most_common_attr[1].get)}")
            
            if anomaly_found:
                descriptions.append(f"{system} issues: {', '.join(system_desc)}")
        
        return "; ".join(descriptions) if descriptions else "No clear pattern identified"
    
    def visualize_patterns(self, df, cluster_labels, top_n=10):
        """
        Create visualizations for pattern analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Cluster distribution
        cluster_counts = pd.Series(cluster_labels).value_counts()
        axes[0, 0].bar(range(len(cluster_counts)), cluster_counts.values)
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of Anomalies')
        
        # 2. System-wise anomaly distribution
        systems = ['CJCM', 'CASSANDRA', 'Product', 'Billing']
        system_anomaly_counts = {}
        
        for system in systems:
            anomaly_cols = [col for col in df.columns if system in col and 'Anomaly' in col]
            if anomaly_cols:
                system_anomaly_counts[system] = df[anomaly_cols[0]].value_counts().get('Yes', 0)
        
        if system_anomaly_counts:
            axes[0, 1].bar(system_anomaly_counts.keys(), system_anomaly_counts.values())
            axes[0, 1].set_title('System-wise Anomaly Count')
            axes[0, 1].set_ylabel('Anomaly Count')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Category distribution
        if 'Category' in df.columns:
            category_counts = df['Category'].value_counts().head(top_n)
            axes[1, 0].bar(range(len(category_counts)), category_counts.values)
            axes[1, 0].set_title('Top Categories with Anomalies')
            axes[1, 0].set_xlabel('Category (top 10)')
            axes[1, 0].set_ylabel('Anomaly Count')
            axes[1, 0].set_xticks(range(len(category_counts)))
            axes[1, 0].set_xticklabels(category_counts.index, rotation=45, ha='right')
        
        # 4. Cluster composition by category
        if 'Category' in df.columns:
            df_viz = df.copy()
            df_viz['Cluster'] = cluster_labels
            
            # Create heatmap of cluster vs category
            cluster_category = pd.crosstab(df_viz['Cluster'], df_viz['Category'])
            sns.heatmap(cluster_category, annot=True, fmt='d', ax=axes[1, 1], cmap='Blues')
            axes[1, 1].set_title('Cluster vs Category Heatmap')
            axes[1, 1].set_xlabel('Category')
            axes[1, 1].set_ylabel('Cluster')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def analyze_anomalies(self, df, clustering_method='kmeans', min_support=0.05):
        """
        Main method to analyze anomaly patterns
        """
        print("Starting anomaly pattern analysis...")
        
        # Preprocess data
        X = self.preprocess_data(df)
        print(f"Preprocessed {len(X)} records with {len(X.columns)} features")
        
        # Perform clustering
        cluster_labels = self.perform_clustering(X, method=clustering_method)
        print(f"Identified {len(set(cluster_labels))} clusters")
        
        # Analyze cluster patterns
        cluster_patterns = self.analyze_cluster_patterns(df, cluster_labels)
        
        # Find frequent patterns
        frequent_patterns = self.find_frequent_patterns(df, min_support)
        
        # Generate insights
        insights = self.generate_insights(cluster_patterns, frequent_patterns)
        
        # Visualize patterns
        fig = self.visualize_patterns(df, cluster_labels)
        
        return {
            'cluster_patterns': cluster_patterns,
            'frequent_patterns': frequent_patterns,
            'insights': insights,
            'cluster_labels': cluster_labels,
            'visualization': fig
        }
    
    def print_insights(self, insights):
        """
        Print formatted insights
        """
        print("\n" + "="*80)
        print("ANOMALY PATTERN ANALYSIS INSIGHTS")
        print("="*80)
        
        print("\n🔍 TOP RECURRING PATTERNS:")
        print("-" * 50)
        for i, cluster in enumerate(insights['top_clusters'], 1):
            print(f"{i}. Cluster {cluster['cluster_id']}: {cluster['size']} anomalies ({cluster['percentage']:.1f}%)")
            print(f"   Description: {cluster['description']}")
            print()
        
        print("\n🏭 SYSTEM-SPECIFIC ISSUES:")
        print("-" * 50)
        for system, issues in insights['system_specific_issues'].items():
            print(f"{system}:")
            for issue in issues[:3]:  # Top 3 issues per system
                print(f"  - Cluster {issue['cluster_id']}: {issue['size']} occurrences")
            print()
        
        print("\n🔗 FREQUENT COMBINATIONS:")
        print("-" * 50)
        for pattern in insights['frequent_combinations'][:5]:
            print(f"Pattern: {pattern['pattern']}")
            print(f"Support: {pattern['support']:.3f} ({pattern['count']} occurrences) - {pattern['impact']} Impact")
            print()

# Example usage and data preparation
def create_sample_data():
    """
    Create sample data based on the provided format
    """
    np.random.seed(42)
    
    categories = ['Apple', 'Samsung', 'Google', 'Microsoft', 'Amazon']
    
    # CJCM attributes
    cjcm_status = ['Falout', 'Success', 'Pending', 'Error']
    cjcm_action = ['Activation', 'Deactivation', 'Update', 'Sync']
    
    # Cassandra attributes
    cassandra_entitlement = ['Pending', 'Active', 'Expired', 'Suspended']
    cassandra_subscription = ['Subscribed', 'Unsubscribed', 'Trial', 'Cancelled']
    cassandra_registration = ['None', 'Pending', 'Complete', 'Failed']
    
    # Product attributes
    product_status = ['None', 'Active', 'Inactive', 'Maintenance']
    
    # Billing attributes
    billing_status = ['Active', 'Inactive', 'Suspended', 'Pending']
    
    # Generate sample data
    n_samples = 1000
    data = []
    
    for i in range(n_samples):
        # Create correlated anomalies for realistic patterns
        has_major_issue = np.random.choice([True, False], p=[0.3, 0.7])
        
        if has_major_issue:
            # Create correlated anomalies
            cjcm_anomaly = 'Yes' if np.random.random() < 0.8 else 'No'
            cassandra_anomaly = 'Yes' if np.random.random() < 0.7 else 'No'
            product_anomaly = 'Yes' if np.random.random() < 0.6 else 'No'
            billing_anomaly = 'Yes' if np.random.random() < 0.4 else 'No'
        else:
            # Random anomalies
            cjcm_anomaly = 'Yes' if np.random.random() < 0.1 else 'No'
            cassandra_anomaly = 'Yes' if np.random.random() < 0.1 else 'No'
            product_anomaly = 'Yes' if np.random.random() < 0.1 else 'No'
            billing_anomaly = 'Yes' if np.random.random() < 0.05 else 'No'
        
        record = {
            'Category': np.random.choice(categories),
            'CategoryID': np.random.randint(1000, 9999),
            'CJCM_Status': np.random.choice(cjcm_status),
            'CJCM_Action': np.random.choice(cjcm_action),
            'CJCM_Anomaly': cjcm_anomaly,
            'CASSANDRA_Entitlement': np.random.choice(cassandra_entitlement),
            'CASSANDRA_Subscription': np.random.choice(cassandra_subscription),
            'CASSANDRA_Registration': np.random.choice(cassandra_registration),
            'CASSANDRA_Anomaly': cassandra_anomaly,
            'Product_Status': np.random.choice(product_status),
            'Product_Anomaly': product_anomaly,
            'Billing_Status': np.random.choice(billing_status),
            'Billing_Anomaly': billing_anomaly
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

# Main execution
if __name__ == "__main__":
    # Create sample data
    df = create_sample_data()
    print(f"Created sample dataset with {len(df)} records")
    print("\nSample records:")
    print(df.head())
    
    # Initialize analyzer
    analyzer = AnomalyPatternAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_anomalies(df, clustering_method='kmeans', min_support=0.05)
    
    # Print insights
    analyzer.print_insights(results['insights'])
    
    # Additional analysis for production use
    print("\n📊 PRODUCTION RECOMMENDATIONS:")
    print("-" * 50)
    print("1. Schedule this analysis to run daily/hourly based on your data volume")
    print("2. Set up alerts for clusters that exceed threshold sizes")
    print("3. Create dashboards to monitor pattern evolution over time")
    print("4. Implement automated issue assignment based on cluster patterns")
    print("5. Use frequent patterns for proactive monitoring rules")
