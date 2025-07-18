import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio
from typing import Dict, List, Optional, Any, TypedDict, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Mock LangGraph classes (replace with actual imports in production)
class StateGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
    
    def add_node(self, name, func):
        self.nodes[name] = func
    
    def add_edge(self, from_node, to_node):
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(to_node)
    
    def compile(self):
        return CompiledGraph(self.nodes, self.edges)

class CompiledGraph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
    
    async def ainvoke(self, initial_state):
        current_node = "start"
        state = initial_state.copy()
        
        execution_path = ["start"]
        while current_node != "END" and len(execution_path) < 10:  # Prevent infinite loops
            if current_node in self.nodes:
                state = await self.nodes[current_node](state)
            
            if current_node in self.edges and self.edges[current_node]:
                current_node = self.edges[current_node][0]
                execution_path.append(current_node)
            else:
                break
        
        return state

# Enums and data classes
class InvestigationStatus(Enum):
    PENDING = "pending"
    CLUSTERING = "clustering"
    PATTERN_ANALYSIS = "pattern_analysis"
    AGENT_INVESTIGATION = "agent_investigation"
    COMPLETED = "completed"

class ClusterQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class ClusterInfo:
    """Information about a KMeans cluster"""
    cluster_id: int
    size: int
    centroid: np.ndarray
    silhouette_score: float
    inertia: float
    dominant_features: Dict[str, Any]
    quality: ClusterQuality
    records: pd.DataFrame

@dataclass
class ClusteringResults:
    """Complete clustering analysis results"""
    clusters: List[ClusterInfo]
    optimal_k: int
    overall_silhouette: float
    cluster_labels: np.ndarray
    feature_importance: Dict[str, float]
    preprocessing_info: Dict[str, Any]

@dataclass
class AgentContext:
    """Enhanced context for agent investigation"""
    cluster_info: ClusterInfo
    clustering_results: ClusteringResults
    pattern_signature: str
    affected_systems: List[str]
    severity: str
    investigation_priority: int

class AgentState(TypedDict):
    """State for the hybrid workflow"""
    agent_context: AgentContext
    investigation_status: InvestigationStatus
    clustering_insights: Dict[str, Any]
    pattern_analysis: Dict[str, Any]
    agent_findings: Dict[str, Any]
    recommendations: List[Dict]
    final_report: Optional[Dict]

class HybridKMeansAgenticAnalyzer:
    """
    Hybrid system that combines KMeans clustering with Agentic AI investigation
    """
    
    def __init__(self, anomaly_df: pd.DataFrame):
        self.anomaly_df = anomaly_df
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.clustering_results = None
        self.workflow = self._create_hybrid_workflow()
        
    def _create_hybrid_workflow(self) -> StateGraph:
        """Create the hybrid workflow combining clustering and agents"""
        workflow = StateGraph()
        
        # Workflow nodes
        workflow.add_node("start", self._start_hybrid_analysis)
        workflow.add_node("kmeans_clustering", self._kmeans_clustering_node)
        workflow.add_node("cluster_analysis", self._cluster_analysis_node)
        workflow.add_node("pattern_extraction", self._pattern_extraction_node)
        workflow.add_node("agent_investigation", self._agent_investigation_node)
        workflow.add_node("synthesis", self._synthesis_node)
        workflow.add_node("final_report", self._final_report_node)
        
        # Workflow edges
        workflow.add_edge("start", "kmeans_clustering")
        workflow.add_edge("kmeans_clustering", "cluster_analysis")
        workflow.add_edge("cluster_analysis", "pattern_extraction")
        workflow.add_edge("pattern_extraction", "agent_investigation")
        workflow.add_edge("agent_investigation", "synthesis")
        workflow.add_edge("synthesis", "final_report")
        
        return workflow.compile()
    
    # ================================
    # KMEANS CLUSTERING NODES
    # ================================
    
    async def _start_hybrid_analysis(self, state: AgentState) -> AgentState:
        """Initialize the hybrid analysis"""
        print("🚀 Starting Hybrid KMeans + Agentic AI Analysis")
        print("="*60)
        
        state['investigation_status'] = InvestigationStatus.CLUSTERING
        state['clustering_insights'] = {}
        state['pattern_analysis'] = {}
        state['agent_findings'] = {}
        state['recommendations'] = []
        
        return state
    
    async def _kmeans_clustering_node(self, state: AgentState) -> AgentState:
        """Perform KMeans clustering analysis"""
        print("\n🔬 PHASE 1: KMEANS CLUSTERING ANALYSIS")
        print("-" * 40)
        
        # Preprocess data
        X_processed, feature_names = self._preprocess_for_clustering(self.anomaly_df)
        print(f"✅ Preprocessed {len(X_processed)} records with {len(feature_names)} features")
        
        # Find optimal clusters
        optimal_k, silhouette_scores = self._find_optimal_clusters(X_processed)
        print(f"✅ Optimal number of clusters: {optimal_k}")
        
        # Perform clustering
        self.kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = self.kmeans_model.fit_predict(X_processed)
        
        # Analyze clusters
        clusters = self._analyze_clusters(X_processed, cluster_labels, feature_names)
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(X_processed, cluster_labels, feature_names)
        
        # Create clustering results
        self.clustering_results = ClusteringResults(
            clusters=clusters,
            optimal_k=optimal_k,
            overall_silhouette=silhouette_score(X_processed, cluster_labels),
            cluster_labels=cluster_labels,
            feature_importance=feature_importance,
            preprocessing_info={
                'feature_names': feature_names,
                'n_samples': len(X_processed),
                'n_features': X_processed.shape[1]
            }
        )
        
        state['clustering_insights'] = {
            'clustering_results': self.clustering_results,
            'n_clusters': optimal_k,
            'silhouette_score': self.clustering_results.overall_silhouette,
            'feature_importance': feature_importance
        }
        
        print(f"✅ Identified {len(clusters)} clusters with silhouette score: {self.clustering_results.overall_silhouette:.3f}")
        
        return state
    
    async def _cluster_analysis_node(self, state: AgentState) -> AgentState:
        """Analyze cluster characteristics and quality"""
        print("\n📊 PHASE 2: CLUSTER ANALYSIS")
        print("-" * 40)
        
        clustering_results = state['clustering_insights']['clustering_results']
        cluster_analysis = {}
        
        for cluster in clustering_results.clusters:
            # Analyze cluster composition
            cluster_composition = self._analyze_cluster_composition(cluster)
            
            # Identify cluster patterns
            cluster_patterns = self._identify_cluster_patterns(cluster)
            
            # Assess cluster quality and significance
            cluster_significance = self._assess_cluster_significance(cluster, clustering_results)
            
            cluster_analysis[f"cluster_{cluster.cluster_id}"] = {
                'composition': cluster_composition,
                'patterns': cluster_patterns,
                'significance': cluster_significance,
                'quality': cluster.quality.value,
                'priority_score': self._calculate_priority_score(cluster, cluster_patterns)
            }
            
            print(f"✅ Cluster {cluster.cluster_id}: {cluster.size} records, Quality: {cluster.quality.value}")
        
        state['pattern_analysis']['cluster_analysis'] = cluster_analysis
        
        return state
    
    async def _pattern_extraction_node(self, state: AgentState) -> AgentState:
        """Extract meaningful patterns from clusters for agent investigation"""
        print("\n🔍 PHASE 3: PATTERN EXTRACTION")
        print("-" * 40)
        
        clustering_results = state['clustering_insights']['clustering_results']
        cluster_analysis = state['pattern_analysis']['cluster_analysis']
        
        # Select clusters for agent investigation based on priority
        investigation_candidates = []
        
        for cluster in clustering_results.clusters:
            cluster_id = cluster.cluster_id
            analysis = cluster_analysis[f"cluster_{cluster_id}"]
            
            # Only investigate high-quality, significant clusters
            if (cluster.quality in [ClusterQuality.EXCELLENT, ClusterQuality.GOOD] and
                analysis['priority_score'] > 0.6):
                
                # Extract pattern signature
                pattern_signature = self._extract_pattern_signature(cluster)
                
                # Identify affected systems
                affected_systems = self._identify_affected_systems(cluster)
                
                # Calculate severity
                severity = self._calculate_severity(cluster, analysis)
                
                investigation_candidates.append({
                    'cluster_info': cluster,
                    'pattern_signature': pattern_signature,
                    'affected_systems': affected_systems,
                    'severity': severity,
                    'priority_score': analysis['priority_score']
                })
        
        # Sort by priority and select top candidates
        investigation_candidates.sort(key=lambda x: x['priority_score'], reverse=True)
        top_candidates = investigation_candidates[:5]  # Investigate top 5 patterns
        
        state['pattern_analysis']['investigation_candidates'] = top_candidates
        
        print(f"✅ Extracted {len(investigation_candidates)} patterns, investigating top {len(top_candidates)}")
        
        return state
    
    # ================================
    # AGENTIC INVESTIGATION NODES  
    # ================================
    
    async def _agent_investigation_node(self, state: AgentState) -> AgentState:
        """Run agentic investigation on selected patterns"""
        print("\n🤖 PHASE 4: AGENTIC AI INVESTIGATION")
        print("-" * 40)
        
        investigation_candidates = state['pattern_analysis']['investigation_candidates']
        clustering_results = state['clustering_insights']['clustering_results']
        
        agent_investigations = []
        
        for i, candidate in enumerate(investigation_candidates):
            print(f"\n🔍 Investigating Pattern {i+1}/{len(investigation_candidates)}")
            
            # Create agent context
            agent_context = AgentContext(
                cluster_info=candidate['cluster_info'],
                clustering_results=clustering_results,
                pattern_signature=candidate['pattern_signature'],
                affected_systems=candidate['affected_systems'],
                severity=candidate['severity'],
                investigation_priority=i+1
            )
            
            # Run agent investigation
            investigation_result = await self._run_agent_investigation(agent_context)
            agent_investigations.append(investigation_result)
        
        state['agent_findings']['investigations'] = agent_investigations
        state['investigation_status'] = InvestigationStatus.AGENT_INVESTIGATION
        
        return state
    
    async def _run_agent_investigation(self, context: AgentContext) -> Dict:
        """Run detailed agent investigation on a specific cluster pattern"""
        cluster = context.cluster_info
        
        # Agent-based analysis
        investigation = {
            'cluster_id': cluster.cluster_id,
            'investigation_priority': context.investigation_priority,
            'cluster_insights': await self._agent_cluster_analysis(context),
            'system_correlation': await self._agent_system_correlation(context),
            'temporal_analysis': await self._agent_temporal_analysis(context),
            'hypothesis_generation': await self._agent_hypothesis_generation(context),
            'recommendations': await self._agent_recommendations(context)
        }
        
        print(f"   ✅ Completed investigation for Cluster {cluster.cluster_id}")
        
        return investigation
    
    async def _agent_cluster_analysis(self, context: AgentContext) -> Dict:
        """Agent analyzes cluster characteristics using ML insights"""
        cluster = context.cluster_info
        
        # Analyze feature importance within cluster
        cluster_features = self._analyze_cluster_features(cluster, context.clustering_results)
        
        # Analyze cluster separability
        separability = self._analyze_cluster_separability(cluster, context.clustering_results)
        
        return {
            'cluster_size': cluster.size,
            'cluster_quality': cluster.quality.value,
            'dominant_features': cluster.dominant_features,
            'feature_analysis': cluster_features,
            'separability': separability,
            'relative_size': cluster.size / len(context.clustering_results.clusters)
        }
    
    async def _agent_system_correlation(self, context: AgentContext) -> Dict:
        """Agent analyzes system correlations using clustering insights"""
        cluster = context.cluster_info
        records = cluster.records
        
        # System anomaly correlation matrix
        systems = ['CJCM', 'CASSANDRA', 'Product', 'Billing']
        correlation_matrix = {}
        
        for sys1 in systems:
            for sys2 in systems:
                if sys1 != sys2:
                    corr = self._calculate_system_correlation(records, sys1, sys2)
                    correlation_matrix[f"{sys1}_{sys2}"] = corr
        
        # Identify cascade patterns
        cascade_analysis = self._detect_cascade_patterns(records, systems)
        
        # Most critical system
        critical_system = self._identify_critical_system(records, systems)
        
        return {
            'correlation_matrix': correlation_matrix,
            'cascade_analysis': cascade_analysis,
            'critical_system': critical_system,
            'affected_systems': context.affected_systems
        }
    
    async def _agent_temporal_analysis(self, context: AgentContext) -> Dict:
        """Agent analyzes temporal patterns"""
        cluster = context.cluster_info
        
        # Simulate temporal analysis (in production, use actual timestamps)
        temporal_insights = {
            'occurrence_frequency': 'High' if cluster.size > 20 else 'Medium' if cluster.size > 10 else 'Low',
            'pattern_stability': 'Stable' if cluster.quality in [ClusterQuality.EXCELLENT, ClusterQuality.GOOD] else 'Unstable',
            'trend_direction': 'Increasing' if cluster.size > 15 else 'Stable'
        }
        
        return temporal_insights
    
    async def _agent_hypothesis_generation(self, context: AgentContext) -> List[Dict]:
        """Agent generates hypotheses based on clustering and correlation analysis"""
        cluster = context.cluster_info
        
        hypotheses = []
        
        # Hypothesis 1: Based on cluster quality and size
        if cluster.quality == ClusterQuality.EXCELLENT and cluster.size > 20:
            hypotheses.append({
                'type': 'SYSTEMATIC_ISSUE',
                'description': f'Well-defined cluster of {cluster.size} anomalies suggests systematic issue',
                'confidence': 0.9,
                'evidence': f'High cluster quality ({cluster.quality.value}) with significant size'
            })
        
        # Hypothesis 2: Based on affected systems
        if len(context.affected_systems) > 2:
            hypotheses.append({
                'type': 'CASCADE_FAILURE',
                'description': f'Multiple systems affected ({", ".join(context.affected_systems)}) indicates cascade failure',
                'confidence': 0.8,
                'evidence': f'Cross-system correlation in cluster {cluster.cluster_id}'
            })
        
        # Hypothesis 3: Based on cluster separation
        if cluster.silhouette_score > 0.7:
            hypotheses.append({
                'type': 'DISTINCT_PATTERN',
                'description': 'High cluster separation indicates distinct anomaly pattern',
                'confidence': 0.85,
                'evidence': f'Silhouette score: {cluster.silhouette_score:.3f}'
            })
        
        return hypotheses
    
    async def _agent_recommendations(self, context: AgentContext) -> List[Dict]:
        """Agent generates recommendations based on analysis"""
        cluster = context.cluster_info
        
        recommendations = []
        
        # High-priority cluster recommendations
        if cluster.size > 25:
            recommendations.append({
                'priority': 'CRITICAL',
                'action': 'IMMEDIATE_INVESTIGATION',
                'description': f'Investigate large anomaly cluster ({cluster.size} occurrences) immediately',
                'target_systems': context.affected_systems,
                'expected_impact': 'High - addresses major recurring issue'
            })
        
        # System-specific recommendations
        for system in context.affected_systems:
            recommendations.append({
                'priority': 'HIGH',
                'action': f'DEEP_DIVE_{system}',
                'description': f'Perform deep analysis of {system} system anomalies',
                'target_systems': [system],
                'expected_impact': f'Medium - targeted {system} issue resolution'
            })
        
        # Monitoring recommendations
        if cluster.quality in [ClusterQuality.EXCELLENT, ClusterQuality.GOOD]:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'ENHANCED_MONITORING',
                'description': f'Set up enhanced monitoring for this pattern signature',
                'target_systems': context.affected_systems,
                'expected_impact': 'Medium - early detection of similar patterns'
            })
        
        return recommendations
    
    # ================================
    # SYNTHESIS AND REPORTING
    # ================================
    
    async def _synthesis_node(self, state: AgentState) -> AgentState:
        """Synthesize insights from clustering and agent investigations"""
        print("\n⚡ PHASE 5: SYNTHESIS")
        print("-" * 40)
        
        clustering_insights = state['clustering_insights']
        agent_investigations = state['agent_findings']['investigations']
        
        # Aggregate findings
        synthesis = {
            'clustering_summary': self._summarize_clustering_results(clustering_insights),
            'investigation_summary': self._summarize_investigations(agent_investigations),
            'cross_pattern_insights': self._identify_cross_pattern_insights(agent_investigations),
            'prioritized_actions': self._prioritize_all_recommendations(agent_investigations),
            'system_impact_analysis': self._analyze_overall_system_impact(agent_investigations)
        }
        
        state['agent_findings']['synthesis'] = synthesis
        
        print("✅ Synthesis completed - combining ML clustering with AI insights")
        
        return state
    
    async def _final_report_node(self, state: AgentState) -> AgentState:
        """Generate comprehensive final report"""
        print("\n📋 PHASE 6: FINAL REPORT GENERATION")
        print("-" * 40)
        
        clustering_insights = state['clustering_insights']
        synthesis = state['agent_findings']['synthesis']
        
        final_report = {
            'report_id': f"HYBRID_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'analysis_type': 'Hybrid KMeans + Agentic AI',
            'clustering_results': {
                'n_clusters': clustering_insights['n_clusters'],
                'silhouette_score': clustering_insights['silhouette_score'],
                'feature_importance': clustering_insights['feature_importance']
            },
            'agent_investigations': len(state['agent_findings']['investigations']),
            'synthesis': synthesis,
            'executive_summary': self._generate_executive_summary(synthesis),
            'timestamp': datetime.now().isoformat(),
            'status': InvestigationStatus.COMPLETED
        }
        
        state['final_report'] = final_report
        state['investigation_status'] = InvestigationStatus.COMPLETED
        
        print("✅ Final report generated")
        
        return state
    
    # ================================
    # HELPER METHODS
    # ================================
    
    def _preprocess_for_clustering(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Preprocess data for KMeans clustering"""
        # Select categorical columns for encoding
        categorical_cols = [col for col in df.columns if col not in ['CategoryID']]
        
        processed_df = df[categorical_cols].copy()
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            processed_df[col] = self.label_encoders[col].fit_transform(processed_df[col].astype(str))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(processed_df)
        
        return X_scaled, categorical_cols
    
    def _find_optimal_clusters(self, X: np.ndarray, max_k: int = 10) -> Tuple[int, List[float]]:
        """Find optimal number of clusters using silhouette analysis"""
        silhouette_scores = []
        K_range = range(2, min(max_k + 1, len(X) // 2))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        
        optimal_k = K_range[np.argmax(silhouette_scores)]
        return optimal_k, silhouette_scores
    
    def _analyze_clusters(self, X: np.ndarray, labels: np.ndarray, feature_names: List[str]) -> List[ClusterInfo]:
        """Analyze each cluster in detail"""
        clusters = []
        
        for cluster_id in np.unique(labels):
            cluster_mask = labels == cluster_id
            cluster_data = X[cluster_mask]
            cluster_records = self.anomaly_df.iloc[cluster_mask]
            
            # Calculate cluster metrics
            centroid = np.mean(cluster_data, axis=0)
            size = len(cluster_data)
            
            # Calculate individual silhouette scores for this cluster
            if len(np.unique(labels)) > 1:
                cluster_silhouette = silhouette_score(X, labels)
            else:
                cluster_silhouette = 0
            
            # Determine dominant features
            dominant_features = self._find_dominant_features(cluster_data, feature_names)
            
            # Assess cluster quality
            quality = self._assess_cluster_quality(size, cluster_silhouette, len(X))
            
            cluster_info = ClusterInfo(
                cluster_id=cluster_id,
                size=size,
                centroid=centroid,
                silhouette_score=cluster_silhouette,
                inertia=np.sum((cluster_data - centroid) ** 2),
                dominant_features=dominant_features,
                quality=quality,
                records=cluster_records
            )
            
            clusters.append(cluster_info)
        
        return clusters
    
    def _find_dominant_features(self, cluster_data: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Find dominant features in a cluster"""
        # Calculate feature means for this cluster
        feature_means = np.mean(cluster_data, axis=0)
        
        # Get top 5 features
        top_indices = np.argsort(feature_means)[-5:]
        
        dominant_features = {}
        for idx in top_indices:
            if idx < len(feature_names):
                dominant_features[feature_names[idx]] = float(feature_means[idx])
        
        return dominant_features
    
    def _assess_cluster_quality(self, size: int, silhouette: float, total_size: int) -> ClusterQuality:
        """Assess the quality of a cluster"""
        size_ratio = size / total_size
        
        if silhouette > 0.6 and size_ratio > 0.05:
            return ClusterQuality.EXCELLENT
        elif silhouette > 0.4 and size_ratio > 0.03:
            return ClusterQuality.GOOD
        elif silhouette > 0.2 and size_ratio > 0.02:
            return ClusterQuality.FAIR
        else:
            return ClusterQuality.POOR
    
    def _calculate_feature_importance(self, X: np.ndarray, labels: np.ndarray, 
                                    feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance for clustering"""
        # Use cluster separation as proxy for feature importance
        importance = {}
        
        for i, feature_name in enumerate(feature_names):
            feature_values = X[:, i]
            
            # Calculate variance between clusters vs within clusters
            between_cluster_var = 0
            within_cluster_var = 0
            
            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_values = feature_values[cluster_mask]
                
                if len(cluster_values) > 1:
                    cluster_mean = np.mean(cluster_values)
                    global_mean = np.mean(feature_values)
                    
                    between_cluster_var += len(cluster_values) * (cluster_mean - global_mean) ** 2
                    within_cluster_var += np.sum((cluster_values - cluster_mean) ** 2)
            
            # F-ratio as importance measure
            if within_cluster_var > 0:
                f_ratio = between_cluster_var / within_cluster_var
                importance[feature_name] = f_ratio
            else:
                importance[feature_name] = 0
        
        # Normalize importance scores
        max_importance = max(importance.values()) if importance.values() else 1
        for feature in importance:
            importance[feature] = importance[feature] / max_importance
        
        return importance
    
    def _analyze_cluster_composition(self, cluster: ClusterInfo) -> Dict:
        """Analyze the composition of a cluster"""
        records = cluster.records
        
        composition = {
            'size': cluster.size,
            'categories': records['Category'].value_counts().to_dict() if 'Category' in records.columns else {},
            'system_anomalies': {}
        }
        
        # Count anomalies per system
        systems = ['CJCM', 'CASSANDRA', 'Product', 'Billing']
        for system in systems:
            anomaly_col = f'{system}_Anomaly'
            if anomaly_col in records.columns:
                anomaly_count = (records[anomaly_col] == 'Yes').sum()
                composition['system_anomalies'][system] = {
                    'count': anomaly_count,
                    'rate': anomaly_count / len(records)
                }
        
        return composition
    
    def _identify_cluster_patterns(self, cluster: ClusterInfo) -> Dict:
        """Identify patterns within a cluster"""
        records = cluster.records
        
        patterns = {
            'dominant_category': records['Category'].mode().iloc[0] if 'Category' in records.columns and not records['Category'].mode().empty else None,
            'anomaly_pattern': self._extract_anomaly_pattern(records),
            'status_patterns': self._extract_status_patterns(records)
        }
        
        return patterns
    
    def _extract_anomaly_pattern(self, records: pd.DataFrame) -> str:
        """Extract anomaly pattern from records"""
        anomaly_systems = []
        systems = ['CJCM', 'CASSANDRA', 'Product', 'Billing']
        
        for system in systems:
            anomaly_col = f'{system}_Anomaly'
            if anomaly_col in records.columns:
                anomaly_rate = (records[anomaly_col] == 'Yes').mean()
                if anomaly_rate > 0.5:  # More than 50% have anomalies
                    anomaly_systems.append(system)
        
        return '+'.join(anomaly_systems) if anomaly_systems else 'No_Clear_Pattern'
    
    def _extract_status_patterns(self, records: pd.DataFrame) -> Dict:
        """Extract status patterns from records"""
        patterns = {}
        systems = ['CJCM', 'CASSANDRA', 'Product', 'Billing']
        
        for system in systems:
            status_col = f'{system}_Status'
            if status_col in records.columns:
                most_common = records[status_col].mode()
                patterns[system] = most_common.iloc[0] if not most_common.empty else 'Unknown'
        
        return patterns
    
    def _assess_cluster_significance(self, cluster: ClusterInfo, clustering_results: ClusteringResults) -> Dict:
        """Assess the significance of a cluster"""
        total_records = sum(c.size for c in clustering_results.clusters)
        
        significance = {
            'relative_size': cluster.size / total_records,
            'silhouette_rank': self._get_silhouette_rank(cluster, clustering_results),
            'anomaly_density': self._calculate_anomaly_density(cluster),
            'uniqueness_score': self._calculate_uniqueness_score(cluster, clustering_results)
        }
        
        return significance
    
    def _get_silhouette_rank(self, cluster: ClusterInfo, clustering_results: ClusteringResults) -> int:
        """Get the rank of cluster by silhouette score"""
        scores = [c.silhouette_score for c in clustering_results.clusters]
        scores.sort(reverse=True)
        return scores.index(cluster.silhouette_score) + 1
    
    def _calculate_anomaly_density(self, cluster: ClusterInfo) -> float:
        """Calculate the density of anomalies in the cluster"""
        records = cluster.records
        total_anomalies = 0
        total_possible = 0
        
        systems = ['CJCM', 'CASSANDRA', 'Product', 'Billing']
        for system in systems:
            anomaly_col = f'{system}_Anomaly'
            if anomaly_col in records.columns:
                total_anomalies += (records[anomaly_col] == 'Yes').sum()
                total_possible += len(records)
        
        return total_anomalies / total_possible if total_possible > 0 else 0
    
    def _calculate_uniqueness_score(self, cluster: ClusterInfo, clustering_results: ClusteringResults) -> float:
        """Calculate how unique this cluster is compared to others"""
        # Use distance from other cluster centroids as uniqueness measure
        distances = []
        for other_cluster in clustering_results.clusters:
            if other_cluster.cluster_id != cluster.cluster_id:
                distance = np.linalg.norm(cluster.centroid - other_cluster.centroid)
                distances.append(distance)
        
        return np.mean(distances) if distances else 0
    
    def _calculate_priority_score(self, cluster: ClusterInfo, patterns: Dict) -> float:
        """Calculate priority score for cluster investigation"""
        # Combine multiple factors for priority
        size_score = min(cluster.size / 50, 1.0)  # Normalize by max expected size
        quality_score = {
            ClusterQuality.EXCELLENT: 1.0,
            ClusterQuality.GOOD: 0.8,
            ClusterQuality.FAIR: 0.6,
            ClusterQuality.POOR: 0.3
        }[cluster.quality]
        
        # Anomaly pattern complexity (more systems = higher priority)
        anomaly_pattern = patterns.get('anomaly_pattern', '')
        complexity_score = len(anomaly_pattern.split('+')) / 4 if anomaly_pattern != 'No_Clear_Pattern' else 0.1
        
        # Weighted combination
        priority_score = (0.4 * size_score + 0.3 * quality_score + 0.3 * complexity_score)
        
        return priority_score
    
    def _extract_pattern_signature(self, cluster: ClusterInfo) -> str:
        """Extract pattern signature from cluster"""
        records = cluster.records
        signature_parts = []
        
        systems = ['CJCM', 'CASSANDRA', 'Product', 'Billing']
        for system in systems:
            anomaly_col = f'{system}_Anomaly'
            status_col = f'{system}_Status'
            
            if anomaly_col in records.columns:
                anomaly_rate = (records[anomaly_col] == 'Yes').mean()
                if anomaly_rate > 0.3:  # Significant anomaly rate
                    if status_col in records.columns:
                        dominant_status = records[status_col].mode().iloc[0] if not records[status_col].mode().empty else 'Unknown'
                        signature_parts.append(f"{system}:{dominant_status}:ANOMALY")
                    else:
                        signature_parts.append(f"{system}:ANOMALY")
        
        return '|'.join(signature_parts)
    
    def _identify_affected_systems(self, cluster: ClusterInfo) -> List[str]:
        """Identify systems affected in the cluster"""
        records = cluster.records
        affected_systems = []
        
        systems = ['CJCM', 'CASSANDRA', 'Product', 'Billing']
        for system in systems:
            anomaly_col = f'{system}_Anomaly'
            if anomaly_col in records.columns:
                anomaly_rate = (records[anomaly_col] == 'Yes').mean()
                if anomaly_rate > 0.2:  # 20% threshold for affected system
                    affected_systems.append(system)
        
        return affected_systems
    
    def _calculate_severity(self, cluster: ClusterInfo, analysis: Dict) -> str:
        """Calculate severity based on cluster characteristics"""
        size = cluster.size
        quality = cluster.quality
        priority_score = analysis['priority_score']
        
        if size >= 30 and quality == ClusterQuality.EXCELLENT and priority_score > 0.8:
            return 'CRITICAL'
        elif size >= 20 and quality in [ClusterQuality.EXCELLENT, ClusterQuality.GOOD] and priority_score > 0.6:
            return 'HIGH'
        elif size >= 10 and priority_score > 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _analyze_cluster_features(self, cluster: ClusterInfo, clustering_results: ClusteringResults) -> Dict:
        """Analyze features that characterize this cluster"""
        feature_importance = clustering_results.feature_importance
        dominant_features = cluster.dominant_features
        
        # Find features that are most characteristic of this cluster
        characteristic_features = {}
        for feature, importance in feature_importance.items():
            if feature in dominant_features and importance > 0.5:
                characteristic_features[feature] = {
                    'global_importance': importance,
                    'cluster_value': dominant_features[feature],
                    'characterizes_cluster': True
                }
        
        return {
            'characteristic_features': characteristic_features,
            'feature_count': len(characteristic_features),
            'dominant_features': dominant_features
        }
    
    def _analyze_cluster_separability(self, cluster: ClusterInfo, clustering_results: ClusteringResults) -> Dict:
        """Analyze how well separated this cluster is"""
        return {
            'silhouette_score': cluster.silhouette_score,
            'cluster_quality': cluster.quality.value,
            'uniqueness_score': self._calculate_uniqueness_score(cluster, clustering_results),
            'well_separated': cluster.silhouette_score > 0.5
        }
    
    def _calculate_system_correlation(self, records: pd.DataFrame, sys1: str, sys2: str) -> float:
        """Calculate correlation between two systems' anomalies"""
        sys1_col = f'{sys1}_Anomaly'
        sys2_col = f'{sys2}_Anomaly'
        
        if sys1_col in records.columns and sys2_col in records.columns:
            sys1_anomalies = (records[sys1_col] == 'Yes')
            sys2_anomalies = (records[sys2_col] == 'Yes')
            
            # Calculate Jaccard similarity
            intersection = (sys1_anomalies & sys2_anomalies).sum()
            union = (sys1_anomalies | sys2_anomalies).sum()
            
            return intersection / union if union > 0 else 0
        
        return 0
    
    def _detect_cascade_patterns(self, records: pd.DataFrame, systems: List[str]) -> Dict:
        """Detect cascade failure patterns"""
        cascade_sequences = []
        
        for _, record in records.iterrows():
            anomalous_systems = []
            for system in systems:
                anomaly_col = f'{system}_Anomaly'
                if anomaly_col in record and record[anomaly_col] == 'Yes':
                    anomalous_systems.append(system)
            
            if len(anomalous_systems) > 1:
                cascade_sequences.append(anomalous_systems)
        
        # Find most common cascade pattern
        cascade_counter = Counter([tuple(sorted(seq)) for seq in cascade_sequences])
        most_common = cascade_counter.most_common(1)
        
        return {
            'cascade_detected': len(cascade_sequences) > 0,
            'cascade_rate': len(cascade_sequences) / len(records),
            'most_common_cascade': list(most_common[0][0]) if most_common else None,
            'cascade_frequency': most_common[0][1] if most_common else 0
        }
    
    def _identify_critical_system(self, records: pd.DataFrame, systems: List[str]) -> str:
        """Identify the most critical system in anomalies"""
        system_scores = {}
        
        for system in systems:
            anomaly_col = f'{system}_Anomaly'
            if anomaly_col in records.columns:
                anomaly_rate = (records[anomaly_col] == 'Yes').mean()
                
                # Consider both frequency and correlation with other systems
                correlation_sum = 0
                for other_system in systems:
                    if other_system != system:
                        correlation_sum += self._calculate_system_correlation(records, system, other_system)
                
                system_scores[system] = anomaly_rate * (1 + correlation_sum)
        
        return max(system_scores.items(), key=lambda x: x[1])[0] if system_scores else 'Unknown'
    
    def _summarize_clustering_results(self, clustering_insights: Dict) -> Dict:
        """Summarize clustering results"""
        clustering_results = clustering_insights['clustering_results']
        
        return {
            'total_clusters': len(clustering_results.clusters),
            'optimal_k': clustering_results.optimal_k,
            'overall_quality': clustering_results.overall_silhouette,
            'cluster_sizes': [c.size for c in clustering_results.clusters],
            'high_quality_clusters': len([c for c in clustering_results.clusters 
                                        if c.quality in [ClusterQuality.EXCELLENT, ClusterQuality.GOOD]]),
            'top_features': dict(list(clustering_results.feature_importance.items())[:5])
        }
    
    def _summarize_investigations(self, investigations: List[Dict]) -> Dict:
        """Summarize agent investigations"""
        total_investigations = len(investigations)
        
        # Aggregate hypotheses
        all_hypotheses = []
        for inv in investigations:
            all_hypotheses.extend(inv.get('hypothesis_generation', []))
        
        hypothesis_types = [h['type'] for h in all_hypotheses]
        
        # Aggregate recommendations
        all_recommendations = []
        for inv in investigations:
            all_recommendations.extend(inv.get('recommendations', []))
        
        priority_counts = Counter([r['priority'] for r in all_recommendations])
        
        return {
            'total_investigations': total_investigations,
            'total_hypotheses': len(all_hypotheses),
            'common_hypothesis_types': Counter(hypothesis_types).most_common(3),
            'total_recommendations': len(all_recommendations),
            'priority_distribution': dict(priority_counts),
            'avg_hypotheses_per_pattern': len(all_hypotheses) / total_investigations if total_investigations > 0 else 0
        }
    
    def _identify_cross_pattern_insights(self, investigations: List[Dict]) -> Dict:
        """Identify insights across multiple patterns"""
        # Find common affected systems
        all_affected_systems = []
        for inv in investigations:
            cluster_id = inv['cluster_id']
            # Extract affected systems from context (simplified)
            all_affected_systems.extend(['CJCM', 'CASSANDRA'])  # Placeholder
        
        system_frequency = Counter(all_affected_systems)
        
        # Find recurring hypothesis types
        hypothesis_patterns = []
        for inv in investigations:
            hypotheses = inv.get('hypothesis_generation', [])
            for h in hypotheses:
                hypothesis_patterns.append(h['type'])
        
        return {
            'most_affected_systems': system_frequency.most_common(3),
            'recurring_patterns': Counter(hypothesis_patterns).most_common(3),
            'cross_cluster_correlations': self._find_cross_cluster_correlations(investigations)
        }
    
    def _find_cross_cluster_correlations(self, investigations: List[Dict]) -> List[Dict]:
        """Find correlations between different clusters"""
        correlations = []
        
        for i, inv1 in enumerate(investigations):
            for j, inv2 in enumerate(investigations[i+1:], i+1):
                # Check for similar system correlations
                similarity_score = self._calculate_investigation_similarity(inv1, inv2)
                
                if similarity_score > 0.7:
                    correlations.append({
                        'cluster_1': inv1['cluster_id'],
                        'cluster_2': inv2['cluster_id'],
                        'similarity': similarity_score,
                        'common_systems': ['CJCM', 'CASSANDRA']  # Simplified
                    })
        
        return correlations
    
    def _calculate_investigation_similarity(self, inv1: Dict, inv2: Dict) -> float:
        """Calculate similarity between two investigations"""
        # Simplified similarity calculation
        # In production, this would compare actual investigation features
        return 0.8  # Placeholder
    
    def _prioritize_all_recommendations(self, investigations: List[Dict]) -> List[Dict]:
        """Prioritize all recommendations across investigations"""
        all_recommendations = []
        
        for inv in investigations:
            recommendations = inv.get('recommendations', [])
            for rec in recommendations:
                rec['source_cluster'] = inv['cluster_id']
                all_recommendations.append(rec)
        
        # Sort by priority
        priority_order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        all_recommendations.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
        
        return all_recommendations[:10]  # Top 10 recommendations
    
    def _analyze_overall_system_impact(self, investigations: List[Dict]) -> Dict:
        """Analyze overall system impact across all patterns"""
        system_impact = {
            'CJCM': {'frequency': 0, 'severity': 0},
            'CASSANDRA': {'frequency': 0, 'severity': 0},
            'Product': {'frequency': 0, 'severity': 0},
            'Billing': {'frequency': 0, 'severity': 0}
        }
        
        # Aggregate impact across investigations
        for inv in investigations:
            # Simplified impact calculation
            for system in system_impact:
                system_impact[system]['frequency'] += 1
                system_impact[system]['severity'] += 2  # Placeholder
        
        # Calculate overall scores
        for system in system_impact:
            impact = system_impact[system]
            impact['overall_score'] = impact['frequency'] * impact['severity']
        
        return system_impact
    
    def _generate_executive_summary(self, synthesis: Dict) -> Dict:
        """Generate executive summary"""
        clustering_summary = synthesis['clustering_summary']
        investigation_summary = synthesis['investigation_summary']
        
        return {
            'key_findings': [
                f"Identified {clustering_summary['total_clusters']} distinct anomaly patterns",
                f"Conducted {investigation_summary['total_investigations']} detailed investigations",
                f"Generated {investigation_summary['total_recommendations']} actionable recommendations"
            ],
            'critical_issues': len([r for r in synthesis['prioritized_actions'] if r['priority'] == 'CRITICAL']),
            'most_affected_system': max(synthesis['system_impact_analysis'].items(), 
                                      key=lambda x: x[1]['overall_score'])[0],
            'confidence_level': 'High' if clustering_summary['overall_quality'] > 0.6 else 'Medium'
        }
    
    # ================================
    # MAIN EXECUTION METHODS
    # ================================
    
    async def run_hybrid_analysis(self, max_investigations: int = 5) -> Dict:
        """Run the complete hybrid analysis workflow"""
        
        # Initialize state
        initial_state = AgentState(
            agent_context=None,  # Will be populated during workflow
            investigation_status=InvestigationStatus.PENDING,
            clustering_insights={},
            pattern_analysis={},
            agent_findings={},
            recommendations=[],
            final_report=None
        )
        
        # Execute the hybrid workflow
        final_state = await self.workflow.ainvoke(initial_state)
        
        return final_state['final_report']
    
    def print_hybrid_report(self, report: Dict):
        """Print comprehensive hybrid analysis report"""
        print("\n" + "="*80)
        print("🔬🤖 HYBRID KMEANS + AGENTIC AI ANALYSIS REPORT")
        print("="*80)
        
        print(f"\n📋 REPORT SUMMARY:")
        print(f"   Report ID: {report['report_id']}")
        print(f"   Analysis Type: {report['analysis_type']}")
        print(f"   Status: {report['status'].value}")
        print(f"   Timestamp: {report['timestamp']}")
        
        # Clustering Results
        clustering = report['clustering_results']
        print(f"\n🔬 KMEANS CLUSTERING RESULTS:")
        print(f"   • Number of clusters: {clustering['n_clusters']}")
        print(f"   • Silhouette score: {clustering['silhouette_score']:.3f}")
        print(f"   • Agent investigations: {report['agent_investigations']}")
        
        # Executive Summary
        exec_summary = report['synthesis']['investigation_summary']
        print(f"\n📊 EXECUTIVE SUMMARY:")
        for finding in report['synthesis']['clustering_summary']['top_features']:
            print(f"   • {finding}")
        
        print(f"\n🎯 KEY METRICS:")
        print(f"   • Total patterns investigated: {exec_summary['total_investigations']}")
        print(f"   • Hypotheses generated: {exec_summary['total_hypotheses']}")
        print(f"   • Recommendations created: {exec_summary['total_recommendations']}")
        
        # Top Recommendations
        print(f"\n🚨 PRIORITY ACTIONS:")
        for i, rec in enumerate(report['synthesis']['prioritized_actions'][:5], 1):
            print(f"   {i}. [{rec['priority']}] {rec['description']}")
            print(f"      Target: {', '.join(rec['target_systems'])}")
        
        # System Impact
        print(f"\n🏭 SYSTEM IMPACT ANALYSIS:")
        system_impact = report['synthesis']['system_impact_analysis']
        for system, impact in system_impact.items():
            print(f"   • {system}: Impact Score {impact['overall_score']} (Frequency: {impact['frequency']}, Severity: {impact['severity']})")

# Demo and Testing Functions
def create_enhanced_demo_data(n_samples: int = 200) -> pd.DataFrame:
    """Create enhanced demo data with clear patterns for clustering"""
    np.random.seed(42)
    
    categories = ['Apple', 'Samsung', 'Google', 'Microsoft', 'Amazon']
    data = []
    
    # Pattern 1: CJCM-Cassandra cascade failure (40 records)
    for i in range(40):
        data.append({
            'Category': np.random.choice(['Apple', 'Samsung']),
            'CategoryID': np.random.randint(1000, 9999),
            'CJCM_Status': 'Falout',
            'CJCM_Action': 'Activation',
            'CJCM_Anomaly': 'Yes',
            'CASSANDRA_Entitlement': 'Pending',
            'CASSANDRA_Subscription': 'Subscribed',
            'CASSANDRA_Registration': 'None',
            'CASSANDRA_Anomaly': 'Yes',
            'Product_Status': np.random.choice(['None', 'Active']),
            'Product_Anomaly': 'No',
            'Billing_Status': 'Active',
            'Billing_Anomaly': 'No'
        })
    
    # Pattern 2: Product-Billing correlation (30 records)
    for i in range(30):
        data.append({
            'Category': np.random.choice(['Google', 'Microsoft']),
            'CategoryID': np.random.randint(1000, 9999),
            'CJCM_Status': 'Success',
            'CJCM_Action': 'Update',
            'CJCM_Anomaly': 'No',
            'CASSANDRA_Entitlement': 'Active',
            'CASSANDRA_Subscription': 'Subscribed',
            'CASSANDRA_Registration': 'Complete',
            'CASSANDRA_Anomaly': 'No',
            'Product_Status': 'Inactive',
            'Product_Anomaly': 'Yes',
            'Billing_Status': 'Suspended',
            'Billing_Anomaly': 'Yes'
        })
    
    # Pattern 3: Multi-system failure (25 records)
    for i in range(25):
        data.append({
            'Category': np.random.choice(categories),
            'CategoryID': np.random.randint(1000, 9999),
            'CJCM_Status': 'Error',
            'CJCM_Action': 'Sync',
            'CJCM_Anomaly': 'Yes',
            'CASSANDRA_Entitlement': 'Expired',
            'CASSANDRA_Subscription': 'Cancelled',
            'CASSANDRA_Registration': 'Failed',
            'CASSANDRA_Anomaly': 'Yes',
            'Product_Status': 'Maintenance',
            'Product_Anomaly': 'Yes',
            'Billing_Status': 'Pending',
            'Billing_Anomaly': 'No'
        })
    
    # Random noise patterns (remaining records)
    for i in range(n_samples - 95):
        data.append({
            'Category': np.random.choice(categories),
            'CategoryID': np.random.randint(1000, 9999),
            'CJCM_Status': np.random.choice(['Success', 'Pending', 'Error']),
            'CJCM_Action': np.random.choice(['Activation', 'Update', 'Sync']),
            'CJCM_Anomaly': np.random.choice(['Yes', 'No'], p=[0.3, 0.7]),
            'CASSANDRA_Entitlement': np.random.choice(['Active', 'Pending', 'Expired']),
            'CASSANDRA_Subscription': np.random.choice(['Subscribed', 'Trial', 'Cancelled']),
            'CASSANDRA_Registration': np.random.choice(['Complete', 'Pending', 'Failed']),
            'CASSANDRA_Anomaly': np.random.choice(['Yes', 'No'], p=[0.2, 0.8]),
            'Product_Status': np.random.choice(['Active', 'Inactive', 'Maintenance']),
            'Product_Anomaly': np.random.choice(['Yes', 'No'], p=[0.15, 0.85]),
            'Billing_Status': np.random.choice(['Active', 'Suspended', 'Pending']),
            'Billing_Anomaly': np.random.choice(['Yes', 'No'], p=[0.1, 0.9])
        })
    
    return pd.DataFrame(data)

# Main execution example
async def main_hybrid_demo():
    """Main demo function for hybrid analysis"""
    print("🚀 HYBRID KMEANS + AGENTIC AI DEMO")
    print("="*50)
    
    # Create enhanced demo data
    demo_df = create_enhanced_demo_data(200)
    print(f"📊 Created demo dataset with {len(demo_df)} records")
    
    # Initialize hybrid analyzer
    hybrid_analyzer = HybridKMeansAgenticAnalyzer(demo_df)
    
    # Run hybrid analysis
    print(f"\n🔄 Running hybrid analysis...")
    report = await hybrid_analyzer.run_hybrid_analysis()
    
    # Print comprehensive report
    hybrid_analyzer.print_hybrid_report(report)
    
    return report

# Production integration helper
class ProductionHybridIntegration:
    """Production-ready integration for hybrid analysis"""
    
    def __init__(self, anomaly_df: pd.DataFrame):
        self.analyzer = HybridKMeansAgenticAnalyzer(anomaly_df)
        self.analysis_history = []
    
    async def run_scheduled_analysis(self, schedule_interval: str = 'daily') -> Dict:
        """Run scheduled hybrid analysis"""
        print(f"🔄 Running scheduled {schedule_interval} hybrid analysis...")
        
        start_time = datetime.now()
        report = await self.analyzer.run_hybrid_analysis()
        end_time = datetime.now()
        
        # Add execution metadata
        report['execution_metadata'] = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': (end_time - start_time).total_seconds(),
            'schedule_interval': schedule_interval
        }
        
        # Store in history
        self.analysis_history.append(report)
        
        return report
    
    def get_trend_analysis(self) -> Dict:
        """Analyze trends across multiple analysis runs"""
        if len(self.analysis_history) < 2:
            return {'error': 'Insufficient history for trend analysis'}
        
        # Compare recent analyses
        latest = self.analysis_history[-1]
        previous = self.analysis_history[-2]
        
        trend_analysis = {
            'cluster_count_trend': latest['clustering_results']['n_clusters'] - previous['clustering_results']['n_clusters'],
            'quality_trend': latest['clustering_results']['silhouette_score'] - previous['clustering_results']['silhouette_score'],
            'investigation_trend': latest['agent_investigations'] - previous['agent_investigations'],
            'trend_direction': 'improving' if latest['clustering_results']['silhouette_score'] > previous['clustering_results']['silhouette_score'] else 'degrading'
        }
        
        return trend_analysis

if __name__ == "__main__":
    print("🎮 HYBRID ANALYSIS DEMO MODE")
    print("🔧 To run actual demo: asyncio.run(main_hybrid_demo())")
    
    print(f"\n📋 HYBRID ARCHITECTURE OVERVIEW:")
    print("1. 🔬 KMeans Clustering: Discovers anomaly patterns")
    print("2. 📊 Cluster Analysis: Evaluates pattern quality & significance")
    print("3. 🎯 Pattern Extraction: Selects high-priority patterns")
    print("4. 🤖 Agentic Investigation: AI agents investigate each pattern")
    print("5. ⚡ Synthesis: Combines ML insights with AI reasoning")
    print("6. 📋 Reporting: Comprehensive actionable insights")
    
    print(f"\n🎯 KEY HYBRID BENEFITS:")
    print("• Objective pattern discovery via unsupervised ML")
    print("• Intelligent investigation via agentic AI") 
    print("• Quality-based prioritization prevents wasted effort")
    print("• Evidence-based recommendations with confidence scores")
    print("• Scales to handle hundreds of patterns automatically")
    print("• Combines statistical rigor with contextual reasoning")
    
    print(f"\n🚀 PRODUCTION DEPLOYMENT:")
    print("1. Install: pip install scikit-learn langgraph langchain-openai")
    print("2. Configure LLM API keys")
    print("3. Replace mock LangGraph with actual imports")
    print("4. Set up scheduled execution: asyncio.run(main_hybrid_demo())")
    print("5. Integrate with monitoring and alerting systems")
