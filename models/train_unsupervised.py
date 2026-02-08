"""
Unsupervised learning: Clustering analysis of psychosis survey data
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yaml
import os

class UnsupervisedPsychosisAnalyzer:
    def __init__(self):
        self.config = self.load_config()
        self.cluster_labels = None
        self.pca_result = None
        self.cluster_model = None
        
    def load_config(self):
        with open('../config/config.yaml', 'r') as file:
            return yaml.safe_load(file)
    
    def load_processed_data(self):
        """Load preprocessed data for unsupervised learning"""
        data_path = '../data/processed/unsupervised_data.pkl'
        if os.path.exists(data_path):
            data = joblib.load(data_path)
            print("Unsupervised learning data loaded successfully")
            print(f"Total samples: {data['X'].shape[0]}")
            print(f"Number of features: {len(data['feature_names'])}")
            return data
        else:
            # Try loading from CSV
            csv_path = '../data/processed/unsupervised_data.csv'
            if os.path.exists(csv_path):
                X = pd.read_csv(csv_path)
                feature_names = X.columns.tolist()
                return {'X': X.values, 'feature_names': feature_names}
            else:
                raise FileNotFoundError("Unsupervised data not found. Run preprocessing first.")
    
    def determine_optimal_clusters(self, X):
        """Determine optimal number of clusters using multiple methods"""
        print("\n" + "="*60)
        print("DETERMINING OPTIMAL NUMBER OF CLUSTERS")
        print("="*60)
        
        max_clusters = min(10, len(X) // 10)  # At least 10 samples per cluster
        max_clusters = max(3, max_clusters)  # Minimum 3 clusters
        
        inertia = []
        silhouette_scores = []
        davies_bouldin_scores = []
        
        print(f"Testing cluster sizes from 2 to {max_clusters}...")
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.config['model']['random_state'], n_init=10)
            kmeans.fit(X)
            
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, kmeans.labels_))
            davies_bouldin_scores.append(davies_bouldin_score(X, kmeans.labels_))
            
            print(f"  k={k}: Inertia={kmeans.inertia_:.2f}, "
                  f"Silhouette={silhouette_scores[-1]:.4f}, "
                  f"DB={davies_bouldin_scores[-1]:.4f}")
        
        # Find optimal k (maximum silhouette score)
        optimal_k_silhouette = np.argmax(silhouette_scores) + 2
        
        # Find optimal k (minimum Davies-Bouldin score)
        optimal_k_db = np.argmin(davies_bouldin_scores) + 2
        
        # Use config value or determined optimal
        config_k = self.config['unsupervised']['n_clusters']
        
        print(f"\nOptimal clusters by Silhouette Score: {optimal_k_silhouette}")
        print(f"Optimal clusters by Davies-Bouldin Score: {optimal_k_db}")
        print(f"Config value: {config_k}")
        
        # Plot results
        self.plot_cluster_metrics(range(2, max_clusters + 1), inertia, 
                                 silhouette_scores, davies_bouldin_scores)
        
        # Choose final k
        final_k = config_k if 2 <= config_k <= max_clusters else optimal_k_silhouette
        
        print(f"\nSelected number of clusters: {final_k}")
        return final_k
    
    def plot_cluster_metrics(self, k_range, inertia, silhouette_scores, db_scores):
        """Plot cluster evaluation metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Elbow method
        axes[0].plot(k_range, inertia, 'bo-')
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method')
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette scores
        axes[1].plot(k_range, silhouette_scores, 'ro-')
        axes[1].set_xlabel('Number of Clusters')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Analysis')
        axes[1].grid(True, alpha=0.3)
        
        # Davies-Bouldin scores
        axes[2].plot(k_range, db_scores, 'go-')
        axes[2].set_xlabel('Number of Clusters')
        axes[2].set_ylabel('Davies-Bouldin Score')
        axes[2].set_title('Davies-Bouldin Index')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/unsupervised/cluster_metrics.png')
        plt.close()
        
        print("Cluster metrics plot saved to: ../results/unsupervised/cluster_metrics.png")
    
    def perform_clustering(self, X, n_clusters):
        """Perform K-means clustering"""
        print("\n" + "="*60)
        print(f"PERFORMING K-MEANS CLUSTERING (k={n_clusters})")
        print("="*60)
        
        kmeans = KMeans(n_clusters=n_clusters, 
                       random_state=self.config['model']['random_state'],
                       n_init=10)
        
        self.cluster_labels = kmeans.fit_predict(X)
        self.cluster_model = kmeans
        
        # Calculate cluster sizes
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        print("\nCluster distribution:")
        for cluster, count in zip(unique, counts):
            percentage = (count / len(X)) * 100
            print(f"  Cluster {cluster}: {count} samples ({percentage:.1f}%)")
        
        # Calculate clustering quality metrics
        silhouette = silhouette_score(X, self.cluster_labels)
        db_score = davies_bouldin_score(X, self.cluster_labels)
        
        print(f"\nClustering quality metrics:")
        print(f"  Silhouette Score: {silhouette:.4f}")
        print(f"  Davies-Bouldin Score: {db_score:.4f}")
        
        return kmeans
    
    def perform_pca_analysis(self, X):
        """Perform PCA for dimensionality reduction and visualization"""
        print("\n" + "="*60)
        print("PERFORMING PCA ANALYSIS")
        print("="*60)
        
        n_components = min(self.config['unsupervised']['pca_components'], X.shape[1])
        pca = PCA(n_components=n_components)
        self.pca_result = pca.fit_transform(X)
        
        # Plot explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(explained_variance) + 1), explained_variance)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Individual Explained Variance')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../results/unsupervised/pca_variance.png')
        plt.close()
        
        print(f"First 5 components explain {cumulative_variance[:5] * 100}% of variance")
        print(f"PCA plot saved to: ../results/unsupervised/pca_variance.png")
        
        return pca
    
    def visualize_clusters(self):
        """Visualize clustering results"""
        print("\n" + "="*60)
        print("VISUALIZING CLUSTERS")
        print("="*60)
        
        if self.cluster_labels is None or self.pca_result is None:
            print("Please run clustering and PCA first")
            return
        
        # Visualize clusters in 2D (first two PCA components)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.pca_result[:, 0], self.pca_result[:, 1], 
                             c=self.cluster_labels, cmap='viridis', 
                             s=50, alpha=0.7, edgecolor='k')
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Cluster Visualization (PCA)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('../results/unsupervised/cluster_visualization.png')
        plt.close()
        
        print("Cluster visualization saved to: ../results/unsupervised/cluster_visualization.png")
    
    def analyze_cluster_profiles(self, X, feature_names):
        """Analyze and describe each cluster's characteristics"""
        print("\n" + "="*60)
        print("CLUSTER PROFILES ANALYSIS")
        print("="*60)
        
        if self.cluster_labels is None:
            print("Please run clustering first")
            return
        
        # Create dataframe with clusters
        df_with_clusters = pd.DataFrame(X, columns=feature_names)
        df_with_clusters['cluster'] = self.cluster_labels
        
        # Calculate mean values per cluster
        cluster_means = df_with_clusters.groupby('cluster').mean()
        
        # Save cluster profiles
        cluster_means.to_csv('../results/unsupervised/cluster_profiles.csv')
        
        print("\nCluster profiles (mean values):")
        print(cluster_means.round(2))
        print(f"\nCluster profiles saved to: ../results/unsupervised/cluster_profiles.csv")
        
        # Analyze each cluster
        print("\n" + "="*60)
        print("CLUSTER CHARACTERISTICS")
        print("="*60)
        
        for cluster in sorted(df_with_clusters['cluster'].unique()):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster]
            
            print(f"\nCluster {cluster} (n={len(cluster_data)}):")
            print("-" * 40)
            
            # Get most distinctive features (highest variance from overall mean)
            overall_mean = df_with_clusters.drop(columns=['cluster']).mean()
            cluster_mean = cluster_data.drop(columns=['cluster']).mean()
            
            # Calculate absolute differences
            differences = (cluster_mean - overall_mean).abs()
            top_features = differences.nlargest(5).index
            
            print("Most distinctive features (difference from overall mean):")
            for feature in top_features:
                diff = cluster_mean[feature] - overall_mean[feature]
                direction = "above" if diff > 0 else "below"
                print(f"  {feature}:")
                print(f"    Cluster mean: {cluster_mean[feature]:.2f}")
                print(f"    Overall mean: {overall_mean[feature]:.2f}")
                print(f"    Difference: {diff:+.2f} ({direction} average)")
        
        return cluster_means
    
    def save_models_and_results(self, kmeans_model, pca_model):
        """Save trained models and results"""
        print("\n" + "="*60)
        print("SAVING MODELS AND RESULTS")
        print("="*60)
        
        # Create directories
        os.makedirs('../models/saved_models/unsupervised', exist_ok=True)
        os.makedirs('../results/unsupervised', exist_ok=True)
        
        # Save models
        joblib.dump(kmeans_model, '../models/saved_models/unsupervised/kmeans_model.pkl')
        joblib.dump(pca_model, '../models/saved_models/unsupervised/pca_model.pkl')
        
        # Save cluster labels
        np.save('../results/unsupervised/cluster_labels.npy', self.cluster_labels)
        
        print("Models saved to: ../models/saved_models/unsupervised/")
        print("Results saved to: ../results/unsupervised/")
    
    def run(self):
        """Main execution method"""
        print("="*70)
        print("UNSUPERVISED LEARNING: CLUSTERING ANALYSIS")
        print("="*70)
        
        # Create directories
        os.makedirs('../results/unsupervised', exist_ok=True)
        
        # Load data
        data = self.load_processed_data()
        X = data['X']
        feature_names = data['feature_names']
        
        # Determine optimal clusters
        optimal_k = self.determine_optimal_clusters(X)
        
        # Perform clustering
        kmeans_model = self.perform_clustering(X, optimal_k)
        
        # Perform PCA
        pca_model = self.perform_pca_analysis(X)
        
        # Visualize clusters
        self.visualize_clusters()
        
        # Analyze cluster profiles
        cluster_profiles = self.analyze_cluster_profiles(X, feature_names)
        
        # Save everything
        self.save_models_and_results(kmeans_model, pca_model)
        
        print("\n" + "="*70)
        print("UNSUPERVISED LEARNING COMPLETE")
        print("="*70)
        
        # Print summary
        print("\nSUMMARY:")
        print(f"- Identified {optimal_k} distinct clusters in the data")
        print(f"- Cluster sizes range from {np.bincount(self.cluster_labels).min()} to {np.bincount(self.cluster_labels).max()} samples")
        print(f"- Silhouette Score: {silhouette_score(X, self.cluster_labels):.4f}")
        print(f"- Check '../results/unsupervised/' for detailed analysis")
        print(f"- Check '../models/saved_models/unsupervised/' for trained models")

if __name__ == "__main__":
    analyzer = UnsupervisedPsychosisAnalyzer()
    analyzer.run()