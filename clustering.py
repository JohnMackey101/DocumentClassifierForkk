from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix
import hdbscan

class DocumentClustering:
    def __init__(self, max_clusters=3):
        self.max_clusters = max_clusters
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.kmeans = None  # Will be initialized in cluster_documents
        self.hdbscan = None  # Will be initialized in cluster_documents
        
    def cluster_documents(self, processed_texts):
        # processed_texts is a list of strings (tokens joined by spaces).
        # Word2Vec needs a list of token lists, so split each string back into tokens.
        tokenized_texts = [text.split() for text in processed_texts]

        # CBow word2vec — train on token lists, not raw strings
        cbowmodel = Word2Vec(tokenized_texts, min_count=1,
                                vector_size=1000, window=3)

        # Vectorize texts with sentence embeddings (CBOW)
        kv = cbowmodel.wv
        # Pass token lists (not raw strings) to sentence_embedding
        sentence_embeddings = np.vstack([self.sentence_embedding(kv, tok_list) for tok_list in tokenized_texts])
        # Get sentence embeddings as a n * features matrix (NOTSPARSE). 

        # Vectorize the texts (SPARSE doc*features matrix) — TF-IDF takes strings
        vectors = self.vectorizer.fit_transform(processed_texts) 

        print(vectors.shape, sentence_embeddings.shape)

        # Dynamically set number of clusters based on input size
        n_samples = len(processed_texts)
        n_clusters = min(self.max_clusters, n_samples)
        
        # Initialize KMeans with dynamic number of clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(vectors)
        
        # Get cluster centers
        cluster_centers = self.kmeans.cluster_centers_

        # HDBSCAN struggles in high dimensions with small datasets.
        # Reduce to min(n_samples-1, 10) dims with PCA before clustering.
        n_components = min(len(processed_texts) - 1, 10)
        reduced_embeddings = PCA(n_components=n_components).fit_transform(sentence_embeddings)

        # Initialize HDBScan 
        self.hdbscan = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                        gen_min_span_tree=False, leaf_size=40,
                                        metric='euclidean', 
                                        min_cluster_size=2, 
                                        min_samples=1, p=None)
        
        # Perform HDBScan clustering on PCA-reduced sentence embeddings.
        hdbscan_labels = self.hdbscan.fit(reduced_embeddings).labels_

        print(hdbscan_labels)
        print(cluster_labels)
        
        return vectors, sentence_embeddings, cluster_labels, cluster_centers, hdbscan_labels
    
    def get_cluster_statistics(self, cluster_labels):
        unique_labels = np.unique(cluster_labels)
        stats = {}
        
        for label in unique_labels:
            count = np.sum(cluster_labels == label)
            # HDBSCAN uses -1 for noise/unclassified points — label them clearly
            label_name = "Noise (Unclassified)" if label == -1 else f"Cluster {label}"
            stats[label_name] = {
                "Number of documents": count,
                "Percentage": f"{(count/len(cluster_labels))*100:.2f}%"
            }
            
        return pd.DataFrame.from_dict(stats, orient='index')
    
    def calculate_jaccard_similarity(self, doc1, doc2):
        # Calculate Jaccard similarity between two documents with error handling
        try:
            set1 = set(doc1.split())
            set2 = set(doc2.split())
            
            # Handle empty sets
            if not set1 or not set2:
                return 0.0
                
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            # Handle zero division
            return intersection / union if union > 0 else 0.0
        except Exception:
            return 0.0

    def calculate_centroid_similarity(self, vector, cluster_label):
        # Calculate similarity between a document and its cluster centroid with error handling
        try:
            if not hasattr(self.kmeans, 'cluster_centers_'):
                return 0.0

            # HDBSCAN labels -1 = noise; no valid KMeans centroid to compare against
            if cluster_label < 0:
                return 0.0

            centroid = self.kmeans.cluster_centers_[cluster_label]

            # Check if centroid is zero vector
            if np.all(np.abs(centroid) < 1e-10):
                return 0.0

            # Ensure vector is 2D for cosine_similarity (works for both sparse and dense)
            if hasattr(vector, 'getnnz'):
                # Sparse matrix row — already 2D if sliced correctly
                if vector.getnnz() == 0:
                    return 0.0
                vec_2d = vector
            else:
                # Dense numpy array — reshape to (1, n) if it's 1D
                vec_2d = vector.reshape(1, -1) if vector.ndim == 1 else vector

            similarity = cosine_similarity(vec_2d, centroid.reshape(1, -1))[0][0]

            # Handle NaN values
            return float(similarity) if not np.isnan(similarity) else 0.0
        except Exception:
            return 0.0
        
    def sentence_embedding(self, w2v_kv, tokens) -> np.ndarray:

        vecs = [w2v_kv[w] for w in tokens if w in w2v_kv]
        dim = w2v_kv.vector_size
        if not vecs:
            return np.zeros(dim, dtype=np.float32)
        return np.mean(vecs, axis=0)



    def get_similarity_metrics(self, target_doc, doc_names, vectors, cluster_labels, processed_texts):
        # Calculate multiple similarity metrics for documents with proper error handling
        # Define expected columns
        columns = ['Document', 'Cosine Similarity', 'Euclidean Similarity', 
                  'Jaccard Similarity', 'Centroid Similarity', 'Average Similarity']
        
        # Initialize empty DataFrame with zeros
        empty_df = pd.DataFrame(columns=columns)
        empty_df['Document'] = []
        empty_df[columns[1:]] = 0.0
        
        # Return empty DataFrame if target_doc not found
        if target_doc not in doc_names:
            return empty_df
            
        target_idx = doc_names.index(target_doc)
        target_cluster = cluster_labels[target_idx]
        target_vector = vectors[target_idx]
        
        # Calculate similarities for documents in the same cluster
        same_cluster_indices = [i for i in range(len(doc_names)) 
                              if cluster_labels[i] == target_cluster and i != target_idx]
        
        # Return empty DataFrame if no documents in same cluster
        if not same_cluster_indices:
            return empty_df
        
        similarity_metrics = []
        
        for idx in same_cluster_indices:
            try:
                # Initialize metrics with zeros
                metrics_dict = {
                    'Document': doc_names[idx],
                    'Cosine Similarity': 0.0,
                    'Euclidean Similarity': 0.0,
                    'Jaccard Similarity': 0.0,
                    'Centroid Similarity': 0.0,
                    'Average Similarity': 0.0
                }

                # Helper: ensure a vector is 2D for sklearn pairwise functions.
                # Sparse rows are already 2D; dense 1D arrays need reshaping.
                def to_2d(v):
                    if hasattr(v, 'getnnz'):
                        return v  # sparse matrix row is already 2D
                    return v.reshape(1, -1) if v.ndim == 1 else v

                tv = to_2d(target_vector)
                ov = to_2d(vectors[idx])

                # Cosine similarity with error handling
                try:
                    # For sparse: check non-zero; for dense: check norm
                    tv_nonzero = tv.getnnz() > 0 if hasattr(tv, 'getnnz') else np.any(tv != 0)
                    ov_nonzero = ov.getnnz() > 0 if hasattr(ov, 'getnnz') else np.any(ov != 0)
                    if tv_nonzero and ov_nonzero:
                        cosine_sim = cosine_similarity(tv, ov)[0][0]
                        metrics_dict['Cosine Similarity'] = round(float(cosine_sim), 4)
                except Exception:
                    pass
                
                # Euclidean similarity with error handling
                try:
                    eucl_dist = euclidean_distances(tv, ov)[0][0]
                    if eucl_dist != 0:
                        eucl_sim = 1 / (1 + eucl_dist)
                        metrics_dict['Euclidean Similarity'] = round(float(eucl_sim), 4)
                except Exception:
                    pass
                
                # Jaccard similarity with error handling
                try:
                    jaccard_sim = self.calculate_jaccard_similarity(
                        processed_texts[target_idx],
                        processed_texts[idx]
                    )
                    metrics_dict['Jaccard Similarity'] = round(float(jaccard_sim), 4)
                except Exception:
                    pass
                
                # Centroid similarity with error handling
                try:
                    centroid_sim = self.calculate_centroid_similarity(vectors[idx], cluster_labels[idx])
                    metrics_dict['Centroid Similarity'] = round(float(centroid_sim), 4)
                except Exception:
                    pass
                
                # Calculate average similarity excluding zeros
                valid_metrics = [v for k, v in metrics_dict.items() 
                               if k != 'Document' and v > 0]
                if valid_metrics:
                    metrics_dict['Average Similarity'] = round(sum(valid_metrics) / len(valid_metrics), 4)
                
                similarity_metrics.append(metrics_dict)
            except Exception as e:
                print(f"Error calculating similarity metrics for document {doc_names[idx]}: {str(e)}")
                continue
        
        # If no valid metrics were calculated, return empty DataFrame
        if not similarity_metrics:
            return empty_df
        
        # Create DataFrame and sort by Average Similarity
        df = pd.DataFrame(similarity_metrics)
        return df.sort_values('Average Similarity', ascending=False)
