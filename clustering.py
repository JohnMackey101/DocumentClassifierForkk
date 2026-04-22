from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd
import numpy as np
import hdbscan
import hdbscan.validity as hdbscan_validity
import umap

class DocumentClustering:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.hdbscan = None
        self.umap_2d = None
        # Stub so legacy centroid-similarity checks don't crash
        self.kmeans = type('_Stub', (), {})()
        # LDA results: {cluster_label: [keyword, ...]}
        self.lda_keywords = {}

    # ------------------------------------------------------------------
    # High-frequency word filter
    # Remove tokens that appear in more than `threshold` fraction of docs
    # ------------------------------------------------------------------
    @staticmethod
    def filter_high_freq_words(processed_texts: list, threshold: float = 0.75) -> list:
        """
        Remove words that appear in more than `threshold` fraction of documents.
        Works on the already-preprocessed (space-joined token) strings.
        """
        n_docs = len(processed_texts)
        if n_docs == 0:
            return processed_texts

        # Count in how many documents each token appears
        doc_freq = {}
        for text in processed_texts:
            for token in set(text.split()):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # Build set of tokens to remove
        cutoff = threshold * n_docs
        high_freq = {tok for tok, cnt in doc_freq.items() if cnt > cutoff}

        if high_freq:
            filtered = [
                " ".join(t for t in text.split() if t not in high_freq)
                for text in processed_texts
            ]
            return filtered
        return processed_texts

    # ------------------------------------------------------------------
    # LDA: top-N keywords per cluster
    # ------------------------------------------------------------------
    def _fit_lda_per_cluster(self, processed_texts: list, cluster_labels, n_top: int = 3):
        """
        For each real cluster (label != -1) fit a small LDA model on the
        cluster's documents and extract the top-N keywords.
        Stores results in self.lda_keywords = {cluster_id: [kw1, kw2, kw3]}.
        """
        self.lda_keywords = {}
        unique_labels = [l for l in np.unique(cluster_labels) if l != -1]

        for label in unique_labels:
            indices = [i for i, l in enumerate(cluster_labels) if l == label]
            cluster_texts = [processed_texts[i] for i in indices]

            # Need at least 1 document and 1 token
            if not cluster_texts or all(t.strip() == "" for t in cluster_texts):
                self.lda_keywords[int(label)] = []
                continue

            try:
                # CountVectorizer for LDA (LDA needs raw counts, not TF-IDF)
                cv = CountVectorizer(max_features=200, ngram_range=(1, 2))
                X = cv.fit_transform(cluster_texts)

                if X.shape[1] == 0:
                    self.lda_keywords[int(label)] = []
                    continue

                n_components = min(1, X.shape[0])  # 1 topic per cluster
                lda = LatentDirichletAllocation(
                    n_components=n_components,
                    random_state=42,
                    max_iter=20
                )
                lda.fit(X)

                feature_names = cv.get_feature_names_out()
                top_indices = lda.components_[0].argsort()[::-1][:n_top]
                keywords = [feature_names[i] for i in top_indices]
                self.lda_keywords[int(label)] = keywords

            except Exception as e:
                print(f"LDA failed for cluster {label}: {e}")
                self.lda_keywords[int(label)] = []

    # ------------------------------------------------------------------
    # Main pipeline: TF-IDF → UMAP → HDBSCAN
    # Two UMAP projections are computed:
    #   • umap_cluster (5-D, n_neighbors=3) — used for HDBSCAN clustering
    #   • umap_2d      (2-D, n_neighbors=3) — used for visualisation
    # ------------------------------------------------------------------
    def cluster_documents(self, processed_texts):
        n_samples = len(processed_texts)

        # 1. Remove high-frequency words (appear in >75% of docs)
        filtered_texts = self.filter_high_freq_words(processed_texts, threshold=0.75)

        # 2. TF-IDF vectorisation (sparse)
        tfidf_matrix = self.vectorizer.fit_transform(filtered_texts)

        # 3a. UMAP for clustering (5-D)
        umap_cluster = umap.UMAP(
            n_components=6,
            n_neighbors=3,
            min_dist=0.1,
            metric='cosine',
            init='random',
            random_state=42
        )
        coords_cluster = umap_cluster.fit_transform(tfidf_matrix)

        # 3b. UMAP for visualisation (2-D)
        self.umap_2d = umap.UMAP(
            n_neighbors=10,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        coords_2d = self.umap_2d.fit_transform(tfidf_matrix)

        # 4. HDBSCAN on the 5-D UMAP embedding
        self.hdbscan = hdbscan.HDBSCAN(
            min_cluster_size=3,
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        cluster_labels = self.hdbscan.fit_predict(coords_cluster)

        # 5. LDA keywords per cluster (uses filtered texts)
        self._fit_lda_per_cluster(filtered_texts, cluster_labels, n_top=3)

        print(f"TF-IDF shape: {tfidf_matrix.shape}")
        print(f"UMAP cluster shape: {coords_cluster.shape}")
        print(f"UMAP 2-D shape: {coords_2d.shape}")
        print(f"HDBSCAN labels: {cluster_labels}")
        print(f"LDA keywords: {self.lda_keywords}")

        return tfidf_matrix, coords_2d, cluster_labels

    # ------------------------------------------------------------------
    # Per-cluster DBCV scores
    # ------------------------------------------------------------------
    def get_dbcv_scores(self, coords_2d, cluster_labels):
        """
        Compute a per-cluster DBCV (Density-Based Clustering Validation) score
        using hdbscan.validity.validity_index with per_cluster_scores=True.

        Returns a dict {cluster_label: score} for real clusters (label != -1).
        Falls back to an overall score split evenly if per-cluster mode fails.
        Score range: -1 (worst) to 1 (best).
        """
        real_mask = cluster_labels != -1
        if real_mask.sum() < 2:
            return {}

        try:
            scores = hdbscan_validity.validity_index(
                coords_2d[real_mask].astype(np.float64),
                cluster_labels[real_mask],
                per_cluster_scores=True
            )
            # scores is a numpy array aligned to sorted unique real labels
            real_labels = sorted(set(cluster_labels[real_mask].tolist()))
            return {int(lbl): round(float(s), 4) for lbl, s in zip(real_labels, scores)}
        except Exception as e:
            print(f"Per-cluster DBCV failed: {e}")
            # Fallback: single overall score
            try:
                overall = hdbscan_validity.validity_index(
                    coords_2d[real_mask].astype(np.float64),
                    cluster_labels[real_mask]
                )
                real_labels = sorted(set(cluster_labels[real_mask].tolist()))
                return {int(lbl): round(float(overall), 4) for lbl in real_labels}
            except Exception as e2:
                print(f"Overall DBCV also failed: {e2}")
                return {}

    # ------------------------------------------------------------------
    # Cluster statistics
    # ------------------------------------------------------------------
    def get_cluster_statistics(self, cluster_labels):
        unique_labels = np.unique(cluster_labels)
        stats = {}
        for label in unique_labels:
            count = int(np.sum(cluster_labels == label))
            label_name = "Noise (Unclassified)" if label == -1 else f"Cluster {label}"
            stats[label_name] = {
                "Number of documents": count,
                "Percentage": f"{(count / len(cluster_labels)) * 100:.2f}%"
            }
        return pd.DataFrame.from_dict(stats, orient='index')

    # ------------------------------------------------------------------
    # Best representative document per cluster
    # ------------------------------------------------------------------
    def get_best_doc_per_cluster(self, doc_names, coords_2d, cluster_labels):
        """
        Returns {cluster_label: doc_name} for every real cluster.
        Best = smallest mean Euclidean distance to all other cluster members.
        """
        best = {}
        unique_labels = [l for l in np.unique(cluster_labels) if l != -1]

        for label in unique_labels:
            indices = [i for i, l in enumerate(cluster_labels) if l == label]
            if len(indices) == 1:
                best[int(label)] = doc_names[indices[0]]
                continue

            cluster_coords = coords_2d[indices]
            mean_dists = []
            for i in range(len(indices)):
                others = np.delete(cluster_coords, i, axis=0)
                mean_dists.append(np.mean(np.linalg.norm(others - cluster_coords[i], axis=1)))

            best[int(label)] = doc_names[indices[int(np.argmin(mean_dists))]]

        return best

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------
    def calculate_jaccard_similarity(self, doc1, doc2):
        try:
            s1, s2 = set(doc1.split()), set(doc2.split())
            if not s1 or not s2:
                return 0.0
            return len(s1 & s2) / len(s1 | s2)
        except Exception:
            return 0.0

    def calculate_centroid_similarity(self, vector, cluster_label):
        return 0.0  # Not applicable with UMAP/HDBSCAN; kept for API compat

    def get_similarity_metrics(self, target_doc, doc_names, vectors, cluster_labels, processed_texts):
        columns = ['Document', 'Cosine Similarity', 'Euclidean Similarity',
                   'Jaccard Similarity', 'Centroid Similarity', 'Average Similarity']
        empty_df = pd.DataFrame(columns=columns)

        if target_doc not in doc_names:
            return empty_df

        target_idx = doc_names.index(target_doc)
        target_cluster = cluster_labels[target_idx]
        target_vector = vectors[target_idx]

        same_cluster_indices = [i for i in range(len(doc_names))
                                if cluster_labels[i] == target_cluster and i != target_idx]
        if not same_cluster_indices:
            return empty_df

        def to_2d(v):
            if hasattr(v, 'getnnz'):
                return v
            return v.reshape(1, -1) if v.ndim == 1 else v

        similarity_metrics = []
        for idx in same_cluster_indices:
            try:
                tv = to_2d(target_vector)
                ov = to_2d(vectors[idx])

                cosine_sim = 0.0
                try:
                    tv_nz = tv.getnnz() > 0 if hasattr(tv, 'getnnz') else np.any(tv != 0)
                    ov_nz = ov.getnnz() > 0 if hasattr(ov, 'getnnz') else np.any(ov != 0)
                    if tv_nz and ov_nz:
                        cosine_sim = round(float(cosine_similarity(tv, ov)[0][0]), 4)
                except Exception:
                    pass

                eucl_sim = 0.0
                try:
                    d = float(euclidean_distances(tv, ov)[0][0])
                    eucl_sim = round(1 / (1 + d), 4) if d != 0 else 1.0
                except Exception:
                    pass

                jaccard_sim = round(self.calculate_jaccard_similarity(
                    processed_texts[target_idx], processed_texts[idx]), 4)

                valid = [v for v in [cosine_sim, eucl_sim, jaccard_sim] if v > 0]
                avg = round(sum(valid) / len(valid), 4) if valid else 0.0

                similarity_metrics.append({
                    'Document': doc_names[idx],
                    'Cosine Similarity': cosine_sim,
                    'Euclidean Similarity': eucl_sim,
                    'Jaccard Similarity': jaccard_sim,
                    'Centroid Similarity': 0.0,
                    'Average Similarity': avg
                })
            except Exception as e:
                print(f"Error for {doc_names[idx]}: {e}")
                continue

        if not similarity_metrics:
            return empty_df

        df = pd.DataFrame(similarity_metrics)
        return df.sort_values('Average Similarity', ascending=False)
