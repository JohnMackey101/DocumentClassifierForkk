import plotly.express as px
import pandas as pd

class ClusterVisualizer:
    def plot_clusters(self, coords_2d, cluster_labels, doc_names):
        """
        Plot documents in 2-D UMAP space.
        coords_2d  : numpy array of shape (n, 2) — pre-computed UMAP coordinates
        cluster_labels : array-like of integer cluster labels (-1 = noise)
        doc_names  : list of document name strings
        """
        df = pd.DataFrame({
            'x': coords_2d[:, 0],
            'y': coords_2d[:, 1],
            'cluster': [
                'Noise (Unclassified)' if label == -1 else f'Cluster {label}'
                for label in cluster_labels
            ],
            'document': doc_names
        })

        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='cluster',
            hover_data=['document'],
            title='Document Clusters — UMAP 2-D Projection',
            labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'}
        )

        fig.update_traces(marker=dict(size=10, opacity=0.85))
        fig.update_layout(legend_title_text='Cluster')

        return fig
