import streamlit as st
import pandas as pd
import numpy as np
from preprocessor import TextPreprocessor
from clustering import DocumentClustering
from visualizer import ClusterVisualizer
from utils import load_sample_data
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import io
import json

st.set_page_config(page_title="Document Similarity Analyzer", layout="wide")

def get_preprocessing_config():
    """Get preprocessing configuration from sidebar inputs"""
    st.sidebar.header("Preprocessing Options")
    
    with st.sidebar.expander("Text Cleaning Options", expanded=False):
        lowercase = st.checkbox("Convert to lowercase", value=True)
        remove_numbers = st.checkbox("Remove numbers", value=True)
        remove_punctuation = st.checkbox("Remove punctuation", value=True)
        remove_urls = st.checkbox("Remove URLs", value=True)
        remove_emails = st.checkbox("Remove email addresses", value=True)
    
    with st.sidebar.expander("Tokenization Options", expanded=False):
        tokenizer = st.radio(
            "Tokenization method",
            ["word", "sentence", "tweet"],
            index=0,
            help="Choose how to split the text into tokens"
        )
        
        min_word_length = st.number_input("Minimum word length", 1, 10, 2)
        max_word_length = st.number_input("Maximum word length", 10, 100, 100)
    
    with st.sidebar.expander("Stopwords Options", expanded=False):
        remove_stopwords = st.checkbox("Remove stopwords", value=True)
        custom_stopwords = st.text_area(
            "Custom stopwords (one per line)",
            help="Add your own words to remove"
        )
        custom_stopwords = set(custom_stopwords.split('\n')) if custom_stopwords else set()
    
    with st.sidebar.expander("Stemming/Lemmatization", expanded=False):
        stemmer = st.radio(
            "Stemming/Lemmatization method",
            ["lemmatizer", "porter", "snowball", None],
            index=0,
            help="Choose method to normalize words"
        )
    
    with st.sidebar.expander("N-gram Options", expanded=False):
        min_n = st.number_input("Minimum n-gram size", 1, 5, 1)
        max_n = st.number_input("Maximum n-gram size", 1, 5, 1)
    
    return {
        'lowercase': lowercase,
        'remove_numbers': remove_numbers,
        'remove_punctuation': remove_punctuation,
        'remove_urls': remove_urls,
        'remove_emails': remove_emails,
        'tokenizer': tokenizer,
        'min_word_length': min_word_length,
        'max_word_length': max_word_length,
        'remove_stopwords': remove_stopwords,
        'custom_stopwords': custom_stopwords,
        'stemmer': stemmer,
        'ngram_range': (min_n, max_n)
    }

def display_similarity_metrics(similarity_df, selected_doc):
    """Display similarity metrics with improved formatting"""
    st.subheader(f"Similarity Analysis for: {selected_doc}")
    
    if similarity_df.empty:
        st.warning("No similar documents found in the same cluster.")
        return
    
    # Create two columns for metrics
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display the similarity metrics table
        st.markdown("### Detailed Similarity Scores")
        st.markdown("*Higher values (closer to 1.0) indicate greater similarity*")
        st.dataframe(similarity_df)
    
    with col2:
        # Display summary statistics
        st.markdown("### Summary Statistics")
        metrics = ['Cosine Similarity', 'Euclidean Similarity', 
                  'Jaccard Similarity', 'Centroid Similarity']
        
        for metric in metrics:
            values = similarity_df[metric].replace([np.inf, -np.inf], np.nan)
            avg_score = values.mean() if not pd.isna(values.mean()) else 0.0
            max_score = values.max() if not pd.isna(values.max()) else 0.0
            
            st.metric(
                label=f"Average {metric}",
                value=f"{avg_score:.4f}",
                delta=f"Max: {max_score:.4f}"
            )

def export_results(documents, processed_texts, cluster_labels, vectors, clustering_instance):
    """Export clustering results in various formats with complete similarity metrics"""
    st.markdown("---")
    st.header("📥 Export Results")
    
    # Calculate similarity metrics for all documents
    doc_names = [doc["name"] for doc in documents]
    all_similarities = {}
    
    # Calculate similarity metrics for each document
    for doc_idx, doc_name in enumerate(doc_names):
        similarity_metrics = []
        # Calculate similarities with all other documents in the same cluster
        same_cluster_docs = [idx for idx, label in enumerate(cluster_labels) 
                           if label == cluster_labels[doc_idx] and idx != doc_idx]
        
        if same_cluster_docs:
            # Get the document's vector
            doc_vector = vectors[doc_idx]

            for other_idx in same_cluster_docs:
                other_name = doc_names[other_idx]
                other_vector = vectors[other_idx]

                # Ensure vectors are 2D for sklearn pairwise functions.
                # Dense numpy 1D rows (from sentence_embeddings[i]) must be reshaped.
                def to_2d(v):
                    if hasattr(v, 'getnnz'):
                        return v  # sparse matrix row is already 2D
                    return v.reshape(1, -1) if v.ndim == 1 else v

                dv = to_2d(doc_vector)
                ov = to_2d(other_vector)

                # Calculate various similarity metrics
                cosine_sim = float(cosine_similarity(dv, ov)[0][0])
                eucl_dist = float(euclidean_distances(dv, ov)[0][0])
                eucl_sim = 1 / (1 + eucl_dist) if eucl_dist != 0 else 1.0
                
                # Calculate Jaccard similarity
                doc_tokens = set(processed_texts[doc_idx].split())
                other_tokens = set(processed_texts[other_idx].split())
                jaccard_sim = len(doc_tokens.intersection(other_tokens)) / len(doc_tokens.union(other_tokens)) if doc_tokens or other_tokens else 0.0
                
                # Calculate centroid similarity using KMeans centers.
                # HDBSCAN labels may be -1 (noise); skip centroid for those.
                centroid_sim = 0.0
                try:
                    lbl = cluster_labels[doc_idx]
                    if lbl >= 0 and hasattr(clustering_instance.kmeans, 'cluster_centers_'):
                        centroid = clustering_instance.kmeans.cluster_centers_[lbl]
                        centroid_sim = float(cosine_similarity(to_2d(other_vector), centroid.reshape(1, -1))[0][0])
                except Exception:
                    pass

                similarity_metrics.append({
                    "document": other_name,
                    "cosine_similarity": round(cosine_sim, 4),
                    "euclidean_similarity": round(eucl_sim, 4),
                    "jaccard_similarity": round(jaccard_sim, 4),
                    "centroid_similarity": round(centroid_sim, 4),
                    "average_similarity": round((cosine_sim + eucl_sim + jaccard_sim + centroid_sim) / 4, 4)
                })
        
        all_similarities[doc_name] = similarity_metrics
    
    # Prepare export data
    export_data = {
        "documents": [
            {
                "name": doc["name"],
                "content": doc["content"],
                "processed_content": proc_text,
                "cluster": int(cluster),
                "similarity_metrics": all_similarities[doc["name"]]
            }
            for doc, proc_text, cluster in zip(documents, processed_texts, cluster_labels)
        ],
        "cluster_statistics": clustering_instance.get_cluster_statistics(cluster_labels).to_dict()
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Export as JSON
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="clustering_results.json",
            mime="application/json"
        )
    
    with col2:
        # Export as CSV
        documents_df = pd.DataFrame([{
            'name': doc["name"],
            'content': doc["content"],
            'processed_content': proc_text,
            'cluster': int(cluster)
        } for doc, proc_text, cluster in zip(documents, processed_texts, cluster_labels)])
        csv_buffer = io.StringIO()
        documents_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name="clustering_results.csv",
            mime="text/csv"
        )
    
    # Display preview of export data
    with st.expander("Preview Export Data"):
        st.json(export_data)

def main():
    st.title("Document Similarity Analyzer")
    
    # Get preprocessing configuration
    preproc_config = get_preprocessing_config()
    
    # Sidebar for document upload
    st.sidebar.header("Upload Documents")
    upload_option = st.sidebar.radio(
        "Choose input method:",
        ["Upload Files", "Input Text", "Use Sample Data"]
    )
    
    documents = []
    if upload_option == "Upload Files":
        uploaded_files = st.sidebar.file_uploader(
            "Upload text files", 
            type=['txt'], 
            accept_multiple_files=True
        )
        if uploaded_files:
            for file in uploaded_files:
                content = io.StringIO(file.getvalue().decode("utf-8")).read()
                documents.append({"name": file.name, "content": content})
                
    elif upload_option == "Input Text":
        text_input = st.sidebar.text_area("Enter text documents (one per line)")
        if text_input:
            for idx, doc in enumerate(text_input.split('\n')):
                if doc.strip():
                    documents.append({"name": f"Doc_{idx+1}", "content": doc})
                    
    else:  # Use Sample Data
        documents = load_sample_data()

    if documents:
        # Initialize components
        preprocessor = TextPreprocessor()
        preprocessor.set_config(**preproc_config)
        clustering = DocumentClustering()
        visualizer = ClusterVisualizer()

        # Process documents
        processed_texts = [preprocessor.preprocess(doc["content"]) for doc in documents]
        
        # Show preprocessing examples
        if st.checkbox("Show preprocessing results"):
            st.subheader("Preprocessing Examples")
            for orig, proc in zip(documents[:3], processed_texts[:3]):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Original:", orig["content"])
                with col2:
                    st.write("Processed:", proc)
        
        doc_names = [doc["name"] for doc in documents]

        # Perform clustering
        vectors, sentence_embeddings, cluster_labels, cluster_centers, hdbscan_labels = clustering.cluster_documents(processed_texts)
        
        # Lowkey just gonna replace vectors with sentence_embeddings and cluster_labels with hdbscan_labels for visualization since HDBScan is more robust for small datasets and can handle noise better. KMeans can be sensitive to outliers and may not perform well with small datasets, while HDBScan can identify clusters of varying densities and is less affected by noise. 
    
        vectors = sentence_embeddings
        cluster_labels = hdbscan_labels
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Cluster Visualization")
            # this is for kmeans
            #fig = visualizer.plot_clusters(vectors, cluster_labels, doc_names)

            # hdbscan visualization
            fig = visualizer.plot_clusters(sentence_embeddings, hdbscan_labels, doc_names)
            st.plotly_chart(fig)

        with col2:
            st.header("Cluster Statistics")
            cluster_stats = clustering.get_cluster_statistics(cluster_labels)
            st.write(cluster_stats)

        # Display similarity metrics section
        st.markdown("---")
        st.header("📄 Document Similarity Analysis 📄")
        
        # Make the document selection more prominent
        st.markdown("### Select Document for Analysis")
        selected_doc = st.selectbox(
            "Choose a document to analyze its similarities with others in the same cluster:",
            doc_names,
            help="Select a document to see how similar it is to other documents in its cluster"
        )
        
        if selected_doc:
            similarity_df = clustering.get_similarity_metrics(
                selected_doc,
                doc_names,
                vectors,
                cluster_labels,
                processed_texts
            )
            
            # Display similarity metrics
            display_similarity_metrics(similarity_df, selected_doc)
        
        # Add export functionality with complete similarity metrics
        export_results(documents, processed_texts, cluster_labels, vectors, clustering)

if __name__ == "__main__":
    main()
