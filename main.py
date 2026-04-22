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
import fitz  # PyMuPDF

st.set_page_config(page_title="Document Similarity Analyzer", layout="wide")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF file given its raw bytes."""
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


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
        max_n = st.number_input("Maximum n-gram size", 1, 5, 2)

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


def display_best_docs(best_docs: dict, cluster_labels, doc_names: list,
                      documents: list, lda_keywords: dict):
    """Show the most representative document and LDA keywords for each cluster."""
    if not best_docs:
        return

    st.markdown("### 🏆 Best Representative Document per Cluster")
    st.markdown(
        "The document listed for each cluster has the smallest mean distance "
        "to all other members — i.e. it best represents that cluster."
    )

    for cluster_id in sorted(best_docs.keys()):
        best_name = best_docs[cluster_id]
        content = next(
            (d["content"] for d in documents if d["name"] == best_name), ""
        )
        preview = content[:500] + ("…" if len(content) > 500 else "")
        keywords = lda_keywords.get(cluster_id, [])
        kw_str = "  •  ".join(f"`{kw}`" for kw in keywords) if keywords else "*none*"

        with st.expander(f"Cluster {cluster_id} — best doc: **{best_name}**"):
            st.markdown(f"**Document:** `{best_name}`")
            st.markdown(
                f"**Cluster members:** {sum(1 for l in cluster_labels if l == cluster_id)}"
            )
            st.markdown(f"**Top LDA keywords:** {kw_str}")
            st.text_area("Content preview", value=preview, height=150,
                         key=f"best_doc_{cluster_id}", disabled=True)


def display_similarity_metrics(similarity_df, selected_doc):
    """Display similarity metrics with improved formatting"""
    st.subheader(f"Similarity Analysis for: {selected_doc}")

    if similarity_df.empty:
        st.warning("No similar documents found in the same cluster.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Detailed Similarity Scores")
        st.markdown("*Higher values (closer to 1.0) indicate greater similarity*")
        st.dataframe(similarity_df)

    with col2:
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

    doc_names = [doc["name"] for doc in documents]
    all_similarities = {}

    for doc_idx, doc_name in enumerate(doc_names):
        similarity_metrics = []
        same_cluster_docs = [idx for idx, label in enumerate(cluster_labels)
                             if label == cluster_labels[doc_idx] and idx != doc_idx]

        if same_cluster_docs:
            doc_vector = vectors[doc_idx]

            for other_idx in same_cluster_docs:
                other_name = doc_names[other_idx]
                other_vector = vectors[other_idx]

                def to_2d(v):
                    if hasattr(v, 'getnnz'):
                        return v
                    return v.reshape(1, -1) if v.ndim == 1 else v

                dv = to_2d(doc_vector)
                ov = to_2d(other_vector)

                cosine_sim = float(cosine_similarity(dv, ov)[0][0])
                eucl_dist = float(euclidean_distances(dv, ov)[0][0])
                eucl_sim = 1 / (1 + eucl_dist) if eucl_dist != 0 else 1.0

                doc_tokens = set(processed_texts[doc_idx].split())
                other_tokens = set(processed_texts[other_idx].split())
                jaccard_sim = (
                    len(doc_tokens & other_tokens) / len(doc_tokens | other_tokens)
                    if doc_tokens or other_tokens else 0.0
                )

                similarity_metrics.append({
                    "document": other_name,
                    "cosine_similarity": round(cosine_sim, 4),
                    "euclidean_similarity": round(eucl_sim, 4),
                    "jaccard_similarity": round(jaccard_sim, 4),
                    "average_similarity": round((cosine_sim + eucl_sim + jaccard_sim) / 3, 4)
                })

        all_similarities[doc_name] = similarity_metrics

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
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="clustering_results.json",
            mime="application/json"
        )

    with col2:
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

    with st.expander("Preview Export Data"):
        st.json(export_data)


def main():
    col_img, col_spacer = st.columns([0.4, 0.6])
    with col_img:
        st.image("CoEheader.png", use_container_width=True)
    st.title("Document Similarity Analyzer")

    preproc_config = get_preprocessing_config()

    st.sidebar.header("Upload Documents")
    upload_option = st.sidebar.radio(
        "Choose input method:",
        ["Upload Files", "Input Text", "Use Sample Data"]
    )

    documents = []
    if upload_option == "Upload Files":
        uploaded_files = st.sidebar.file_uploader(
            "Upload TXT or PDF files",
            type=['txt', 'pdf'],
            accept_multiple_files=True
        )
        if uploaded_files:
            failed_files = []
            for file in uploaded_files:
                try:
                    if file.name.lower().endswith(".pdf"):
                        content = extract_text_from_pdf(file.getvalue())
                        if not content.strip():
                            st.sidebar.warning(
                                f"⚠️ '{file.name}' appears to be a scanned/image-only PDF "
                                "and no text could be extracted. It will be skipped."
                            )
                            failed_files.append(file.name)
                            continue
                    else:
                        content = io.StringIO(file.getvalue().decode("utf-8")).read()
                    documents.append({"name": file.name, "content": content})
                except Exception as e:
                    st.sidebar.error(f"❌ Failed to read '{file.name}': {e}")
                    failed_files.append(file.name)

            if documents:
                st.sidebar.success(
                    f"OK! - {len(documents)} file(s) loaded successfully"
                    + (f" ({len(failed_files)} skipped)" if failed_files else "")
                )

    elif upload_option == "Input Text":
        text_input = st.sidebar.text_area("Enter text documents (one per line)")
        if text_input:
            for idx, doc in enumerate(text_input.split('\n')):
                if doc.strip():
                    documents.append({"name": f"Doc_{idx+1}", "content": doc})

    else:  # Use Sample Data
        documents = load_sample_data()

    if documents:
        # ── Initialise components ──────────────────────────────────────
        preprocessor = TextPreprocessor()
        preprocessor.set_config(**preproc_config)
        clustering = DocumentClustering()
        visualizer = ClusterVisualizer()

        # ── Preprocess ────────────────────────────────────────────────
        processed_texts = [preprocessor.preprocess(doc["content"]) for doc in documents]

        if st.checkbox("Show preprocessing results"):
            st.subheader("Preprocessing Examples")
            for orig, proc in zip(documents[:3], processed_texts[:3]):
                c1, c2 = st.columns(2)
                with c1:
                    st.write("Original:", orig["content"])
                with c2:
                    st.write("Processed:", proc)

        doc_names = [doc["name"] for doc in documents]

        # ── Cluster: TF-IDF → UMAP (2-D) → HDBSCAN ───────────────────
        tfidf_matrix, coords_2d, cluster_labels = clustering.cluster_documents(processed_texts)

        # ── Visualisation ─────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.header("Cluster Visualization")
            fig = visualizer.plot_clusters(coords_2d, cluster_labels, doc_names)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.header("Cluster Statistics")
            cluster_stats = clustering.get_cluster_statistics(cluster_labels)
            st.write(cluster_stats)

            # DBCV scores per cluster
            dbcv_scores = clustering.get_dbcv_scores(coords_2d, cluster_labels)
            if dbcv_scores:
                st.markdown("**HDBSCAN DBCV Scores** *(−1 worst → 1 best)*")
                for cluster_id, score in sorted(dbcv_scores.items()):
                    # Colour-code: green ≥ 0.5, orange 0–0.5, red < 0
                    if score >= 0.5:
                        colour = "green"
                    elif score >= 0:
                        colour = "orange"
                    else:
                        colour = "red"
                    st.markdown(
                        f"Cluster {cluster_id}: "
                        f"<span style='color:{colour}; font-weight:bold'>{score:.4f}</span>",
                        unsafe_allow_html=True
                    )

        # ── Best representative document per cluster ───────────────────
        st.markdown("---")
        best_docs = clustering.get_best_doc_per_cluster(doc_names, coords_2d, cluster_labels)
        display_best_docs(best_docs, cluster_labels, doc_names, documents, clustering.lda_keywords)

        # ── Per-document similarity analysis ──────────────────────────
        st.markdown("---")
        st.header("📄 Document Similarity Analysis 📄")
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
                tfidf_matrix,
                cluster_labels,
                processed_texts
            )
            display_similarity_metrics(similarity_df, selected_doc)

        # ── Export ────────────────────────────────────────────────────
        export_results(documents, processed_texts, cluster_labels, tfidf_matrix, clustering)


if __name__ == "__main__":
    main()
