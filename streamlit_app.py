# Medium Document: https://medium.com/@mtshomsky/document-similarity-clustering-23638d3aa65c

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

class DocumentClustering:
    def __init__(self, max_clusters=3):
        self.max_clusters = max_clusters
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.kmeans = None  # Will be initialized in cluster_documents
        
    def cluster_documents(self, processed_texts):
        # Vectorize the texts
        vectors = self.vectorizer.fit_transform(processed_texts)
        
        # Dynamically set number of clusters based on input size
        n_samples = len(processed_texts)
        n_clusters = min(self.max_clusters, n_samples)
        
        # Initialize KMeans with dynamic number of clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(vectors)
        
        # Get cluster centers
        cluster_centers = self.kmeans.cluster_centers_
        
        return vectors, cluster_labels, cluster_centers
    
    def get_cluster_statistics(self, cluster_labels):
        unique_labels = np.unique(cluster_labels)
        stats = {}
        
        for label in unique_labels:
            count = np.sum(cluster_labels == label)
            stats[f"Cluster {label}"] = {
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

            # HDBSCAN labels -1 = noise; no valid centroid to compare against
            if cluster_label < 0:
                return 0.0

            centroid = self.kmeans.cluster_centers_[cluster_label]

            # Check if centroid is zero vector
            if np.all(np.abs(centroid) < 1e-10):
                return 0.0

            # Ensure vector is 2D for cosine_similarity (works for both sparse and dense)
            if hasattr(vector, 'getnnz'):
                if vector.getnnz() == 0:
                    return 0.0
                vec_2d = vector
            else:
                vec_2d = vector.reshape(1, -1) if vector.ndim == 1 else vector

            similarity = cosine_similarity(vec_2d, centroid.reshape(1, -1))[0][0]

            # Handle NaN values
            return float(similarity) if not np.isnan(similarity) else 0.0
        except Exception:
            return 0.0

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
                def to_2d(v):
                    if hasattr(v, 'getnnz'):
                        return v  # sparse matrix row is already 2D
                    return v.reshape(1, -1) if v.ndim == 1 else v

                tv = to_2d(target_vector)
                ov = to_2d(vectors[idx])

                # Cosine similarity with error handling
                try:
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
        
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from nltk.util import ngrams
import string
import re

class TextPreprocessor:
    def __init__(self):
        # Download required NLTK resources
        nltk_resources = ['punkt', 'stopwords', 'wordnet']
        for resource in nltk_resources:
            try:
                print(f"Checking NLTK resource: {resource}")
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Error downloading {resource}: {str(e)}")
                raise
        
        # Initialize stemmers and lemmatizer
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.porter_stemmer = PorterStemmer()
            self.snowball_stemmer = SnowballStemmer('english')
            self.tweet_tokenizer = TweetTokenizer()
        except Exception as e:
            print(f"Error initializing text processing components: {str(e)}")
            raise
        
        # Default preprocessing options
        self.config = {
            'lowercase': True,
            'remove_numbers': True,
            'remove_punctuation': True,
            'remove_whitespace': True,
            'remove_urls': True,
            'remove_emails': True,
            'tokenizer': 'word',  # Options: 'word', 'sentence', 'tweet'
            'stemmer': 'lemmatizer',  # Options: 'lemmatizer', 'porter', 'snowball', None
            'remove_stopwords': True,
            'custom_stopwords': set(),
            'min_word_length': 2,
            'max_word_length': 100,
            'ngram_range': (1, 1)  # (min_n, max_n)
        }
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"Error loading stopwords: {str(e)}")
            self.stop_words = set()
            raise
        
    def set_config(self, **kwargs):
        """Update preprocessing configuration"""
        self.config.update(kwargs)
        if 'custom_stopwords' in kwargs:
            try:
                self.stop_words = set(stopwords.words('english')).union(kwargs['custom_stopwords'])
            except Exception as e:
                print(f"Error updating stopwords: {str(e)}")
                raise
    
    def clean_text(self, text):
        """Basic text cleaning"""
        try:
            if self.config['lowercase']:
                text = text.lower()
            
            if self.config['remove_urls']:
                text = re.sub(r'http\S+|www\S+|https\S+', '', text)
                
            if self.config['remove_emails']:
                text = re.sub(r'\S+@\S+', '', text)
                
            if self.config['remove_numbers']:
                text = re.sub(r'\d+', '', text)
                
            if self.config['remove_punctuation']:
                text = text.translate(str.maketrans('', '', string.punctuation))
                
            if self.config['remove_whitespace']:
                text = ' '.join(text.split())
                
            return text
        except Exception as e:
            print(f"Error in text cleaning: {str(e)}")
            raise
    
    def tokenize(self, text):
        """Tokenize text based on selected tokenizer"""
        try:
            if self.config['tokenizer'] == 'sentence':
                return sent_tokenize(text)
            elif self.config['tokenizer'] == 'tweet':
                return self.tweet_tokenizer.tokenize(text)
            else:  # word tokenizer
                return word_tokenize(text)
        except Exception as e:
            print(f"Error in tokenization: {str(e)}")
            raise
    
    def apply_stemming(self, token):
        """Apply selected stemming/lemmatization method"""
        try:
            if self.config['stemmer'] == 'porter':
                return self.porter_stemmer.stem(token)
            elif self.config['stemmer'] == 'snowball':
                return self.snowball_stemmer.stem(token)
            elif self.config['stemmer'] == 'lemmatizer':
                return self.lemmatizer.lemmatize(token)
            return token
        except Exception as e:
            print(f"Error in stemming/lemmatization: {str(e)}")
            raise
    
    def generate_ngrams(self, tokens):
        """Generate n-grams from tokens"""
        try:
            min_n, max_n = self.config['ngram_range']
            all_ngrams = []
            for n in range(min_n, max_n + 1):
                all_ngrams.extend([' '.join(gram) for gram in ngrams(tokens, n)])
            return all_ngrams
        except Exception as e:
            print(f"Error generating n-grams: {str(e)}")
            raise
    
    def preprocess(self, text):
        """Main preprocessing pipeline"""
        try:
            # Clean text
            text = self.clean_text(text)
            
            # Tokenize
            tokens = self.tokenize(text)
            
            # Filter tokens
            tokens = [token for token in tokens 
                     if self.config['min_word_length'] <= len(token) <= self.config['max_word_length']]
            
            # Remove stopwords
            if self.config['remove_stopwords']:
                tokens = [token for token in tokens if token not in self.stop_words]
            
            # Apply stemming/lemmatization
            if self.config['stemmer']:
                tokens = [self.apply_stemming(token) for token in tokens]
            
            # Generate n-grams if required
            if self.config['ngram_range'] != (1, 1):
                tokens = self.generate_ngrams(tokens)
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error in preprocessing pipeline: {str(e)}")
            raise
            
def load_sample_data():
    """Load sample documents for demonstration"""
    return [
        {
            "name": "Sample1",
            "content": """Machine learning is a subset of artificial intelligence 
                         that focuses on training algorithms to learn from data."""
        },
        {
            "name": "Sample2",
            "content": """Deep learning is a type of machine learning that uses 
                         neural networks with multiple layers."""
        },
        {
            "name": "Sample3",
            "content": """Data science combines statistics, programming, and 
                         domain expertise to extract insights from data."""
        },
        {
            "name": "Sample4",
            "content": """Artificial intelligence aims to create systems that 
                         can mimic human intelligence and decision-making."""
        },
        {
            "name": "Sample5",
            "content": """Statistical analysis involves collecting, organizing, 
                         and interpreting numerical data to find patterns."""
        }
    ]
    
import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd

class ClusterVisualizer:
    def plot_clusters(self, vectors, cluster_labels, doc_names):
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2)
        coords = pca.fit_transform(vectors.toarray())
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'cluster': [f'Cluster {label}' for label in cluster_labels],
            'document': doc_names
        })
        
        # Create interactive scatter plot
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='cluster',
            hover_data=['document'],
            title='Document Clusters Visualization',
            labels={'x': 'First Principal Component', 
                   'y': 'Second Principal Component'}
        )
        
        return fig
        
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
                
                # Calculate centroid similarity — guard against HDBSCAN noise label (-1)
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
        vectors, cluster_labels, cluster_centers = clustering.cluster_documents(processed_texts)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Cluster Visualization")
            fig = visualizer.plot_clusters(vectors, cluster_labels, doc_names)
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
