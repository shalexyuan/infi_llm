import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from constants import mp3d_category


def compute_tfidf_embeddings(categories):
    """
    Compute TF-IDF embeddings for categories.
    
    Args:
        categories: List of category names
    
    Returns:
        embeddings: numpy array of shape (n_categories, embedding_dim)
    """
    print(f"Computing TF-IDF embeddings for {len(categories)} categories...")
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),  # Use unigrams and bigrams
        max_features=1000
    )
    
    # Fit and transform
    embeddings = vectorizer.fit_transform(categories).toarray()
    
    print(f"TF-IDF embeddings shape: {embeddings.shape}")
    return embeddings


def compute_similarity_matrix(embeddings):
    """
    Compute cosine similarity matrix between embeddings.
    
    Args:
        embeddings: numpy array of shape (n_categories, embedding_dim)
    
    Returns:
        similarity_matrix: numpy array of shape (n_categories, n_categories)
    """
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix


def plot_similarity_matrix(similarity_matrix, categories, title="Text Similarity Matrix (TF-IDF)"):
    """
    Plot the similarity matrix as a heatmap.
    
    Args:
        similarity_matrix: numpy array of similarity scores
        categories: list of category names
        title: title for the plot
    """
    plt.figure(figsize=(14, 12))
    
    # Create heatmap
    sns.heatmap(
        similarity_matrix,
        xticklabels=categories,
        yticklabels=categories,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        cbar_kws={'label': 'Cosine Similarity', 'shrink': 0.8}
    )
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Categories', fontsize=12)
    plt.ylabel('Categories', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return plt.gcf()


def analyze_similarities(similarity_matrix, categories, top_k=8):
    """
    Analyze the most similar and dissimilar category pairs.
    
    Args:
        similarity_matrix: numpy array of similarity scores
        categories: list of category names
        top_k: number of top pairs to show
    """
    n_categories = len(categories)
    
    # Get upper triangle indices (excluding diagonal)
    upper_tri_indices = np.triu_indices(n_categories, k=1)
    similarities = similarity_matrix[upper_tri_indices]
    
    # Get indices of top similarities
    top_similar_indices = np.argsort(similarities)[-top_k:][::-1]
    
    print(f"\n=== TOP {top_k} MOST SIMILAR PAIRS ===")
    for i, idx in enumerate(top_similar_indices):
        row, col = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
        similarity = similarities[idx]
        print(f"{i+1:2d}. {categories[row]:<15} <-> {categories[col]:<15}: {similarity:.3f}")
    
    # Get indices of least similar pairs
    least_similar_indices = np.argsort(similarities)[:top_k]
    
    print(f"\n=== TOP {top_k} LEAST SIMILAR PAIRS ===")
    for i, idx in enumerate(least_similar_indices):
        row, col = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
        similarity = similarities[idx]
        print(f"{i+1:2d}. {categories[row]:<15} <-> {categories[col]:<15}: {similarity:.3f}")


def main():
    """
    Main function to run the similarity analysis.
    """
    print("MP3D Category Text Similarity Analysis (TF-IDF)")
    print("=" * 55)
    print(f"Categories: {mp3d_category}")
    print(f"Number of categories: {len(mp3d_category)}")
    
    # Compute TF-IDF embeddings
    embeddings = compute_tfidf_embeddings(mp3d_category)
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    # Print basic statistics
    print(f"\n=== SIMILARITY MATRIX STATISTICS ===")
    print(f"Matrix shape: {similarity_matrix.shape}")
    print(f"Mean similarity: {np.mean(similarity_matrix):.3f}")
    print(f"Std similarity: {np.std(similarity_matrix):.3f}")
    print(f"Min similarity: {np.min(similarity_matrix):.3f}")
    print(f"Max similarity: {np.max(similarity_matrix):.3f}")
    
    # Analyze similarities
    analyze_similarities(similarity_matrix, mp3d_category)
    
    # Plot similarity matrix
    fig = plot_similarity_matrix(similarity_matrix, mp3d_category)
    plt.savefig('mp3d_category_tfidf_similarity_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: mp3d_category_tfidf_similarity_matrix.png")
    plt.show()
    
    # Category-wise analysis
    print(f"\n=== CATEGORY-WISE SIMILARITY STATISTICS ===")
    for i, category in enumerate(mp3d_category):
        similarities = similarity_matrix[i]
        similarities_no_self = np.concatenate([similarities[:i], similarities[i+1:]])
        print(f"{category:<15}: mean={np.mean(similarities_no_self):.3f}, "
              f"max={np.max(similarities_no_self):.3f}, "
              f"min={np.min(similarities_no_self):.3f}")


if __name__ == "__main__":
    main()
