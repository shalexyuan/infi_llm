import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import clip
import sys
import os

# Add parent directory to path to import constants
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import mp3d_category, rooms


def compute_clip_text_embeddings(categories, model_name='ViT-B/32'):
    """
    Compute text embeddings for categories using CLIP model.
    
    Args:
        categories: List of category names
        model_name: CLIP model name
    
    Returns:
        embeddings: numpy array of shape (n_categories, embedding_dim)
    """
    print(f"Loading CLIP model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    
    print(f"Computing CLIP embeddings for {len(categories)} categories...")
    
    # Tokenize text
    text_inputs = clip.tokenize(categories).to(device)
    
    # Get text features
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Convert to numpy
    embeddings = text_features.cpu().numpy()
    
    print(f"CLIP embeddings shape: {embeddings.shape}")
    return embeddings


def compute_similarity_matrix(embeddings):
    """
    Compute cosine similarity matrix between embeddings.
    
    Args:
        embeddings: numpy array of shape (n_categories, embedding_dim)
    
    Returns:
        similarity_matrix: numpy array of shape (n_categories, n_categories)
    """
    similarity_matrix = np.dot(embeddings, embeddings.T)
    return similarity_matrix


def plot_similarity_matrix(similarity_matrix, categories, title="CLIP Text Embedding Similarity Matrix"):
    """
    Plot the similarity matrix as a heatmap.
    
    Args:
        similarity_matrix: numpy array of similarity scores
        categories: list of category names
        title: title for the plot
    """
    plt.figure(figsize=(16, 14))
    
    # Create heatmap with better formatting
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)  # Mask upper triangle
    
    sns.heatmap(
        similarity_matrix,
        xticklabels=categories,
        yticklabels=categories,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        center=0.5,
        square=True,
        mask=mask,  # Only show lower triangle
        cbar_kws={'label': 'CLIP Cosine Similarity', 'shrink': 0.8}
    )
    
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Categories', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    return plt.gcf()


def analyze_similarities(similarity_matrix, categories, top_k=10):
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
    
    print(f"\n=== TOP {top_k} MOST SIMILAR PAIRS (CLIP) ===")
    for i, idx in enumerate(top_similar_indices):
        row, col = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
        similarity = similarities[idx]
        print(f"{i+1:2d}. {categories[row]:<15} <-> {categories[col]:<15}: {similarity:.3f}")
    
    # Get indices of least similar pairs
    least_similar_indices = np.argsort(similarities)[:top_k]
    
    print(f"\n=== TOP {top_k} LEAST SIMILAR PAIRS (CLIP) ===")
    for i, idx in enumerate(least_similar_indices):
        row, col = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
        similarity = similarities[idx]
        print(f"{i+1:2d}. {categories[row]:<15} <-> {categories[col]:<15}: {similarity:.3f}")


def find_semantic_clusters(similarity_matrix, categories, threshold=0.7):
    """
    Find semantic clusters based on similarity threshold.
    
    Args:
        similarity_matrix: numpy array of similarity scores
        categories: list of category names
        threshold: similarity threshold for clustering
    """
    n_categories = len(categories)
    visited = set()
    clusters = []
    
    for i in range(n_categories):
        if i in visited:
            continue
            
        cluster = [i]
        visited.add(i)
        
        for j in range(i + 1, n_categories):
            if j in visited:
                continue
                
            if similarity_matrix[i, j] >= threshold:
                cluster.append(j)
                visited.add(j)
        
        if len(cluster) > 1:  # Only keep clusters with more than one item
            clusters.append([categories[idx] for idx in cluster])
    
    print(f"\n=== SEMANTIC CLUSTERS (threshold={threshold}) ===")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {cluster}")
    
    return clusters


def compute_room_object_similarities(rooms, objects, model_name='ViT-B/32'):
    """
    Compute similarity matrix between rooms and objects.
    
    Args:
        rooms: List of room names
        objects: List of object names
        model_name: CLIP model name
    
    Returns:
        similarity_matrix: numpy array of shape (n_rooms, n_objects)
    """
    print(f"\n🏠 Computing room-object similarities...")
    print(f"Rooms: {len(rooms)}")
    print(f"Objects: {len(objects)}")
    
    # Get embeddings for both rooms and objects
    room_embeddings = compute_clip_text_embeddings(rooms, model_name)
    object_embeddings = compute_clip_text_embeddings(objects, model_name)
    
    # Compute cross-similarity matrix
    similarity_matrix = np.dot(room_embeddings, object_embeddings.T)
    
    print(f"Room-object similarity matrix shape: {similarity_matrix.shape}")
    return similarity_matrix


def plot_room_object_similarities(similarity_matrix, rooms, objects, title="Room-Object Similarity Matrix"):
    """
    Plot the room-object similarity matrix as a heatmap.
    
    Args:
        similarity_matrix: numpy array of similarity scores
        rooms: list of room names
        objects: list of object names
        title: title for the plot
    """
    plt.figure(figsize=(20, 12))
    
    # Create heatmap
    sns.heatmap(
        similarity_matrix,
        xticklabels=objects,
        yticklabels=rooms,
        annot=True,
        fmt='.2f',
        cmap='RdYlBu_r',
        center=0.5,
        cbar_kws={'label': 'CLIP Cosine Similarity', 'shrink': 0.8}
    )
    
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Objects', fontsize=14)
    plt.ylabel('Rooms', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    return plt.gcf()


def analyze_room_object_similarities(similarity_matrix, rooms, objects, top_k=10):
    """
    Analyze the most similar room-object pairs.
    
    Args:
        similarity_matrix: numpy array of similarity scores
        rooms: list of room names
        objects: list of object names
        top_k: number of top pairs to show
    """
    n_rooms, n_objects = similarity_matrix.shape
    
    # Flatten and get top similarities
    flat_similarities = similarity_matrix.flatten()
    top_indices = np.argsort(flat_similarities)[-top_k:][::-1]
    
    print(f"\n=== TOP {top_k} ROOM-OBJECT SIMILARITIES ===")
    for i, idx in enumerate(top_indices):
        room_idx = idx // n_objects
        object_idx = idx % n_objects
        similarity = flat_similarities[idx]
        print(f"{i+1:2d}. {rooms[room_idx]:<15} ↔ {objects[object_idx]:<15}: {similarity:.3f}")
    
    # Find most similar object for each room
    print(f"\n=== MOST SIMILAR OBJECT FOR EACH ROOM ===")
    for i, room in enumerate(rooms):
        best_object_idx = np.argmax(similarity_matrix[i])
        best_similarity = similarity_matrix[i, best_object_idx]
        print(f"{room:<15} → {objects[best_object_idx]:<15}: {best_similarity:.3f}")
    
    # Find most similar room for each object
    print(f"\n=== MOST SIMILAR ROOM FOR EACH OBJECT ===")
    for j, obj in enumerate(objects):
        best_room_idx = np.argmax(similarity_matrix[:, j])
        best_similarity = similarity_matrix[best_room_idx, j]
        print(f"{obj:<15} → {rooms[best_room_idx]:<15}: {best_similarity:.3f}")


def find_room_object_clusters(similarity_matrix, rooms, objects, threshold=0.6):
    """
    Find clusters of rooms and objects that are highly similar.
    
    Args:
        similarity_matrix: numpy array of similarity scores
        rooms: list of room names
        objects: list of object names
        threshold: similarity threshold for clustering
    """
    print(f"\n=== ROOM-OBJECT CLUSTERS (threshold={threshold}) ===")
    
    # Find high-similarity pairs
    high_sim_pairs = []
    for i, room in enumerate(rooms):
        for j, obj in enumerate(objects):
            if similarity_matrix[i, j] >= threshold:
                high_sim_pairs.append((room, obj, similarity_matrix[i, j]))
    
    # Sort by similarity
    high_sim_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"Found {len(high_sim_pairs)} high-similarity pairs:")
    for room, obj, sim in high_sim_pairs:
        print(f"  {room:<15} ↔ {obj:<15}: {sim:.3f}")
    
    return high_sim_pairs


def main():
    """
    Main function to run the CLIP similarity analysis.
    """
    print("MP3D Category CLIP Text Embedding Similarity Analysis")
    print("=" * 60)
    print(f"Categories: {mp3d_category}")
    print(f"Number of categories: {len(mp3d_category)}")
    print(f"Rooms: {rooms}")
    print(f"Number of rooms: {len(rooms)}")
    
    try:
        # 1. OBJECT-OBJECT SIMILARITY ANALYSIS
        print(f"\n{'='*60}")
        print("1. OBJECT-OBJECT SIMILARITY ANALYSIS")
        print(f"{'='*60}")
        
        # Compute CLIP embeddings for objects
        object_embeddings = compute_clip_text_embeddings(mp3d_category)
        
        # Compute object similarity matrix
        object_similarity_matrix = compute_similarity_matrix(object_embeddings)
        
        # Print basic statistics
        print(f"\n=== OBJECT SIMILARITY MATRIX STATISTICS ===")
        print(f"Matrix shape: {object_similarity_matrix.shape}")
        print(f"Mean similarity: {np.mean(object_similarity_matrix):.3f}")
        print(f"Std similarity: {np.std(object_similarity_matrix):.3f}")
        print(f"Min similarity: {np.min(object_similarity_matrix):.3f}")
        print(f"Max similarity: {np.max(object_similarity_matrix):.3f}")
        
        # Analyze object similarities
        analyze_similarities(object_similarity_matrix, mp3d_category)
        
        # Find object semantic clusters
        object_clusters = find_semantic_clusters(object_similarity_matrix, mp3d_category, threshold=0.6)
        
        # Plot object similarity matrix
        fig1 = plot_similarity_matrix(object_similarity_matrix, mp3d_category, 
                                    "Object-Object CLIP Similarity Matrix")
        plt.savefig('mp3d_category_clip_similarity_matrix.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved plot: mp3d_category_clip_similarity_matrix.png")
        plt.show()
        
        # 2. ROOM-OBJECT SIMILARITY ANALYSIS
        print(f"\n{'='*60}")
        print("2. ROOM-OBJECT SIMILARITY ANALYSIS")
        print(f"{'='*60}")
        
        # Compute room-object similarities
        room_object_similarity_matrix = compute_room_object_similarities(rooms, mp3d_category)
        
        # Print room-object statistics
        print(f"\n=== ROOM-OBJECT SIMILARITY MATRIX STATISTICS ===")
        print(f"Matrix shape: {room_object_similarity_matrix.shape}")
        print(f"Mean similarity: {np.mean(room_object_similarity_matrix):.3f}")
        print(f"Std similarity: {np.std(room_object_similarity_matrix):.3f}")
        print(f"Min similarity: {np.min(room_object_similarity_matrix):.3f}")
        print(f"Max similarity: {np.max(room_object_similarity_matrix):.3f}")
        
        # Analyze room-object similarities
        analyze_room_object_similarities(room_object_similarity_matrix, rooms, mp3d_category)
        
        # Find room-object clusters
        room_object_clusters = find_room_object_clusters(room_object_similarity_matrix, rooms, mp3d_category, threshold=0.5)
        
        # Plot room-object similarity matrix
        fig2 = plot_room_object_similarities(room_object_similarity_matrix, rooms, mp3d_category,
                                            "Room-Object CLIP Similarity Matrix")
        plt.savefig('room_object_clip_similarity_matrix.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved plot: room_object_clip_similarity_matrix.png")
        plt.show()
        
        # 3. ADDITIONAL ANALYSIS
        print(f"\n{'='*60}")
        print("3. ADDITIONAL ANALYSIS")
        print(f"{'='*60}")
        
        # Object-wise statistics
        print(f"\n=== OBJECT-WISE SIMILARITY STATISTICS ===")
        for i, category in enumerate(mp3d_category):
            similarities = object_similarity_matrix[i]
            similarities_no_self = np.concatenate([similarities[:i], similarities[i+1:]])
            print(f"{category:<15}: mean={np.mean(similarities_no_self):.3f}, "
                  f"max={np.max(similarities_no_self):.3f}, "
                  f"min={np.min(similarities_no_self):.3f}")
        
        # Room-wise statistics
        print(f"\n=== ROOM-WISE SIMILARITY STATISTICS ===")
        for i, room in enumerate(rooms):
            similarities = room_object_similarity_matrix[i]
            print(f"{room:<15}: mean={np.mean(similarities):.3f}, "
                  f"max={np.max(similarities):.3f}, "
                  f"min={np.min(similarities):.3f}")
        
        print(f"\n🎉 Analysis complete! Generated plots:")
        print(f"  • mp3d_category_clip_similarity_matrix.png")
        print(f"  • room_object_clip_similarity_matrix.png")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Make sure you have CLIP installed: pip install clip-by-openai")


if __name__ == "__main__":
    main()
