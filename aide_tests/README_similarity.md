# Text Embedding Similarity Analysis for MP3D Categories

This directory contains scripts to analyze text embedding similarities between MP3D object categories using different embedding methods.

## Files

1. **`test_simple_sim.py`** - Simple TF-IDF based similarity (no external dependencies)
2. **`test_clip_sim.py`** - Sentence transformer based similarity 
3. **`test_clip_sim_clip.py`** - CLIP model based similarity
4. **`requirements_similarity.txt`** - Required packages

## Quick Start

### Option 1: Simple TF-IDF Analysis (No Installation Required)
```bash
cd aide_tests
python test_simple_sim.py
```

### Option 2: Sentence Transformer Analysis
```bash
pip install sentence-transformers scikit-learn matplotlib seaborn
python test_clip_sim.py
```

### Option 3: CLIP Analysis (Best Quality)
```bash
pip install clip-by-openai torch matplotlib seaborn scikit-learn
python test_clip_sim_clip.py
```

## What Each Script Does

### 1. **test_simple_sim.py**
- Uses TF-IDF vectorization (no external ML models)
- Fast and lightweight
- Good for basic text similarity analysis
- Output: `mp3d_category_tfidf_similarity_matrix.png`

### 2. **test_clip_sim.py** 
- Uses sentence transformer models (all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)
- Better semantic understanding than TF-IDF
- Tests multiple models and saves separate plots
- Output: Multiple similarity matrix plots

### 3. **test_clip_sim_clip.py**
- Uses CLIP model for text embeddings
- Best semantic understanding
- Finds semantic clusters
- Output: `mp3d_category_clip_similarity_matrix.png`

## Output Analysis

Each script provides:

1. **Similarity Matrix Heatmap**: Visual representation of category similarities
2. **Top Similar Pairs**: Most semantically similar category pairs
3. **Top Dissimilar Pairs**: Least similar category pairs  
4. **Category Statistics**: Mean, max, min similarities per category
5. **Semantic Clusters**: Groups of similar categories (CLIP version)

## Expected Results

### High Similarity Pairs (Expected):
- `chair` ↔ `stool` (both seating)
- `sofa` ↔ `couch` (same object)
- `bed` ↔ `seating` (furniture)
- `plant` ↔ `picture` (decorative)

### Low Similarity Pairs (Expected):
- `toilet` ↔ `picture` (very different functions)
- `shower` ↔ `chair` (different room types)
- `gym equipment` ↔ `towel` (different contexts)

## MP3D Categories Analyzed

The scripts analyze these 22 categories from `constants.py`:

```
['chair', 'table', 'picture', 'cabinet', 'cushion', 'couch', 'bed', 
 'drawer', 'plant', 'sink', 'toilet', 'stool', 'towel', 'tv', 
 'shower', 'bathtub', 'counter', 'fireplace', 'gym equipment', 
 'seating', 'clothes', 'background']
```

## Usage in Multi-Robot Systems

This similarity analysis can be used for:

1. **Object Grouping**: Group semantically similar objects
2. **Task Assignment**: Assign robots to similar object types
3. **Semantic Navigation**: Use similarity for navigation decisions
4. **Object Recognition**: Improve object detection accuracy

## Troubleshooting

### Common Issues:

1. **CUDA/GPU Issues**: CLIP script will fallback to CPU if CUDA unavailable
2. **Memory Issues**: Use smaller models or reduce batch size
3. **Installation Issues**: Use conda instead of pip for better dependency management

### Performance Tips:

- Use `test_simple_sim.py` for quick analysis
- Use `test_clip_sim_clip.py` for best quality results
- Adjust `top_k` parameter to see more/fewer similar pairs
- Modify `threshold` in clustering for different cluster sizes
