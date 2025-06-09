<div align="center">
  <h1>Beyond Accuracy: Metrics that Uncover what makes a 'Good' Visual Descriptor</h1>
    <p>
        <a href="https://ethan-y-lin.github.io/beyond-accuracy-project-page/">
        <img src="https://ethan-y-lin.github.io/beyond-accuracy-project-page/static/images/fig_concept_main.png" alt="Beyond Accuracy Teaser" width="800"/>
        </a>
        <!-- add a caption -->
        <br>
        Achieving high accuracy does not guarantee that a set of visual descriptors is "good". Other factors like interpretability may suffer. Global Alignment and CLIP Similarity can serve as new metrics for evaluating and understanding different sets of visual descriptors.
    </p>
</div>

More details can be found on the [project page](https://ethan-y-lin.github.io/beyond-accuracy-project-page/).

## Getting started
### 1. Clone the repository
```bash
git clone https://github.com/ethan-y-lin/beyond-accuracy.git
cd beyond-accuracy
```
### 2. Create and Activate Environment and Install Dependencies
```bash
conda create --name beyond-acc python=3.12
conda activate beyond-acc
conda env update --file environment.yml --prune
```
### 3. Set Environment Variables
Set paths to relevant directories.
```bash
cp .env.example .env
```
### 4. Downloading Datasets

## Computing Global Alignment Metrics and Accuracy
```bash
python dino_align/scripts/cache_all_embeddings
python dino_align/scripts/compute_accuracies --descriptors_path "descriptors/class_names"
python dino_align/scripts/compute_alignments --descriptors_path "descriptors/no_class_names"
```
## Computing CLIP Similarity Metrics
```bash
python clip_sim/main.py
```

## General Structure / Useful Files

- `beyond_accuracy/`
    - `dino_align/`
      - `eval/`: Contains functions to compute alignment and accuracy.
      - `scripts`: Scripts that compute Global Alignment and accuracy metrics.
    - `clip_sim/`
      - `main.py`: Script to compute CLIP Similarity metrics.
    - `utils/`
      - `get_embeddings.py`: Functions to generate and cache image and text embeddings.
      - `metrics.py`: Metric file from Platonic Representation Hypothesis paper.
    - `data/`: Contains class files for each dataset.
    - `descriptors/`: Contains the descriptors for each dataset and for each descriptor set type.
      - `class_names`: Contains the original descriptors. 
      - `no_class_names`: Contains descriptors with class names removed.
    - `cache/`: This is a cache for the image/text embeddings for all the datasets.
    - `config.py`: File that instantiates global variables for important paths and parameters.
