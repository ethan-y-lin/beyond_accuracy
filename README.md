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
1. **CUB Download**: Download at this link: https://www.vision.caltech.edu/datasets/cub_200_2011/
2. **NABirds Download**: Use bash command
```bash
curl -L -A "Mozilla/5.0" "https://www.dropbox.com/scl/fi/yas70u9uzkeyzrmrfwcru/nabirds.tar.gz?rlkey=vh0uduhckom5jyp73igjugqtr&dl=1" -o nabirds.tar.gz
```
3. **CIFAR100 Download:** Set the download flag to True in data/cifar100/cifar100.py.

### 5. Download CLIP Training Data
1. Clone the datacomp repo: https://github.com/mlfoundations/datacomp
```bash
git clone https://github.com/mlfoundations/datacomp.git
```
2. Set the METADATA_DIR in .env: /path_to_datacomp/datacomp/commonpool_small/metadata/
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
