
# DATASET ADVENTURE KID WAVEFORM (AKWF)  - Data Analysis & Preprocessing

## Project Overview
This repository is focused on analyzing and preprocessing the Adventure Kid Waveform (AKWF) dataset for use with a Conditional Variational Autoencoder (CVAE).

The dataset includes single-cycle waveforms paired with JSON metadata describing technical, spectral, tonal and psychoacoustic features.

## Repository Structure
- `dash.py` — Streamlit dashboard for browsing and analyzing the preprocessed CVAE dataset.
- `json_analysis.ipynb` — Jupyter notebook for dataset download, loading, cleanup, merging, integrity checks and feature exploration.
- `data/` — Local dataset folder. It is initially empty and is populated by the notebook download step.
- `.gitignore` — Ignore rules for local artifacts.

## Data Location
The notebook downloads and extracts the dataset into:
- `data/AKWF_44k1_600s/AKWF_44k1_600s/`

After running the notebook, this folder will contain:
- `.wav` files for raw waveform data
- `_analysis.json` files for extracted metadata and feature vectors

## Dataset Overview
This project utilizes the [AKWF dataset](https://github.com/KristofferKarlAxelEkstrand/AKWF-FREE), a collection of thousands of single-cycle waveforms. The repository version is enhanced by pairing each audio sample with a corresponding metadata file.

**Example data pair:**
- `AKWF_0001.wav` — Raw single-cycle waveform
- `AKWF_0001_analysis.json` — Extracted features and metadata

### Technical Specifications
* **Format:** `.wav` audio + `.json` metadata
* **Sample Length:** 600 samples
* **Sample Rate:** 44.1 kHz
* **Channels:** Mono

## Notebook Workflow (`json_analysis.ipynb`)
The notebook performs the following steps:
1. Download and extract the processed AKWF dataset from Google Drive.
2. Load JSON label files into a Pandas DataFrame.
3. Load WAV files using `torchaudio` and align them by sample name.
4. Merge audio and metadata into a single dataset.
5. Run data integrity checks for missing values, duplicates, data types, sample rate, duration, and waveform length.
6. Visualize outliers and normalized waveforms.
7. Export the cleaned dataset for downstream use.

## Key Features and Analysis
The dataset contains both technical and perceptual descriptors. Relevant JSON feature categories include:
* **Technical:** `duration`, `samplerate`, `bitrate`, `codec`, `filesize`
* **Tonal / Pitch:** `tonality`, `note_frequency`, `note_confidence`
* **Spectral:** `SpectralCentroid`, `SpectralSpread`, `SpectralKurtosis`, `SpectralComplexity`, `OddToEvenHarmonicEnergyRatio`, `Dissonance`, `PitchSalience`, `HNR`
* **Psychoacoustic:** `warmth`, `brightness`, `roughness`, `hardness`, `sharpness`, `boominess`

Feature selection is driven by:
* correlation analysis
* variance and distribution
* redundancy reduction

## Streamlit Dashboard (`dash.py`)
The dashboard loads a preprocessed dataset file named `dataset_cvae.pkl` and provides:
* interactive latent-space visualization
* axis selection for acoustic features
* filtering by feature ranges
* sample waveform preview and metrics
* correlation guidance for feature pair selection

### Running the dashboard
```bash
streamlit run dash.py
```

> Note: `dataset_cvae.pkl` is expected to be generated from the notebook export step.


## Setup
Create a dedicated Conda environment with Python 3.8.5 and install the project requirements:

```bash
conda create -n 'env_name' python=3.8.5 -y
conda activate 'env_name'
pip install -r requirements.txt
```
> If you prefer Conda-only installation, install `python=3.8.5` first and use `pip install -r requirements.txt` for the Python packages.

## Usage
1. Open `json_analysis.ipynb` and run the preprocessing pipeline.
2. Export the cleaned dataset to `dataset_cvae.pkl`.
3. Launch the dashboard with `streamlit run dash.py`.

## Notes
This repository is aimed at creating a clean, label-rich dataset for CVAE-based sound generation and controllable wavetable synthesis.



