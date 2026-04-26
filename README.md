
# DATASET ADVENTURE KID WAVEFORM (AKWF)  - Data Analysis & Preprocessing

## Dataset Overview
This project utilizes the [AKWF dataset](https://github.com/KristofferKarlAxelEkstrand/AKWF-FREE), a collection of thousands of single-cycle waveforms. This version is enhanced by pairing each audio sample with a corresponding metadata file.

The complete dataset (Audio + JSON labels) is available here:  
[**Download Processed Dataset**](https://drive.google.com/uc?id=13UhP_6tccMgPfv9-LF4zKeCGdrdPEgx8)

### Technical Specifications:
* **Format:** File pairs of `.wav` (Audio) and `.json` (Metadata/Features)
* **Sample Length:** 600 samples (Fixed)
* **Bit Depth:** 16-bit
* **Sample Rate:** 44.1 kHz
* **Channels:** Mono

---

## Dataset Structure
The dataset is organized as a collection of file pairs. For every audio sample, there is a matching JSON file containing rich descriptors.

**Example:**
- `AKWF_0001.wav` – Raw single-cycle waveform.
- `AKWF_0001_analysis.json` – Extracted features and metadata.

### Key JSON Features:
* **Technical:** `duration`, `samplerate`, `bitrate`, `codec`, `filesize`.
* **Tonal/Pitch:** `tonality`, `note_name`, `note_frequency`, `note_confidence`.
* **Spectral:** `SpectralCentroid`, `SpectralComplexity`, `HNR`, `Dissonance`.
* **Psychoacoustic:** `warmth`, `brightness`, `roughness`, `hardness`, `sharpness`, `boominess`.
* **Synthesis (DCO):** `dco_brightness`, `dco_richness`, `dco_oddenergy`, `dco_zcr`.

---

The primary goal is to prepare a hybrid dataset for a **Conditional Variational Autoencoder (CVAE)**. The model is designed to receive a dual input:
1. **Raw Audio Data:** The 600-sample single-cycle waveforms.
2. **Selected JSON Attributes:** A curated set of features used as conditioning labels.

---

## Analysis & Feature Selection
Since each waveform is paired with a JSON file containing over 40 attributes, a significant part of this project is dedicated to **Feature Selection**. Not all attributes are equally relevant for generative modeling; therefore, we analyze which features provide the most meaningful control over the synthesized sound.

### Selection Criteria:
* **Correlation Analysis:** Identifying which spectral features (like `SpectralCentroid`) best represent subjective qualities like `brightness`.
* **Variance & Distribution:** Selecting features with high variance that clearly distinguish different waveform categories.
* **Redundancy Reduction:** Removing highly correlated features to prevent bias and simplify the model's latent space.

### Available Feature Categories for Selection:
* **Spectral:** `SpectralCentroid`, `SpectralComplexity`, `HNR`, `Dissonance`.
* **Psychoacoustic:** `warmth`, `brightness`, `roughness`, `hardness`, `sharpness`.
* **Tonal/Pitch:** `tonality`, `note_frequency`, `note_confidence`.
* **Synthesis (DCO):** `dco_brightness`, `dco_richness`, `dco_oddenergy`.

---

## Data Preparation Pipeline
This repository contains the logic for:
1. **Multimodal Pairing:** Aligning `.wav` audio files with their corresponding `.json` metadata.
2. **Normalization:** Implementing `ess_yeojohnson` scaling to prepare both audio and numerical features for neural network input.
3. **Filtering:** Scripts to down-select the most relevant features based on the analysis results.
4. **Data Formatting:** Converting the selected hybrid data into a format (e.g., NumPy arrays or Tensors) ready for the CVAE training loop.

