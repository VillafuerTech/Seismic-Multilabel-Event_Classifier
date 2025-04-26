# Project ML2024: Multi-Label Classification of Seismic Signals

## Overview

This repository implements a full end-to-end data science pipeline for multi-label classification of seismic signals captured in `.at2` format. The objective is to preprocess raw seismic data, extract spectral features using FFT, train and evaluate multiple machine learning models, and deploy the optimal model for production use.

## Repository Structure

```
├── data
│   ├── raw/           # Original `.at2` data files
│   ├── interim/       # Intermediate data after preprocessing
│   └── processed/     # Final feature matrices and labels (.csv/.npy)
├── notebooks/         # Jupyter Notebooks organized by project stage
├── src/               # Modular Python source code (preprocessing, features, modeling)
├── tests/             # Unit tests for project code
├── outputs/           # Executed notebooks, model artifacts, and visualizations
├── environment.yml    # Conda environment specification
├── Makefile           # Automation tasks (environment setup, notebook execution)
└── .gitignore         # Git ignore rules for unnecessary files
```

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/) installed
- Python 3.12 (specified in `environment.yml`)

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/ml2024-seismic-classification.git
   cd ml2024-seismic-classification
   ```
2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate ml_project
   ```

## Usage

### Execute All Notebooks

Run the complete pipeline and generate executed notebook outputs:

```bash
make run-notebooks
```

This command will produce executed versions of each notebook in the `outputs/` directory, suffixed with `_output.ipynb`.

## Testing

Ensure core functionalities behave as expected by running unit tests:

```bash
pytest --maxfail=1 --disable-warnings -q
```

## Notebook Descriptions

1. **01_Data_Preprocessing.ipynb**: Data ingestion from `.at2`, cleanup, and normalization.
2. **02_Exploratory_Data_Analysis.ipynb**: Exploratory data analysis and visualization of raw signals and label distributions.
3. **03_Feature_Engineering.ipynb**: FFT computation and construction of feature vectors.
4. **04_Baseline_Model.ipynb**: Implementation and evaluation of a baseline classifier.
5. **05_SVM_Model.ipynb**: Training and evaluation of an SVM with RBF kernel.
6. **06_Random_Forest_Model.ipynb**: Training and evaluation of a Random Forest multi-label classifier.
7. **07_Neural_Network_Model.ipynb**: Implementation of a Feedforward or 1D-CNN model for multi-label classification.
8. **08_Model_Comparison.ipynb**: Systematic comparison of model performance metrics and runtime.
9. **09_Results_Visualization.ipynb**: In-depth analysis of best model results, error cases, and learning curves.
10. **10_Final_Report_and_Deployment.ipynb**: Executive summary, model export, and deployment guidelines.

## Artifacts

- **Model artifacts**: Serialized final model (`model.joblib`) or TensorFlow `saved_model/` directory.
- **Visualizations**: Performance plots and comparison charts.
- **Reports**: Optional HTML/PDF export of report notebooks via `nbconvert`.

## Contributing

Contributions are welcome. Please open issues for bug reports or feature requests, and submit pull requests for enhancements.

## License

This project is intended for academic purposes and is not licensed for commercial use.
