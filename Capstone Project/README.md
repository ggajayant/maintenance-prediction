# Predictive Maintenance Capstone

The notebook includes the interim analysis and the final deployment steps in one file.
It follows the Week 3 Streamlit Cloud case study and the tourism project notebook.

- Data: `data/engine_data.csv`
- Notebook: `Capstone Project.ipynb`
- Registration, preparation and training: `scripts/data_register.py`, `scripts/prep.py`, `scripts/train.py`
- Streamlit app: `app.py`
- GitHub Actions: repository-root `.github/workflows/pipeline.yml`

Use Python 3.11. Install `requirements-training.txt` for the notebook and training.
Run `streamlit run app.py` from the project folder to try the app locally.

The notebook's GitHub section asks for a token using a masked prompt, uploads the
listed files and prints each result. The token needs repository contents and
workflow write access. Never put the token in notebook source or a Git URL.

Repository: https://github.com/ggajayant/maintenance-prediction
For Streamlit Community Cloud, use branch `main`, app path `Capstone Project/app.py`
and Python 3.11. Hosting and successful workflow execution still require verification.

Submission files are in `output/pdf/`: one combined PDF report and its executed HTML notebook.
The learner-name placeholder still needs to be replaced before submission.
