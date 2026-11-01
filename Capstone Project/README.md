# Predictive Maintenance Capstone

The notebook includes the interim analysis and the final deployment steps in one file.
It follows the Week 3 Streamlit Cloud case study and the tourism project notebook.

- Data: `data/engine_data.csv`
- Notebook: `Capstone Project.ipynb`
- Registration, preparation and training: `scripts/data_register.py`, `scripts/prep.py`, `scripts/train.py`
- Streamlit app: `app.py`
- App dependencies: repository-root `requirements.txt`
- GitHub Actions: repository-root `.github/workflows/pipeline.yml`

Use Python 3.11. Install `requirements-training.txt` for the notebook and training.
Run `streamlit run app.py` from the project folder to try the app locally.

The notebook's GitHub section asks for a token using a masked prompt, uploads the
listed files and prints each result. The token needs repository contents and
workflow write access. Never put the token in notebook source or a Git URL.

Repository: https://github.com/ggajayant/maintenance-prediction

Live Streamlit app: [https://maintenance-prediction-qmugu29g65zgsgrrdtjdgu.streamlit.app/](https://maintenance-prediction-qmugu29g65zgsgrrdtjdgu.streamlit.app/)

Deployment details: branch `main`, app path `Capstone Project/app.py`, Python 3.11.
The supplied app URL responds over HTTPS. Add a screenshot of the hosted app with a
prediction result for the submission.

Successful workflow: https://github.com/ggajayant/maintenance-prediction/actions/runs/41293696874

Submission files are in `output/pdf/`: one combined PDF report and its executed HTML notebook.
The learner-name placeholder still needs to be replaced before submission.

## Streamlit requirements-path fix

The cloud installer split the space in `Capstone Project/requirements.txt`.
Runtime dependencies now live in repository-root `requirements.txt`.
The training file includes `../requirements.txt`.

Rerun the notebook file-writing cells for requirements and workflow, then its
GitHub authentication and upload cells. Upload removes the old requirements file
beside the app, so Community Cloud can find the root copy. Reboot the app after
upload succeeds. Keep the app path as `Capstone Project/app.py`; no model retraining
is needed for this path fix.

Screenshot evidence: `evidence/streamlit_sample.png` shows the hosted input screen. The prediction result is outside the captured area. A local check of the same six values returns class 1 with score 0.6621 for class 1.
