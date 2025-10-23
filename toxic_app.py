"""toxic_app.py

Safe loading of optional model/vectorizer pickle files so the Flask app
can run even if those artifacts aren't present. If real artifacts are
present they will be used.
"""

from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# optional sklearn imports; fall back to dummy classes if unavailable
SKLEARN_AVAILABLE = True
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
except Exception:
    SKLEARN_AVAILABLE = False

app = Flask(__name__)


class DummyVectorizer:
    def transform(self, data):
        return np.zeros((len(data), 1))


class DummyModel:
    def predict_proba(self, X):
        n = X.shape[0]
        # (n_samples, 2) -> prob for classes [0, 1]
        return np.hstack([np.ones((n, 1)) * 0.5, np.ones((n, 1)) * 0.5])


def load_pickle_or_dummy(path, kind="model"):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: failed loading {path}: {e}")

    # create a reasonable dummy depending on kind
    if kind == "vect":
        if SKLEARN_AVAILABLE:
            return TfidfVectorizer()
        return DummyVectorizer()
    else:
        if SKLEARN_AVAILABLE:
            return RandomForestClassifier()
        return DummyModel()


# vectorizers
tox = load_pickle_or_dummy("toxic_vect.pkl", kind="vect")
sev = load_pickle_or_dummy("severe_toxic_vect.pkl", kind="vect")
obs = load_pickle_or_dummy("obscene_vect.pkl", kind="vect")
ins = load_pickle_or_dummy("insult_vect.pkl", kind="vect")
thr = load_pickle_or_dummy("threat_vect.pkl", kind="vect")
ide = load_pickle_or_dummy("identity_hate_vect.pkl", kind="vect")

# models
tox_model = load_pickle_or_dummy("toxic_model.pkl", kind="model")
sev_model = load_pickle_or_dummy("severe_toxic_model.pkl", kind="model")
obs_model = load_pickle_or_dummy("obscene_model.pkl", kind="model")
ins_model = load_pickle_or_dummy("insult_model.pkl", kind="model")
thr_model = load_pickle_or_dummy("threat_model.pkl", kind="model")
ide_model = load_pickle_or_dummy("identity_hate_model.pkl", kind="model")

@app.route("/")
def home():
    return render_template('index_toxic.html')

@app.route("/predict", methods=['POST'])
def predict():
    user_input = request.form['text']
    data = [user_input]

    vect = tox.transform(data)
    pred_tox = tox_model.predict_proba(vect)[:, 1]

    vect = sev.transform(data)
    pred_sev = sev_model.predict_proba(vect)[:, 1]

    vect = obs.transform(data)
    pred_obs = obs_model.predict_proba(vect)[:, 1]

    vect = thr.transform(data)
    pred_thr = thr_model.predict_proba(vect)[:, 1]

    vect = ins.transform(data)
    pred_ins = ins_model.predict_proba(vect)[:, 1]

    vect = ide.transform(data)
    pred_ide = ide_model.predict_proba(vect)[:, 1]

    out_tox = round(pred_tox[0], 2)
    out_sev = round(pred_sev[0], 2)
    out_obs = round(pred_obs[0], 2)
    out_ins = round(pred_ins[0], 2)
    out_thr = round(pred_thr[0], 2)
    out_ide = round(pred_ide[0], 2)

    return render_template('index_toxic.html',
                           pred_tox='Prob (Toxic): {}'.format(out_tox),
                           pred_sev='Prob (Severe Toxic): {}'.format(out_sev),
                           pred_obs='Prob (Obscene): {}'.format(out_obs),
                           pred_ins='Prob (Insult): {}'.format(out_ins),
                           pred_thr='Prob (Threat): {}'.format(out_thr),
                           pred_ide='Prob (Identity Hate): {}'.format(out_ide)
                           )

if __name__ == "__main__":
    app.run(debug=True)
