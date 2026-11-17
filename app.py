import numpy as np
import pandas as pd
import streamlit as st
import itertools
from sklearn.base import clone
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="wide")

CAT_FILL = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History", "Loan_Amount_Term"]
DISCLAIMER = ("Learning prototype trained on historical decisions from a public dataset. "
              "It is NOT a real lending decision system.")


def needs_scale(name):
    return name.startswith(("Logistic", "KNN", "SVM"))


def engineer(df):
    """Feature engineering shared by training data and the prediction form."""
    df = df.copy()
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["LogTotalIncome"] = np.log1p(df["TotalIncome"])
    df["LogLoanAmount"] = np.log1p(df["LoanAmount"])
    df["LoanToIncome"] = df["LoanAmount"] / df["TotalIncome"].replace(0, np.nan)
    df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"]
    return df.drop(columns=["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "TotalIncome"])


@st.cache_resource(show_spinner="Cleaning data and training models...")
def build(file):
    df = pd.read_csv(file)
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])
    df = df.drop_duplicates()
    for c in CAT_FILL:
        df[c] = df[c].fillna(df[c].mode()[0])
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Dependents"] = df["Dependents"].astype(str).replace("3+", "3").astype(int)
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    sensitive = df[["Gender", "Married", "Property_Area"]].copy()
    y = df["Loan_Status"]
    X = pd.get_dummies(engineer(df.drop(columns=["Loan_Status"])), drop_first=True).astype(float)
    X = X.fillna(X.median())

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler().fit(X_tr)

    models = {}
    for bal in (None, "balanced"):
        sfx = " (balanced)" if bal else ""
        models["Logistic Regression" + sfx] = LogisticRegression(max_iter=1000, class_weight=bal)
        models["Decision Tree" + sfx] = DecisionTreeClassifier(max_depth=4, class_weight=bal, random_state=42)
        models["Random Forest" + sfx] = RandomForestClassifier(n_estimators=200, class_weight=bal, random_state=42)

    models["Naive Bayes"] = GaussianNB()
    models["KNN"] = KNeighborsClassifier(n_neighbors=7)
    models["SVM"] = SVC(probability=True, random_state=42)
    models["AdaBoost"] = AdaBoostClassifier(n_estimators=150, random_state=42)
    models["Gradient Boosting"] = GradientBoostingClassifier(random_state=42)
    models["Stacking Ensemble"] = StackingClassifier(
        estimators=[("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
                    ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
                    ("gb", GradientBoostingClassifier(random_state=42))],
        final_estimator=LogisticRegression(max_iter=1000), cv=3)

    rows = []
    for name, m in models.items():
        lr = needs_scale(name)
        m.fit(scaler.transform(X_tr) if lr else X_tr, y_tr)
        Xte = scaler.transform(X_te) if lr else X_te
        pred = m.predict(Xte)
        rep = classification_report(y_te, pred, output_dict=True, zero_division=0)
        rows.append({"Model": name, "Precision": rep["1"]["precision"], "Recall": rep["1"]["recall"],
                     "F1": rep["1"]["f1-score"], "Accuracy": rep["accuracy"],
                     "AUC": roc_auc_score(y_te, m.predict_proba(Xte)[:, 1])})
    return dict(models=models, scaler=scaler, X=X, X_te=X_te, y_te=y_te, y=y,
                results=pd.DataFrame(rows).set_index("Model").round(3),
                sens_te=sensitive.loc[X_te.index], raw=df)


def predict_many(b, name, rows):
    X_rows = pd.get_dummies(engineer(pd.DataFrame(rows))).reindex(columns=b["X"].columns, fill_value=0).astype(float)
    if needs_scale(name):
        X_rows = b["scaler"].transform(X_rows)
    return b["models"][name].predict_proba(X_rows)[:, 1]


def predict(b, name, row):
    return float(predict_many(b, name, [row])[0])


import io
import json
import re
import streamlit.components.v1 as components

VOICE_FIELDS = {"f_gender": "Gender", "f_married": "Married", "f_deps": "Dependents", "f_edu": "Education",
                "f_selfemp": "Self employed", "f_area": "Property area", "f_credit": "Credit history",
                "f_term": "Loan term", "f_inc": "Applicant income", "f_co": "Co-applicant income",
                "f_amt": "Loan amount"}
TERMS = [360, 180, 480, 300, 240, 120, 84, 60, 36, 12]


def parse_voice(text):
    """Pull form values out of a spoken sentence (English)."""
    t = text.lower().replace(",", "").replace("-", " ")
    out, num = {}, r"(\d+(?:\.\d+)?)"

    def grab(pattern):
        nonlocal t
        m = re.search(pattern, t)
        if m:
            t = t.replace(m.group(0), " ", 1)
            return float(m.group(1))
        return None

    v = grab(r"co ?applicant(?: income)?\D{0,15}" + num)
    if v is not None: out["f_co"] = int(min(v, 100000))
    v = grab(num + r"\s*months?") or grab(r"term\D{0,15}" + num)
    if v is not None and int(v) in TERMS: out["f_term"] = int(v)
    if re.search(r"no dependents?", t): out["f_deps"] = 0
    else:
        v = grab(num + r"\s*dependents?")
        if v is None: v = grab(r"dependents?\D{0,10}" + num)
        if v is not None: out["f_deps"] = int(min(v, 3))
    v = grab(r"income\D{0,15}" + num)
    if v is not None: out["f_inc"] = int(min(v, 200000))
    v = grab(r"loan(?: amount)?\D{0,15}" + num)
    if v is not None:
        out["f_amt"] = int(min(max(v / 1000 if v > 700 else v, 1), 700))
    if re.search(r"(no|bad|poor|without) credit|credit(?: history)? (is )?(bad|poor|no)", t): out["f_credit"] = 0.0
    elif "credit" in t and re.search(r"good|yes|clear|clean|meets|fine", t): out["f_credit"] = 1.0
    if re.search(r"not graduate|non graduate|undergraduate", t): out["f_edu"] = "Not Graduate"
    elif "graduate" in t: out["f_edu"] = "Graduate"
    if re.search(r"unmarried|not married|single", t): out["f_married"] = "No"
    elif "married" in t: out["f_married"] = "Yes"
    if re.search(r"not self|salaried", t): out["f_selfemp"] = "No"
    elif "self employed" in t: out["f_selfemp"] = "Yes"
    if re.search(r"semi ?urban", t): out["f_area"] = "Semiurban"
    elif "urban" in t: out["f_area"] = "Urban"
    elif "rural" in t: out["f_area"] = "Rural"
    if re.search(r"female|woman|girl", t): out["f_gender"] = "Female"
    elif re.search(r"\bmale\b|\bman\b|\bboy\b", t): out["f_gender"] = "Male"
    return out


def transcribe(wav_bytes, lang):
    import speech_recognition as sr  # pip install SpeechRecognition (needs internet for Google's recognizer)
    r = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(wav_bytes)) as src:
        audio = r.record(src)
    return r.recognize_google(audio, language=lang)


def speak_button(text, lang="en-US"):
    html = f"""
    <button id="b" style="padding:8px 14px;border-radius:8px;border:1px solid #888;cursor:pointer;
      font-size:15px">🔊 Read result aloud</button>
    <script>
      const t = {json.dumps(text)};
      document.getElementById('b').onclick = () => {{
        speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(t); u.lang = {json.dumps(lang)}; u.rate = 0.95;
        speechSynthesis.speak(u);
      }};
    </script>"""
    if hasattr(st, "iframe"):          # newer Streamlit
        st.iframe(html, height=55)
    else:
        components.html(html, height=55)


# ---------- Sidebar ----------
st.sidebar.title("🏦 Loan Approval Prediction")
up = st.sidebar.file_uploader("Dataset CSV (Loan Prediction Problem)", type="csv")
from pathlib import Path
HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE.parent / "data" / "loan_data.csv", HERE / "loan_data.csv",
              Path("data/loan_data.csv"), Path("loan_data.csv")]
found = next((c for c in CANDIDATES if c.exists()), None)
path = up if up else found
if path is None:
    st.error("Dataset not found. Put `loan_data.csv` inside the `data/` folder, or upload it in the sidebar.")
    st.stop()
b = build(path)
model_name = st.sidebar.selectbox("Model", list(b["models"].keys()), index=2)
with st.sidebar.expander("🎙️ Voice settings"):
    voice_on = st.checkbox("Enable voice", value=True)
    tts_lang = st.selectbox("Read-aloud voice", ["en-US", "en-GB", "en-IN"])
    stt_lang = st.selectbox("I will speak in", ["en-PK", "en-IN", "en-US"])
st.sidebar.warning(DISCLAIMER)

st.title("Loan Approval Prediction")
st.caption(DISCLAIMER)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🔮 Predict", "📊 Model comparison", "⭐ Feature importance", "⚖️ Fairness audit",
    "💰 Cost & threshold", "🧬 Proxy detection", "🛠️ Bias mitigation", "⚡ Scalability", "📄 Model card", "🔬 Research methods"])

# ---------- Predict ----------
with tab1:
    st.subheader("Try your own details")
    st.write("Change anything and see how the model responds. There are no wrong answers here, just experiments. 🙂")
    for k, v in dict(f_gender="Male", f_married="Yes", f_deps=0, f_edu="Graduate", f_selfemp="No", f_area="Urban",
                     f_credit=1.0, f_term=360, f_inc=5000, f_co=0, f_amt=130).items():
        st.session_state.setdefault(k, v)

    if voice_on:
        with st.expander("🎙️ Speak your details (optional)"):
            st.caption("Example: \"income 6000, co-applicant income 1000, loan amount 150, 360 months, 1 dependent, "
                       "graduate, married, urban, good credit history\"")
            if hasattr(st, "audio_input"):
                rec = st.audio_input("Press the mic, speak, then stop")
                if rec is not None and st.session_state.get("voice_sig") != hash(rec.getvalue()):
                    st.session_state["voice_sig"] = hash(rec.getvalue())
                    try:
                        said = transcribe(rec.getvalue(), stt_lang)
                        found = parse_voice(said)
                        for k, v in found.items():
                            st.session_state[k] = v
                        st.session_state["voice_msg"] = (said, found, None)
                    except Exception as ex:
                        st.session_state["voice_msg"] = (None, {}, f"{type(ex).__name__}: {ex}")
                msg = st.session_state.get("voice_msg")
                if msg:
                    said, found, err = msg
                    if err:
                        st.error("Could not understand the audio. Check the internet connection and that "
                                 f"`SpeechRecognition` is installed. ({err})")
                    else:
                        st.write(f"🗣️ I heard: *{said}*")
                        st.success("Filled in: " + (", ".join(VOICE_FIELDS[k] for k in found) or "nothing recognised") +
                                   ". Please check the form below, then press the button.")
            else:
                st.info("Voice recording needs a newer Streamlit: `pip install -U streamlit`.")

    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", ["Male", "Female"], key="f_gender")
    married = c1.selectbox("Married", ["Yes", "No"], key="f_married")
    deps = c1.selectbox("Dependents", [0, 1, 2, 3], key="f_deps")
    edu = c2.selectbox("Education", ["Graduate", "Not Graduate"], key="f_edu")
    selfemp = c2.selectbox("Self employed", ["No", "Yes"], key="f_selfemp")
    area = c2.selectbox("Property area", ["Urban", "Semiurban", "Rural"], key="f_area")
    credit = c3.selectbox("Credit history meets guidelines", [1.0, 0.0], key="f_credit",
                          format_func=lambda v: "Yes" if v == 1.0 else "No")
    term = c3.selectbox("Loan term (months)", TERMS, key="f_term")
    inc = c1.number_input("Applicant income (monthly)", 0, 200000, step=500, key="f_inc")
    co_inc = c2.number_input("Co-applicant income (monthly)", 0, 100000, step=500, key="f_co")
    amt = c3.number_input("Loan amount (in thousands)", 1, 700, step=5, key="f_amt")

    if st.button("See how the model responds", type="primary"):
        row = dict(Gender=gender, Married=married, Dependents=deps, Education=edu, Self_Employed=selfemp,
                   ApplicantIncome=inc, CoapplicantIncome=co_inc, LoanAmount=amt,
                   Loan_Amount_Term=float(term), Credit_History=credit, Property_Area=area)
        p = predict(b, model_name, row)

        if p >= 0.7:
            st.success(f"🌟 Looks promising! The model sees about a {p:.0%} likelihood of approval for these details.")
            st.balloons()
        elif p >= 0.5:
            st.info(f"👍 Leaning positive. The model sees about a {p:.0%} likelihood. A few small changes could make it stronger.")
        else:
            st.warning(f"🌱 Room to grow. The model sees about a {p:.0%} likelihood right now, "
                       "and this can change a lot with the details below.")
        st.progress(float(p))
        spoken = [f"The model sees about a {p:.0%} likelihood of approval. " +
                  ("This looks promising." if p >= 0.7 else "This is leaning positive." if p >= 0.5
                   else "There is room to grow.")]

        # ---- Action plan: realistic changes the user CAN make (never sensitive attributes) ----
        if p >= 0.85:
            st.markdown("#### ✅ Strong profile for this model. Try changing a few values and see what moves the result.")
        else:
            target = 0.5 if p < 0.5 else min(p + 0.15, 0.9)
            levers = []
            if credit == 0.0:
                levers.append(("Credit_History", "time", [(1.0, "Bring your credit history up to lender guidelines "
                                                                "(pay on time, clear overdue amounts)")]))
            levers.append(("LoanAmount", "now", [(amt * f, f"Request about {amt * f:.0f}k instead of {amt}k")
                                                 for f in (0.9, 0.8, 0.7, 0.6, 0.5)]))
            longer = [t for t in (360.0, 480.0) if t > term]
            if longer:
                levers.append(("Loan_Amount_Term", "now", [(t, f"Choose a {int(t)}-month term") for t in longer]))
            levers.append(("CoapplicantIncome", "now", [(co_inc + x, f"Add a co-applicant earning about {x:,}/month")
                                                        for x in (1000, 2000, 4000)]))
            levers.append(("ApplicantIncome", "time", [(inc * f, f"Grow your income to about {inc * f:,.0f}/month")
                                                       for f in (1.1, 1.2, 1.3)]))
            choices = [[None] + [(n, k, i, v, l) for i, (v, l) in enumerate(o)] for n, k, o in levers]
            plans = [[c for c in combo if c] for combo in itertools.product(*choices)]
            plans = [c for c in plans if 1 <= len(c) <= 2]
            probs_ = predict_many(b, model_name, [{**row, **{c[0]: c[3] for c in ch}} for ch in plans])
            scored = []
            for ch, pp in zip(plans, probs_):
                effort = sum((c[2] + 1) * (2 if c[1] == "time" else 1) for c in ch)
                scored.append((pp >= target, effort, pp, ch))
            ok = sorted([x for x in scored if x[0]], key=lambda x: (x[1], -x[2]))
            best = ok[:3] if ok else sorted(scored, key=lambda x: -x[2])[:3]

            if p < 0.5:
                st.markdown("#### 🧭 Your path to a likelier yes")
            else:
                st.markdown("#### 🧭 Ways to make it even stronger")
            if not ok:
                st.write("No single small step gets all the way there with this model, but these come closest. "
                         "Combining steps over time can add up.")
            lines = [f"Action plan (learning model, current likelihood {p:.0%})"]
            for n, (_, _, pp, ch) in enumerate(best, 1):
                spoken.append(f"Plan {chr(64 + n)}: about {pp:.0%}. " + ". ".join(c[4] for c in ch) + ".")
                st.markdown(f"**Plan {chr(64 + n)}: about {pp:.0%}** ({(pp - p) * 100:+.0f} points)")
                lines.append(f"\nPlan {chr(64 + n)}: about {pp:.0%}")
                for c in ch:
                    tag = "⏳ Takes time" if c[1] == "time" else "✅ You can do now"
                    st.markdown(f"- {tag}: {c[4]}")
                    lines.append(f"- {tag}: {c[4]}")
            with st.expander("📋 General things worth doing (not model-specific)"):
                tips = ["Check your credit report for mistakes and ask for corrections.",
                        "Pay down existing debts and avoid new ones before applying.",
                        "Apply for an amount you can comfortably repay.",
                        "Keep income proof and documents ready; a stable co-applicant can help.",
                        "Talk to a loan officer or free financial counselor about your real options."]
                for t in tips:
                    st.markdown(f"- {t}")
                lines += ["\nGeneral tips:"] + [f"- {t}" for t in tips]
            st.download_button("⬇️ Download my plan", "\n".join(lines), file_name="my_action_plan.txt")
            st.caption("Plans only use things a person can actually change. Gender, marital status and similar "
                       "details are never suggested as things to change.")

        if voice_on:
            speak_button(" ".join(spoken) + " This is a learning tool, not a real lending decision.", tts_lang)
        st.info("💙 This is a learning tool trained on past decisions. It can't see your whole story, and real lenders "
                "look at much more. Treat the result as something to explore, not a verdict on anyone.")

# ---------- Comparison ----------
with tab2:
    st.subheader("Test-set metrics (positive class = approved)")
    st.dataframe(b["results"].style.highlight_max(axis=0, color="#cfe8cf"))
    st.bar_chart(b["results"][["Precision", "Recall", "F1"]])
    st.subheader("Class balance")
    st.bar_chart(b["y"].map({1: "Approved", 0: "Rejected"}).value_counts())
    st.caption(f"Approval rate: {b['y'].mean():.1%}. Predicting 'approve' for everyone already scores about this "
               "in accuracy, so rely on precision, recall and F1.")
    st.subheader(f"Confusion matrix: {model_name}")
    m = b["models"][model_name]
    Xte = b["scaler"].transform(b["X_te"]) if needs_scale(model_name) else b["X_te"]
    cm = confusion_matrix(b["y_te"], m.predict(Xte))
    st.dataframe(pd.DataFrame(cm, index=["Actual rejected", "Actual approved"],
                              columns=["Predicted rejected", "Predicted approved"]))

# ---------- Importance ----------
with tab3:
    cols = b["X"].columns
    a, c = st.columns(2)
    lr = b["models"]["Logistic Regression"]
    a.subheader("Logistic regression coefficients")
    a.bar_chart(pd.Series(lr.coef_[0], index=cols).sort_values())
    rf = b["models"]["Random Forest"]
    c.subheader("Random forest importances")
    c.bar_chart(pd.Series(rf.feature_importances_, index=cols).sort_values())
    st.caption("Check whether Credit_History dominates and whether the models agree.")

# ---------- Fairness ----------
with tab4:
    st.subheader(f"Group metrics: {model_name}")
    Xte = b["scaler"].transform(b["X_te"]) if needs_scale(model_name) else b["X_te"]
    audit = b["sens_te"].assign(actual=b["y_te"].values, pred=b["models"][model_name].predict(Xte))
    for col in ["Gender", "Married", "Property_Area"]:
        g = audit.groupby(col).apply(lambda d: pd.Series({
            "n": len(d),
            "Actual approval rate": d.actual.mean(),
            "Selection rate": d.pred.mean(),
            "Recall (TPR)": ((d.pred == 1) & (d.actual == 1)).sum() / max((d.actual == 1).sum(), 1),
            "False positive rate": ((d.pred == 1) & (d.actual == 0)).sum() / max((d.actual == 0).sum(), 1),
        }), include_groups=False).round(3)
        st.markdown(f"**By {col}**")
        st.dataframe(g)
        gap = g["Selection rate"].max() - g["Selection rate"].min()
        st.caption(f"Selection-rate gap: {gap:.3f}")
    st.warning("Subgroups are small, so these numbers are unreliable. Proxy variables (e.g. property area) "
               "and biased historical labels can still cause unfair outcomes even without sensitive columns.")


# =====================================================================
# Extra interactive features
# =====================================================================
import time
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def get_probs(b, name):
    m = b["models"][name]
    Xte = b["scaler"].transform(b["X_te"]) if needs_scale(name) else b["X_te"]
    return m.predict_proba(Xte)[:, 1]


def confusion_counts(y_true, pred):
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn


# ---------- Cost-sensitive threshold ----------
with tab5:
    st.subheader("Cost-sensitive decision threshold")
    st.write("A false approval (bad loan) and a false rejection (lost good customer) rarely cost the same. "
             "Set the costs and the threshold, and see what happens.")
    c1, c2, c3 = st.columns(3)
    cost_fp = c1.slider("Cost of a false approval", 1, 20, 5)
    cost_fn = c2.slider("Cost of a false rejection", 1, 20, 1)
    thr = c3.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01)

    probs = get_probs(b, model_name)
    y_true = b["y_te"].values
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    cost_curve = []
    for t in grid:
        _, fp_, fn_, _ = confusion_counts(y_true, (probs >= t).astype(int))
        cost_curve.append(cost_fp * fp_ + cost_fn * fn_)
    best_t = float(grid[int(np.argmin(cost_curve))])

    tp, fp, fn, tn = confusion_counts(y_true, (probs >= thr).astype(int))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total cost", cost_fp * fp + cost_fn * fn)
    m2.metric("Precision", f"{prec:.2f}")
    m3.metric("Recall", f"{rec:.2f}")
    m4.metric("F1", f"{f1:.2f}")
    st.success(f"Lowest-cost threshold for these costs: **{best_t:.2f}** "
               f"(cost {min(cost_curve)}). Your threshold: {thr:.2f}.")
    st.line_chart(pd.DataFrame({"Total cost": cost_curve}, index=grid))
    st.dataframe(pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual rejected", "Actual approved"],
                              columns=["Predicted rejected", "Predicted approved"]))

# ---------- Proxy detection ----------
with tab6:
    st.subheader("Can the other features reveal a sensitive attribute?")
    st.write("Remove the sensitive attribute, then try to predict it from everything else. "
             "If that works well, the remaining features act as proxies, so deleting the column does not remove bias.")
    target_col = st.selectbox("Sensitive attribute to predict", ["Gender", "Married", "Property_Area"])
    if st.button("Run proxy test"):
        y_s = b["raw"].loc[b["X"].index, target_col]
        X_p = b["X"].drop(columns=[c for c in b["X"].columns if c.startswith(target_col + "_")])
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        acc = cross_val_score(clf, X_p, y_s, cv=5).mean()
        base = y_s.value_counts(normalize=True).max()
        clf.fit(X_p, y_s)
        a1, a2, a3 = st.columns(3)
        a1.metric("Accuracy predicting " + target_col, f"{acc:.1%}")
        a2.metric("Majority-class baseline", f"{base:.1%}")
        a3.metric("Gain over baseline", f"{(acc - base) * 100:+.1f} pts")
        if acc - base > 0.05:
            st.warning("Noticeably better than the baseline: other features likely carry information about "
                       f"{target_col}, so removing it alone is not enough.")
        else:
            st.info("Close to the baseline: little evidence of strong proxies with this model. "
                    "That does not prove there are none.")
        st.markdown("**Features that leak the most information**")
        st.bar_chart(pd.Series(clf.feature_importances_, index=X_p.columns).sort_values().tail(8))
        st.caption("Rough diagnostic only. Other dummy columns of related attributes are kept in the features.")

# ---------- Bias mitigation ----------
with tab7:
    st.subheader("Bias mitigation: before vs after")
    attr = st.selectbox("Sensitive attribute", ["Gender", "Married", "Property_Area"], key="mit_attr")
    base_thr = st.slider("Base threshold", 0.05, 0.95, 0.50, 0.01, key="mit_base")
    mode = st.radio("Mitigation", ["None (single threshold)", "Manual per-group thresholds",
                                   "Auto: match selection rates"], horizontal=True)
    probs = get_probs(b, model_name)
    groups = b["sens_te"][attr].values
    y_true = b["y_te"].values
    uniq = sorted(pd.unique(groups))
    thr_map = {g: base_thr for g in uniq}
    if mode.startswith("Manual"):
        for g in uniq:
            thr_map[g] = st.slider(f"Threshold for {g}", 0.05, 0.95, base_thr, 0.01, key=f"thr_{attr}_{g}")
    elif mode.startswith("Auto"):
        target_rate = float((probs >= base_thr).mean())
        for g in uniq:
            pg = probs[groups == g]
            thr_map[g] = float(np.quantile(pg, 1 - target_rate)) if len(pg) else base_thr

    def summarize(pred):
        rows = {}
        for g in uniq:
            mask = groups == g
            tp_, fp_, fn_, tn_ = confusion_counts(y_true[mask], pred[mask])
            rows[g] = {"n": int(mask.sum()),
                       "Selection rate": pred[mask].mean(),
                       "Recall (TPR)": tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0,
                       "False positive rate": fp_ / (fp_ + tn_) if (fp_ + tn_) else 0.0}
        t = pd.DataFrame(rows).T.round(3)
        return t, t["Selection rate"].max() - t["Selection rate"].min()

    pred_before = (probs >= base_thr).astype(int)
    pred_after = (probs >= np.array([thr_map[g] for g in groups])).astype(int)
    tab_b, gap_b = summarize(pred_before)
    tab_a, gap_a = summarize(pred_after)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Selection-rate gap", f"{gap_a:.3f}", f"{gap_a - gap_b:+.3f} vs before", delta_color="inverse")
    k2.metric("Accuracy", f"{accuracy_score(y_true, pred_after):.3f}",
              f"{accuracy_score(y_true, pred_after) - accuracy_score(y_true, pred_before):+.3f}")
    k3.metric("F1", f"{f1_score(y_true, pred_after, zero_division=0):.3f}",
              f"{f1_score(y_true, pred_after, zero_division=0) - f1_score(y_true, pred_before, zero_division=0):+.3f}")
    k4.metric("Groups", len(uniq))
    cb, ca = st.columns(2)
    cb.markdown("**Before (single threshold)**")
    cb.dataframe(tab_b)
    ca.markdown("**After (selected mitigation)**")
    ca.dataframe(tab_a)
    st.caption("Narrowing the gap usually costs some accuracy or F1: this is the fairness vs accuracy trade-off. "
               "Using group-specific thresholds can itself raise legal and ethical questions in real lending. "
               "Small groups make these numbers noisy.")

# ---------- Scalability ----------
with tab8:
    st.subheader("Scalability benchmark")
    st.write("Measure training time as the data grows. Rows are resampled with replacement from your dataset, "
             "so this tests speed only, not model quality.")
    sizes = st.multiselect("Row counts", [1000, 5000, 10000, 50000, 100000, 200000],
                           default=[1000, 10000, 50000])
    if st.button("Run benchmark") and sizes:
        rows, prog = [], st.progress(0.0)
        for i, n in enumerate(sorted(sizes)):
            idx = np.random.RandomState(0).choice(len(b["X"]), n, replace=True)
            Xs, ys = b["X"].iloc[idx], b["y"].iloc[idx]
            t0 = time.time()
            LogisticRegression(max_iter=1000).fit(StandardScaler().fit_transform(Xs), ys)
            t_lr = time.time() - t0
            t0 = time.time()
            RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42).fit(Xs, ys)
            t_rf = time.time() - t0
            rows.append({"Rows": n, "Logistic Regression (s)": round(t_lr, 3), "Random Forest (s)": round(t_rf, 3)})
            prog.progress((i + 1) / len(sizes))
        res = pd.DataFrame(rows).set_index("Rows")
        st.line_chart(res)
        st.dataframe(res)
        st.caption("Single machine, scikit-learn. Run the same sizes in PySpark and add those times to your report "
                   "to compare.")

# ---------- Model card ----------
with tab9:
    st.subheader("Model card")
    st.markdown(f"""
**Purpose.** Educational prototype for a Big Data Analysis course. It predicts historical loan approval
outcomes from applicant and loan attributes.

**Data.** {len(b['raw']):,} cleaned records, approval rate {b['y'].mean():.1%}. Missing values filled with
mode (categorical) and median (loan amount). Skewed amounts log-transformed.

**Models.** Logistic Regression, Decision Tree (depth 4) and Random Forest (200 trees), each with and without
class weights, plus Naive Bayes, KNN, SVM, AdaBoost, Gradient Boosting and a Stacking Ensemble. Stratified 80/20 split, scaler fitted on training data only.

**Intended use.** Learning, demonstration, discussion of fairness and bias.

**Not intended for.** Real lending, credit scoring, or any decision about real people.

**Known limitations.**
- Labels are past human decisions and may contain historical bias.
- Small subgroups make fairness numbers unreliable.
- Proxy variables (for example property area) can encode sensitive traits even when those columns are removed.
- No validation on real-world outcomes such as repayment or default.

**Fairness checks.** Group selection rate, recall and false positive rate; proxy test; threshold-based mitigation.
""")
    st.warning(DISCLAIMER)


# ---------- Research methods ----------
with tab10:
    st.subheader("Methods from the research papers")
    st.markdown("""
| Method | Where it comes from |
|---|---|
| KNN, SVM, K-Best and RFE feature selection, K-fold cross-validation | Sinap (2024) comparative study |
| AdaBoost, Gaussian Naive Bayes, SVM | Haque and Hassan (2024) |
| Stacking ensemble | Stacking approach on Kaggle loan data (JoVE, 2025) |
| Subgroup thresholds for fairness | Ying (2024); Kozodoi et al. (2021) |
""")
    base_models = ["Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes", "KNN", "SVM",
                   "AdaBoost", "Gradient Boosting"]
    c1, c2, c3, c4 = st.columns(4)
    rm_model = c1.selectbox("Model", base_models, key="rm_model")
    rm_method = c2.selectbox("Feature selection", ["All features", "K-Best (ANOVA)", "RFE"], key="rm_method")
    n_feat = len(b["X"].columns)
    rm_k = c3.slider("Features to keep", 3, n_feat, min(8, n_feat), key="rm_k")
    rm_folds = c4.slider("CV folds", 3, 10, 5, key="rm_folds")
    if st.button("Run cross-validation"):
        def make_pipe(method):
            sel = "passthrough" if method == "All features" else (
                SelectKBest(f_classif, k=rm_k) if method.startswith("K-Best")
                else RFE(LogisticRegression(max_iter=1000), n_features_to_select=rm_k))
            return Pipeline([("scale", StandardScaler()), ("sel", sel), ("model", clone(b["models"][rm_model]))])
        cv = StratifiedKFold(rm_folds, shuffle=True, random_state=42)
        out = {}
        for meth in {"All features", rm_method}:
            sc = cross_val_score(make_pipe(meth), b["X"], b["y"], cv=cv, scoring="f1")
            out[meth] = {"F1 mean": sc.mean(), "F1 std": sc.std()}
        st.dataframe(pd.DataFrame(out).T.round(3))
        if rm_method != "All features":
            pipe = make_pipe(rm_method).fit(b["X"], b["y"])
            keep = b["X"].columns[pipe.named_steps["sel"].get_support()]
            st.markdown("**Selected features:** " + ", ".join(keep))
        st.caption("Selection happens inside each CV fold, so there is no data leakage.")
