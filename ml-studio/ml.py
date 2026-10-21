"""Model code (no screens here). Run `python ml.py` once to train and cache everything."""
import io, os, warnings
import joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

BASE = os.path.dirname(os.path.abspath(__file__))
DATA, CACHE, RS = os.path.join(BASE, "data"), os.path.join(BASE, "cache"), 42
CLS_CSV = os.path.join(DATA, "faults_with_missing.csv")
CLU_CSV = os.path.join(DATA, "marketing_campaign.csv")

# ═════════════════════════ Classification (steel plate faults) ═════════════════════════
FAULTS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
pipe = lambda m: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", m)])


def _train_classification():
    df = pd.read_csv(CLS_CSV)
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(df[FAULTS].idxmax(axis=1)), index=df.index)
    X = df.drop(columns=FAULTS)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)
    models = {
        "Logistic Regression": pipe(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RS)),
        "KNN": pipe(KNeighborsClassifier(n_neighbors=5)),
        "Gaussian Naive Bayes": pipe(GaussianNB()),
        "Decision Tree": pipe(DecisionTreeClassifier(class_weight="balanced", random_state=RS)),
        "SVC": pipe(SVC(C=1, kernel="rbf", probability=True, class_weight="balanced", random_state=RS)),
        "Random Forest": pipe(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RS, n_jobs=-1)),
        "AdaBoost": pipe(AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=RS)),
        "Gradient Boosting": pipe(GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=RS)),
        "Bagging (Decision Tree)": pipe(BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RS), n_estimators=100, random_state=RS, n_jobs=-1)),
        "MLP Neural Network": pipe(MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, early_stopping=True, random_state=RS)),
    }
    rows, cms, reports, rocs = [], {}, {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p, pr = m.predict(Xte), m.predict_proba(Xte)
        rocs[name] = dict(y=yte.values, p=pr)
        try:
            auc = roc_auc_score(yte, pr, labels=m.classes_, multi_class="ovr", average="weighted")
        except Exception:
            auc = np.nan
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, p),
                     "Precision": precision_score(yte, p, average="weighted", zero_division=0),
                     "Recall": recall_score(yte, p, average="weighted", zero_division=0),
                     "F1": f1_score(yte, p, average="weighted", zero_division=0), "ROC-AUC": auc})
        cms[name] = confusion_matrix(yte, p)
        rep = classification_report(yte, p, target_names=le.classes_, output_dict=True, zero_division=0)
        reports[name] = pd.DataFrame([{"Fault": c, "Precision": rep[c]["precision"], "Recall": rep[c]["recall"],
                                       "F1": rep[c]["f1-score"], "Plates": int(rep[c]["support"])} for c in le.classes_])
    table = pd.DataFrame(rows).set_index("Model").sort_values("F1", ascending=False)
    feats = []
    for c in X.columns:
        vals = set(X[c].dropna().unique())
        feats.append(dict(name=c, min=float(X[c].min()), max=float(X[c].max()), med=float(X[c].median()),
                          binary=vals <= {0, 1, 0.0, 1.0}))
    samples = Xte.copy()
    samples["Actual"] = le.classes_[yte.values]
    rf = models["Random Forest"].named_steps["model"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return dict(models=models, table=table, cms=cms, reports=reports, rocs=rocs, importance=imp,
                class_counts=dict(zip(le.classes_, np.bincount(y).tolist())),
                classes=list(le.classes_), cols=list(X.columns),
                feats=feats, samples=samples.reset_index(drop=True),
                info=dict(n=len(df), n_train=len(ytr), n_test=len(yte), n_feat=X.shape[1]))


# friendly name, plain-English help, group  (used to make the input form easy to understand)
FEATURE_INFO = {
    "X_Minimum": ("Left edge (X min)", "Where the defect starts along the plate width, in pixels.", "📍 Where is the defect?"),
    "X_Maximum": ("Right edge (X max)", "Where the defect ends along the plate width, in pixels.", "📍 Where is the defect?"),
    "Y_Minimum": ("Top edge (Y min)", "Where the defect starts along the plate length, in pixels.", "📍 Where is the defect?"),
    "Y_Maximum": ("Bottom edge (Y max)", "Where the defect ends along the plate length, in pixels.", "📍 Where is the defect?"),
    "Length_of_Conveyer": ("Conveyor length", "Length of the conveyor belt the plate travelled on.", "📍 Where is the defect?"),
    "Pixels_Areas": ("Defect area (pixels)", "How many pixels the defect covers. Bigger = larger defect.", "📏 How big is it?"),
    "X_Perimeter": ("Width outline (px)", "Length of the defect outline in the X direction.", "📏 How big is it?"),
    "Y_Perimeter": ("Height outline (px)", "Length of the defect outline in the Y direction.", "📏 How big is it?"),
    "LogOfAreas": ("Log of area", "Logarithm of the defect area (a smoother way to compare sizes).", "📏 How big is it?"),
    "Log_X_Index": ("Log of width", "Logarithm of the defect width.", "📏 How big is it?"),
    "Log_Y_Index": ("Log of height", "Logarithm of the defect height.", "📏 How big is it?"),
    "SigmoidOfAreas": ("Area score (0 to 1)", "Area squashed into a 0 to 1 scale.", "📏 How big is it?"),
    "Sum_of_Luminosity": ("Total brightness", "Sum of brightness of every pixel in the defect.", "💡 How bright is it?"),
    "Minimum_of_Luminosity": ("Darkest pixel (0 to 255)", "Brightness of the darkest pixel. 0 is black.", "💡 How bright is it?"),
    "Maximum_of_Luminosity": ("Brightest pixel (0 to 255)", "Brightness of the brightest pixel. 255 is white.", "💡 How bright is it?"),
    "Luminosity_Index": ("Brightness index (-1 to 1)", "Negative = darker than the plate, positive = brighter.", "💡 How bright is it?"),
    "TypeOfSteel_A300": ("Steel A300", "Is the plate made of steel type A300?", "🔩 What is the plate?"),
    "TypeOfSteel_A400": ("Steel A400", "Is the plate made of steel type A400?", "🔩 What is the plate?"),
    "Steel_Plate_Thickness": ("Plate thickness", "Thickness of the steel plate.", "🔩 What is the plate?"),
    "Edges_Index": ("Edge index (0 to 1)", "How much of the defect sits on the plate edge.", "🔷 What shape is it?"),
    "Empty_Index": ("Empty index (0 to 1)", "How much empty space is inside the defect box.", "🔷 What shape is it?"),
    "Square_Index": ("Squareness (0 to 1)", "1 = perfect square, near 0 = long and thin.", "🔷 What shape is it?"),
    "Outside_X_Index": ("Outside X index", "Share of the defect that lies outside the main X range.", "🔷 What shape is it?"),
    "Edges_X_Index": ("Edges X index", "Edge presence along the X direction.", "🔷 What shape is it?"),
    "Edges_Y_Index": ("Edges Y index", "Edge presence along the Y direction.", "🔷 What shape is it?"),
    "Outside_Global_Index": ("Outside global index", "Share of the defect outside the plate's main area.", "🔷 What shape is it?"),
    "Orientation_Index": ("Orientation (-1 to 1)", "Which way the defect points. -1 = vertical, +1 = horizontal.", "🔷 What shape is it?"),
}
FAULT_INFO = {
    "Pastry": ("🥐", "Flaky, pastry-like surface patches."), "Z_Scratch": ("⚡", "Z-shaped scratch marks."),
    "K_Scatch": ("🪓", "Deep K-type scratches."), "Stains": ("💧", "Discoloured stains on the surface."),
    "Dirtiness": ("🌫️", "Dirt or dust on the plate."), "Bumps": ("⛰️", "Raised bumps or dents."),
    "Other_Faults": ("❓", "Any other kind of fault."),
}


def load_classification():
    """Train once, then reuse the saved copy in cache/ (delete cache/ to retrain)."""
    f = os.path.join(CACHE, "classification_v4.joblib")
    if os.path.exists(f):
        try:
            return joblib.load(f)
        except Exception:
            pass
    art = _train_classification()
    try:
        os.makedirs(CACHE, exist_ok=True)
        joblib.dump(art, f)
    except OSError:
        pass
    return art


def cls_predict(art, model_name, values):
    row = pd.DataFrame([{c: values.get(c, np.nan) for c in art["cols"]}])
    pr = art["models"][model_name].predict_proba(row)[0]
    classes = art["classes"]
    votes = {n: classes[int(m.predict(row)[0])] for n, m in art["models"].items()}
    return dict(label=classes[int(np.argmax(pr))], probs=dict(zip(classes, map(float, pr))), votes=votes)


# ═════════════════════════ Clustering (customer segments) ═════════════════════════
ALGOS = {"kmeans": "K-Means", "ward": "Hierarchical (Ward)"}
KRANGE = list(range(2, 11))
SP = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
PU = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumDealsPurchases"]
EXC = ["ID", "Year_Birth", "Dt_Customer", "Response", "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3",
       "AcceptedCmp4", "AcceptedCmp5", "Complain", "Z_CostContact", "Z_Revenue"]
PROFILE = {"Income": "Avg income", "TotalSpend": "Avg total spend", "TotalPurchases": "Avg purchases",
           "Age": "Avg age", "HouseholdChildren": "Avg children"}
# name -> (label, kind) for every value a person can type in on the Predict tab
FIELDS = {"Income": ("Yearly income", "money"), "Age": ("Age", "int"), "Kidhome": ("Kids at home", "int"),
          "Teenhome": ("Teens at home", "int"), "Recency": ("Days since last purchase", "int"),
          "MntWines": ("Spent on wine", "int"), "MntFruits": ("Spent on fruit", "int"),
          "MntMeatProducts": ("Spent on meat", "int"), "MntFishProducts": ("Spent on fish", "int"),
          "MntSweetProducts": ("Spent on sweets", "int"), "MntGoldProds": ("Spent on gold products", "int"),
          "NumWebPurchases": ("Web purchases", "int"), "NumCatalogPurchases": ("Catalog purchases", "int"),
          "NumStorePurchases": ("Store purchases", "int"), "NumDealsPurchases": ("Purchases with a deal", "int"),
          "NumWebVisitsMonth": ("Web visits per month", "int")}


def derive(d):
    d = d.copy()
    if "Year_Birth" in d:
        d["Age"] = 2026 - d["Year_Birth"]
    d["TotalSpend"] = d[SP].sum(axis=1)
    d["TotalPurchases"] = d[PU].sum(axis=1)
    d["HouseholdChildren"] = d["Kidhome"] + d["Teenhome"]
    return d


class Clu:  # plain container for everything the clustering screen needs
    pass


def build_clustering(raw: bytes):
    df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    need = SP + PU + ["Income", "Kidhome", "Teenhome"]
    miss = [c for c in need if c not in df]
    if not ({"Age", "Year_Birth"} & set(df)):
        miss.append("Year_Birth")
    if miss:
        raise ValueError("This does not look like marketing_campaign.csv. Missing columns: " + ", ".join(miss))
    f = derive(df)
    f = f[(f["Age"] <= 100) & ~(f["Income"] > 200000)]                      # drop obvious outliers
    f = f.drop(columns=[c for c in EXC if c in f]).reset_index(drop=True)
    f = f.loc[:, f.nunique(dropna=True) > 1]
    num_c = f.select_dtypes(include=np.number).columns.tolist()
    cat_c = [c for c in f.columns if c not in num_c]
    c = Clu()
    c.pre = ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num_c),
        ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                        ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_c)])
    c.f, c.num_c, c.cat_c = f, num_c, cat_c
    c.X = c.pre.fit_transform(f)
    c.km = {k: KMeans(k, random_state=RS, n_init=10).fit(c.X) for k in KRANGE}
    c.inertia = {k: float(m.inertia_) for k, m in c.km.items()}
    c.sil = {"kmeans": {k: float(silhouette_score(c.X, m.labels_)) for k, m in c.km.items()}}
    c.sil["ward"] = {k: float(silhouette_score(c.X, AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(c.X)))
                     for k in KRANGE}
    c.best_k = max(c.sil["kmeans"], key=c.sil["kmeans"].get)
    c.Z = linkage(c.X, "ward")
    c.pca = PCA(2, random_state=RS)
    c.emb = {"PCA": c.pca.fit_transform(c.X)}
    c.fits = {}
    c.fields = [dict(name=n, label=lab, kind=kind, med=float(f[n].median()), min=float(f[n].min()), max=float(f[n].max()))
                for n, (lab, kind) in FIELDS.items() if n in f]
    c.cats = {n: dict(opts=sorted(f[n].dropna().astype(str).unique().tolist()), mode=str(f[n].mode()[0])) for n in cat_c}
    return c


def fit(c, algo, k):
    """Cluster with one algorithm at k. Segments are numbered biggest first."""
    if (algo, k) in c.fits:
        return c.fits[(algo, k)]
    raw = c.km[k].labels_ if algo == "kmeans" else AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(c.X)
    rank = {old: new for new, old in enumerate(np.argsort(-np.bincount(raw, minlength=k)))}
    lab = np.array([rank[v] for v in raw])
    cent = np.array([c.X[lab == i].mean(0) for i in range(k)])
    prof = c.f.assign(seg=lab).groupby("seg").mean(numeric_only=True)

    def level(s):
        r = s.rank(method="first").values
        if k == 2:
            return ["Low" if x == 1 else "High" for x in r]
        return [["Low", "Mid", "High"][min(2, int((x - 1) * 3 / k))] for x in r]

    tags = [f"{a} income, {b} spend" for a, b in zip(level(prof["Income"]), level(prof["TotalSpend"]))]
    table = pd.DataFrame({"Segment": [f"Segment {i + 1}" for i in range(k)], "Type": tags,
                          "Customers": np.bincount(lab, minlength=k)})
    for col, name in PROFILE.items():
        table[name] = prof[col].values
    metrics = {"Silhouette": float(silhouette_score(c.X, lab)),
               "Davies-Bouldin": float(davies_bouldin_score(c.X, lab)),
               "Calinski-Harabasz": float(calinski_harabasz_score(c.X, lab))}
    out = dict(labels=lab, cent=cent, tags=tags, table=table, metrics=metrics)
    c.fits[(algo, k)] = out
    return out


def roc_curves(art, model):
    """Per-class one-vs-rest ROC curves plus the macro average, from the saved test-set probabilities."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    r = art["rocs"][model]
    n = len(art["classes"])
    Y = label_binarize(r["y"], classes=list(range(n)))
    out, grid, tprs = {}, np.linspace(0, 1, 101), []
    for i, cname in enumerate(art["classes"]):
        if Y[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(Y[:, i], r["p"][:, i])
        out[cname] = (fpr, tpr, float(auc(fpr, tpr)))
        tprs.append(np.interp(grid, fpr, tpr))
    macro = np.mean(tprs, axis=0)
    return out, (grid, macro, float(auc(grid, macro)))


def tsne(c):
    if "t-SNE" not in c.emb:
        c.emb["t-SNE"] = TSNE(2, perplexity=30, init="pca", learning_rate="auto", random_state=RS).fit_transform(c.X)
    return c.emb["t-SNE"]


def dendro(c, k):
    cut = float((c.Z[-(k - 1), 2] + c.Z[-k, 2]) / 2)
    d = dendrogram(c.Z, truncate_mode="lastp", p=40, no_plot=True, color_threshold=cut)
    return d["icoord"], d["dcoord"], cut, d["color_list"]


def clu_predict(c, algo, k, values):
    """Place one new customer: transform it like the training data, then pick the nearest segment centre."""
    base = {n: c.f[n].median() for n in c.num_c}
    base.update({n: c.cats[n]["mode"] for n in c.cat_c})
    base.update(values)
    d = derive(pd.DataFrame([base]))[list(c.f.columns)]
    x = c.pre.transform(d)
    fi = fit(c, algo, k)
    dist = np.linalg.norm(fi["cent"] - x, axis=1)
    seg = int(np.argmin(dist))
    return dict(seg=seg, tag=fi["tags"][seg], dist=dist, pca=c.pca.transform(x)[0], derived=d.iloc[0])


if __name__ == "__main__":
    load_classification()
    print("Classification models trained and saved to cache/")
    if os.path.exists(CLU_CSV):
        c = build_clustering(open(CLU_CSV, "rb").read())
        print(f"Clustering ready ({len(c.f)} customers)")
