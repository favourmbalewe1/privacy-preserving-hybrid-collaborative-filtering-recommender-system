# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy.sparse import csr_matrix
import diffprivlib as dp
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

MAX_USERS    = 2000
MAX_ITEMS    = 3000
MAX_SAMPLE   = 20000
TOP_CATS     = 15
SIM_DIM      = 300
RANDOM_STATE = 42

KNOWN_USER_COLS  = ("user_id","userid","customerid","customer_id",
                    "visitorid","visitor_id","uid")
KNOWN_ITEM_COLS  = ("item_id","itemid","stockcode","stock_code",
                    "movieid","movie_id","productid","product_id")
KNOWN_SCORE_COLS = ("interaction_score","rating","quantity","score","value")
KNOWN_CAT_COLS   = ("category","genres","genre","description",
                    "categoryid","category_id")


def _detect(columns, candidates):
    cl = [c.lower().strip() for c in columns]
    for cand in candidates:
        if cand in cl:
            return columns[cl.index(cand)]
    for cand in candidates:
        for c in columns:
            if cand in c.lower():
                return c
    return None


def _normalise(df):
    orig = list(df.columns)
    u = _detect(orig, KNOWN_USER_COLS)
    i = _detect(orig, KNOWN_ITEM_COLS)
    s = _detect(orig, KNOWN_SCORE_COLS)
    c = _detect(orig, KNOWN_CAT_COLS)
    if u is None or i is None:
        raise KeyError(
            "Cannot detect user/item columns. Found: " + str(orig) +
            ". Rename to user_id / item_id.")
    rename = {}
    if u != "user_id":              rename[u] = "user_id"
    if i != "item_id":              rename[i] = "item_id"
    if s and s != "interaction_score": rename[s] = "interaction_score"
    if c and c != "category":          rename[c] = "category"
    if rename:
        df = df.rename(columns=rename)
        print("[NORM] Renamed: " + str(rename))
    if "interaction_score" not in df.columns:
        df["interaction_score"] = 1.0
    if "category" not in df.columns:
        df["category"] = "Unknown"
    df["user_id"]           = df["user_id"].astype(str).str.strip()
    df["item_id"]           = df["item_id"].astype(str).str.strip()
    df["interaction_score"] = pd.to_numeric(df["interaction_score"],
                                             errors="coerce").fillna(1.0)
    df["category"]          = df["category"].fillna("Unknown").astype(str).str.strip()
    return df


class HybridCFPipeline:

    def __init__(self, epsilon=1.0, n_top=10, random_state=RANDOM_STATE):
        self.epsilon      = epsilon
        self.n_top        = n_top
        self.random_state = random_state
        self.le_user      = LabelEncoder()
        self.le_item      = LabelEncoder()
        self.le_cat       = LabelEncoder()
        self.scaler       = StandardScaler()
        self.svm_model    = None
        self.user_sim     = None
        self.item_sim     = None
        self.mat          = None
        self.df           = None
        self.items_meta   = None
        self.unique_items = None

    def load_data(self, path):
        df = pd.read_csv(path)
        df = _normalise(df)
        top_cats  = df["category"].value_counts().nlargest(TOP_CATS).index.tolist()
        df        = df[df["category"].isin(top_cats)].reset_index(drop=True)
        top_users = df["user_id"].value_counts().nlargest(MAX_USERS).index.tolist()
        top_items = df["item_id"].value_counts().nlargest(MAX_ITEMS).index.tolist()
        df        = df[df["user_id"].isin(top_users) &
                       df["item_id"].isin(top_items)].reset_index(drop=True)
        if len(df) > MAX_SAMPLE:
            df = df.sample(MAX_SAMPLE, random_state=self.random_state).reset_index(drop=True)
        valid = df["category"].value_counts()
        df    = df[df["category"].isin(valid[valid >= 5].index)].reset_index(drop=True)
        self.df           = df
        self.unique_items = df["item_id"].unique()
        self.items_meta   = (
            df.groupby("item_id")
            .agg(avg_score=("interaction_score","mean"),
                 category=("category","first"),
                 popularity=("interaction_score","count"))
            .reset_index()
        )
        print("[DATA] Rows=" + str(len(df)) +
              "  Users=" + str(df["user_id"].nunique()) +
              "  Items=" + str(df["item_id"].nunique()) +
              "  Cats="  + str(df["category"].nunique()))
        return df

    def _build_matrix(self):
        ue       = self.le_user.fit_transform(self.df["user_id"])
        ie       = self.le_item.fit_transform(self.df["item_id"])
        scores   = self.df["interaction_score"].values
        n_u      = len(self.le_user.classes_)
        n_i      = len(self.le_item.classes_)
        self.mat = csr_matrix((scores, (ue, ie)), shape=(n_u, n_i))
        sp       = 1.0 - self.mat.nnz / max(1, n_u * n_i)
        print("[MATRIX] shape=" + str(self.mat.shape) +
              "  nnz=" + str(self.mat.nnz) +
              "  sparsity=" + str(round(sp, 4)))

    def _fast_cosine(self, M, max_dim=SIM_DIM):
        if hasattr(M, "toarray"):
            M = M.toarray()
        M     = M[:max_dim].astype(np.float32)
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        Mn    = M / norms
        return np.clip((Mn @ Mn.T).astype(np.float32), 0.0, 1.0)

    def _build_similarities(self):
        self.user_sim = self._fast_cosine(self.mat,   SIM_DIM)
        self.item_sim = self._fast_cosine(self.mat.T, SIM_DIM)
        print("[SIM] user_sim=" + str(self.user_sim.shape) +
              "  item_sim=" + str(self.item_sim.shape))

    def _apply_dp(self, matrix):
        if self.epsilon >= 9.9:
            return matrix
        sensitivity = float(np.max(np.abs(matrix))) if matrix.max() > 0 else 1.0
        mech        = dp.mechanisms.Laplace(epsilon=self.epsilon,
                                             sensitivity=sensitivity)
        return np.vectorize(lambda x: mech.randomise(float(x)))(matrix)

    def _user_item_affinity(self, user_enc_idx, item_enc_idx, noisy_user_sim):
        u_idx_c = min(user_enc_idx, noisy_user_sim.shape[0] - 1)
        user_sim_row = noisy_user_sim[u_idx_c]
        if item_enc_idx < self.mat.shape[1]:
            item_col = self.mat.getcol(item_enc_idx).toarray().flatten()
            n_item   = len(item_col)
            sim_len  = len(user_sim_row)
            overlap  = min(n_item, sim_len)
            affinity = float(np.dot(user_sim_row[:overlap],
                                     item_col[:overlap]))
            max_poss = float(item_col[:overlap].sum()) + 1e-10
            return affinity / max_poss
        return 0.0

    def _build_features(self):
        self._build_matrix()
        self._build_similarities()
        ue = self.le_user.transform(self.df["user_id"])
        ie = self.le_item.transform(self.df["item_id"])
        ce = self.le_cat.fit_transform(self.df["category"])
        nu = self._apply_dp(self.user_sim)
        ni = self._apply_dp(self.item_sim)
        u_idx = np.minimum(ue, nu.shape[0] - 1)
        i_idx = np.minimum(ie, ni.shape[0] - 1)
        ucf   = nu[u_idx]
        icf   = ni[i_idx]
        u_act = np.array([float(self.mat.getrow(u).nnz)
                          for u in u_idx]).reshape(-1, 1)
        i_pop = np.array([float(self.mat.getcol(i).nnz)
                          if i < self.mat.shape[1] else 0.0
                          for i in i_idx]).reshape(-1, 1)
        aff   = np.array([self._user_item_affinity(u, i, nu)
                          for u, i in zip(u_idx, i_idx)]).reshape(-1, 1)
        X = np.hstack([
            ucf.mean(axis=1, keepdims=True),
            ucf.max(axis=1,  keepdims=True),
            (ucf > 0.01).sum(axis=1, keepdims=True).astype(np.float32),
            icf.mean(axis=1, keepdims=True),
            icf.max(axis=1,  keepdims=True),
            u_act, i_pop, aff,
            ce.reshape(-1, 1),
            self.df["interaction_score"].values.reshape(-1, 1)
        ])
        return X, ce

    def _safe_stratify(self, y):
        uniq, cnts = np.unique(y, return_counts=True)
        if cnts.min() < 2:
            print("[WARN] Stratify disabled: class has < 2 samples.")
            return None
        if int(len(y) * 0.2 / len(uniq)) < 1:
            print("[WARN] Stratify disabled: too few samples per class.")
            return None
        return y

    def train(self, path):
        self.load_data(path)
        X, y     = self._build_features()
        X_scaled = self.scaler.fit_transform(X)
        stratify = self._safe_stratify(y)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_scaled, y, test_size=0.2,
            random_state=self.random_state, stratify=stratify)
        print("[SPLIT] train=" + str(len(X_tr)) + "  test=" + str(len(X_te)))
        self.svm_model = SVC(kernel="rbf", C=1.0, gamma="scale",
                              decision_function_shape="ovo",
                              probability=True,
                              random_state=self.random_state)
        print("[TRAIN] Fitting SVM...")
        self.svm_model.fit(X_tr, y_tr)
        print("[TRAIN] Done.")
        y_pred  = self.svm_model.predict(X_te)
        metrics = {
            "accuracy":  float(accuracy_score(y_te, y_pred)),
            "precision": float(precision_score(y_te, y_pred,
                                                average="weighted", zero_division=0)),
            "recall":    float(recall_score(y_te, y_pred,
                                             average="weighted", zero_division=0)),
            "f1":        float(f1_score(y_te, y_pred,
                                         average="weighted", zero_division=0)),
            "epsilon":   self.epsilon,
        }
        return metrics

    def predict_for_user(self, user_id):
        if self.svm_model is None or self.df is None:
            raise RuntimeError("Model not trained.")
        user_id = str(user_id).strip()
        if user_id not in self.le_user.classes_:
            user_id = self.le_user.classes_[0]

        uidx    = self.le_user.transform([user_id])[0]
        row     = self.mat.getrow(uidx).toarray().flatten()
        done    = (set(self.le_item.inverse_transform(np.where(row > 0)[0]))
                   if row.any() else set())

        candidates = [it for it in self.unique_items if it not in done]
        if not candidates:
            candidates = list(self.unique_items)

        np.random.seed(uidx % 9999)
        if len(candidates) > 600:
            candidates = list(np.random.choice(candidates, 600, replace=False))

        noisy_u = self._apply_dp(self.user_sim)
        noisy_i = self._apply_dp(self.item_sim)

        u_idx_c      = min(uidx, noisy_u.shape[0] - 1)
        user_cf_row  = noisy_u[u_idx_c]
        u_mean       = float(user_cf_row.mean())
        u_max        = float(user_cf_row.max())
        u_nnz        = float((user_cf_row > 0.01).sum())
        user_activity = float(self.mat.getrow(uidx).nnz)

        max_pop = float(self.items_meta["popularity"].max()) if len(self.items_meta) > 0 else 1.0
        max_avg = float(self.items_meta["avg_score"].max())  if len(self.items_meta) > 0 else 5.0

        records = []
        for item in candidates:
            iidx    = (self.le_item.transform([item])[0]
                       if item in self.le_item.classes_ else 0)
            i_idx_c = min(iidx, noisy_i.shape[0] - 1)
            icf_row = noisy_i[i_idx_c]
            i_mean  = float(icf_row.mean())
            i_max   = float(icf_row.max())
            i_pop   = float(self.mat.getcol(iidx).nnz
                            if iidx < self.mat.shape[1] else 0)

            affinity = self._user_item_affinity(uidx, iidx, noisy_u)

            meta    = self.items_meta[self.items_meta["item_id"] == item]
            cat_str = meta["category"].values[0] if len(meta) > 0 else "Unknown"
            cat_enc = (self.le_cat.transform([cat_str])[0]
                       if cat_str in self.le_cat.classes_ else 0)
            avg_val = float(meta["avg_score"].values[0]) if len(meta) > 0 else 1.0

            feat    = np.array([[u_mean, u_max, u_nnz,
                                  i_mean, i_max,
                                  user_activity, i_pop, affinity,
                                  cat_enc, avg_val]], dtype=np.float32)
            feat_s  = self.scaler.transform(feat)
            proba   = self.svm_model.predict_proba(feat_s)[0]

            svm_conf      = float(np.max(proba))
            pop_norm      = i_pop / max(1.0, max_pop)
            avg_norm      = avg_val / max(1.0, max_avg)
            affinity_norm = min(1.0, affinity * 10.0)

            combined = (
                0.40 * affinity_norm +
                0.25 * svm_conf      +
                0.20 * avg_norm      +
                0.15 * pop_norm
            )

            records.append({
                "item_id":         item,
                "category":        cat_str,
                "score":           round(combined, 4),
                "avg_interaction": avg_val,
                "item_popularity": int(i_pop),
            })

        result = (
            pd.DataFrame(records)
            .sort_values(["score","avg_interaction"], ascending=[False, False])
            .head(self.n_top)
            .reset_index(drop=True)
        )
        result.index += 1
        return result

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        out = os.path.join(path, "hybrid_cf_model.pkl")
        joblib.dump(self, out)
        print("[SAVED] " + out)

    @staticmethod
    def load(path="model_artifacts/hybrid_cf_model.pkl"):
        return joblib.load(path)
