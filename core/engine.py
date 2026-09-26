import numpy as np
import pandas as pd

class SPDSEngine:
    def __init__(self, df):
        self.df = df.copy()

    def profile(self):
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "missing_cells": int(self.df.isna().sum().sum()),
            "duplicate_rows": int(self.df.duplicated().sum()),
        }

    def column_profile(self):
        return pd.DataFrame([{
            "column": c,
            "dtype": str(self.df[c].dtype),
            "missing": int(self.df[c].isna().sum()),
            "missing_pct": round(self.df[c].isna().mean()*100, 2),
            "unique": int(self.df[c].nunique(dropna=True))
        } for c in self.df.columns])

    def quality_report(self):
        return pd.DataFrame([{
            "column": c,
            "missing": int(self.df[c].isna().sum()),
            "missing_pct": round(self.df[c].isna().mean()*100, 2),
            "unique": int(self.df[c].nunique(dropna=True)),
            "status": "Review" if self.df[c].isna().any() or self.df[c].nunique(dropna=True) <= 1 else "OK"
        } for c in self.df.columns])

    def numeric_summary(self):
        n = self.df.select_dtypes(include=np.number)
        return n.describe().T.reset_index().rename(columns={"index":"column"}).round(4) if not n.empty else pd.DataFrame()

    def correlation_matrix(self):
        n = self.df.select_dtypes(include=np.number)
        return n.corr().round(3) if n.shape[1] >= 2 else pd.DataFrame()

    def anomaly_report(self):
        n = self.df.select_dtypes(include=np.number)
        rows = []
        for c in n.columns:
            s = n[c].dropna()
            if len(s) < 5: continue
            q1, q3 = s.quantile([.25, .75])
            iqr = q3-q1
            if iqr == 0: continue
            lo, hi = q1-1.5*iqr, q3+1.5*iqr
            mask = (self.df[c] < lo) | (self.df[c] > hi)
            rows.append({"column":c, "lower_bound":round(float(lo),4),
                         "upper_bound":round(float(hi),4),
                         "anomaly_count":int(mask.fillna(False).sum()),
                         "anomaly_pct":round(mask.fillna(False).mean()*100,2)})
        return pd.DataFrame(rows)

    def predict(self, target):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.metrics import r2_score, accuracy_score

        data = self.df.dropna(subset=[target])
        if len(data) < 20:
            return {"ok":False, "message":"At least 20 usable rows are recommended."}
        X, y = data.drop(columns=[target]), data[target]
        if X.shape[1] == 0:
            return {"ok":False, "message":"No feature columns remain."}

        num = X.select_dtypes(include=np.number).columns.tolist()
        cat = [c for c in X.columns if c not in num]
        parts = []
        if num: parts.append(("num", SimpleImputer(strategy="median"), num))
        if cat:
            parts.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat))
        prep = ColumnTransformer(parts)
        classification = (not pd.api.types.is_numeric_dtype(y)) or (1 < y.nunique() <= 10)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) if classification else RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        pipe = Pipeline([("prep", prep), ("model", model)])

        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42,
                stratify=y if classification and y.value_counts().min() >= 2 else None)
            pipe.fit(Xtr, ytr)
            pred = pipe.predict(Xte)
            score = accuracy_score(yte, pred) if classification else r2_score(yte, pred)
            names = num[:]
            if cat:
                enc = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
                names += enc.get_feature_names_out(cat).tolist()
            fi = pd.DataFrame({"feature":names, "importance":pipe.named_steps["model"].feature_importances_})
            return {"ok":True,
                    "model":"Random Forest classifier" if classification else "Random Forest regressor",
                    "score_text":f"{score:.3f} ({'accuracy' if classification else 'R²'})",
                    "explanation":"Baseline validation only; this does not guarantee future performance.",
                    "feature_importance":fi.sort_values("importance", ascending=False).head(20)}
        except Exception as e:
            return {"ok":False, "message":f"Modeling failed: {e}"}

    def business_findings(self):
        out=[]
        p=self.profile()
        if p["missing_cells"]: out.append(f"{p['missing_cells']:,} missing cells should be reviewed.")
        if p["duplicate_rows"]: out.append(f"{p['duplicate_rows']:,} duplicate rows were detected.")
        n=self.df.select_dtypes(include=np.number)
        if not n.empty:
            s=n.std().sort_values(ascending=False)
            if not s.empty: out.append(f"Highest numeric dispersion: '{s.index[0]}'.")
        c=self.correlation_matrix()
        if not c.empty:
            vals=[]
            cols=list(c.columns)
            for i in range(len(cols)):
                for j in range(i+1,len(cols)):
                    if pd.notna(c.iloc[i,j]): vals.append((abs(c.iloc[i,j]), c.iloc[i,j], cols[i], cols[j]))
            if vals:
                _,v,a,b=max(vals)
                out.append(f"Strongest observed pairwise correlation: '{a}' vs '{b}' = {v:.2f}. Correlation is not causation.")
        return out or ["No major automatic findings were triggered."]

    def next_steps(self):
        return [
            "Validate findings against business definitions and domain context.",
            "Segment important metrics by time, product, geography, customer, or other meaningful dimensions.",
            "Add time-indexed and external data for forecasting and market-intelligence workflows.",
            "Add authentication, authorization, audit logs, monitoring, and secure connectors before enterprise use."
        ]
