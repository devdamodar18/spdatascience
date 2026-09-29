import pandas as pd
import numpy as np

class SPDSEngine:
    def __init__(self,df): self.df=df.copy()
    def profile(self):
        return {"rows":len(self.df),"columns":len(self.df.columns),"missing":int(self.df.isna().sum().sum()),"duplicates":int(self.df.duplicated().sum())}
    def quality(self):
        return pd.DataFrame([{"column":c,"dtype":str(self.df[c].dtype),"missing":int(self.df[c].isna().sum()),"missing_pct":round(self.df[c].isna().mean()*100,2),"unique":int(self.df[c].nunique(dropna=True))} for c in self.df.columns])
    def correlations(self):
        n=self.df.select_dtypes(include=np.number)
        if n.shape[1]<2:return pd.DataFrame()
        c=n.corr(); rows=[]; cols=list(c.columns)
        for i in range(len(cols)):
            for j in range(i+1,len(cols)):
                rows.append({"column_1":cols[i],"column_2":cols[j],"correlation":round(float(c.iloc[i,j]),3)})
        return pd.DataFrame(rows).sort_values("correlation",key=lambda s:s.abs(),ascending=False)
    def anomalies(self):
        rows=[]
        for c in self.df.select_dtypes(include=np.number):
            s=self.df[c].dropna()
            if len(s)<5: continue
            q1,q3=s.quantile([.25,.75]); iqr=q3-q1
            if iqr==0: continue
            mask=(self.df[c]<q1-1.5*iqr)|(self.df[c]>q3+1.5*iqr)
            rows.append({"column":c,"candidate_count":int(mask.fillna(False).sum())})
        return pd.DataFrame(rows)
    def charts(self):
        n=self.df.select_dtypes(include=np.number).columns.tolist(); cat=[c for c in self.df.columns if c not in n]; out=[]
        for c in n[:4]: out.append({"type":"hist","x":c,"label":f"Distribution of {c}","reason":"Shows distribution and possible outliers."})
        if len(n)>1: out.append({"type":"scatter","x":n[0],"y":n[1],"label":f"{n[1]} vs {n[0]}","reason":"Shows relationships and clusters."})
        if cat and n: out.append({"type":"bar","x":cat[0],"y":n[0],"label":f"{n[0]} by {cat[0]}","reason":"Compares a numeric measure across categories."})
        return out
    def auto_analysis(self):
        p=self.profile(); c=self.correlations(); a=self.anomalies(); summary=[f"Found {p['rows']:,} rows and {p['columns']:,} columns."]
        summary.append(f"Found {p['missing']:,} missing cells and {p['duplicates']:,} duplicate rows.")
        if not c.empty:
            r=c.iloc[0]; summary.append(f"Strongest observed numeric relationship: {r['column_1']} vs {r['column_2']} (correlation {r['correlation']:.2f}).")
        if not a.empty:
            r=a.sort_values("candidate_count",ascending=False).iloc[0]; summary.append(f"Most anomaly candidates: {r['column']} ({int(r['candidate_count'])}).")
        return {"summary":summary,"quality":self.quality(),"correlations":c,"charts":self.charts()}
    def findings(self):
        r=self.auto_analysis(); return r["summary"]+["Correlation is not causation; validate important findings with domain context."]
    def next_steps(self):
        return ["Review data quality.","Explore distributions and relationships.","Segment important metrics.","Use a meaningful target for baseline prediction.","Validate findings with domain context."]
    def predict(self,target):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
        from sklearn.metrics import r2_score,accuracy_score
        d=self.df.dropna(subset=[target])
        if len(d)<20:return {"ok":False,"message":"At least 20 usable rows are recommended."}
        X,y=d.drop(columns=[target]),d[target]; num=X.select_dtypes(include=np.number).columns.tolist(); cat=[c for c in X.columns if c not in num]
        parts=[]
        if num:parts.append(("num",SimpleImputer(strategy="median"),num))
        if cat:parts.append(("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat))
        classification=(not pd.api.types.is_numeric_dtype(y)) or (1<y.nunique()<=10)
        model=RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1) if classification else RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1)
        pipe=Pipeline([("prep",ColumnTransformer(parts)),("model",model)])
        try:
            xt,xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42,stratify=y if classification and y.value_counts().min()>=2 else None)
            pipe.fit(xt,yt); pred=pipe.predict(xv); score=accuracy_score(yv,pred) if classification else r2_score(yv,pred)
            names=num[:]
            if cat:names+=pipe.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat).tolist()
            imp=pd.DataFrame({"feature":names,"importance":pipe.named_steps["model"].feature_importances_}).sort_values("importance",ascending=False).head(20)
            return {"ok":True,"model":"Random Forest classifier" if classification else "Random Forest regressor","score":f"{score:.3f} ({'accuracy' if classification else 'R²'})","explanation":"Baseline validation only; it does not guarantee future performance.","importance":imp}
        except Exception as e:return {"ok":False,"message":str(e)}
