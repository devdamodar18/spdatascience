import plotly.express as px
import numpy as np

class SPDSAssistant:
    def __init__(self,df,engine): self.df=df; self.engine=engine
    def answer(self,q):
        q=q.lower(); p=self.engine.profile()
        if "explain" in q or "beginner" in q:
            return f"This dataset has {p['rows']:,} rows, {p['columns']:,} columns, {p['missing']:,} missing cells and {p['duplicates']:,} duplicate rows. Start with quality checks, then distributions and relationships."
        if "quality" in q or "problem" in q or "missing" in q:
            return f"I found {p['missing']:,} missing cells and {p['duplicates']:,} duplicate rows. Review these before important modeling."
        if "pattern" in q or "relationship" in q:
            c=self.engine.correlations()
            if c.empty:return "There are not enough numeric fields for relationship analysis."
            r=c.iloc[0]; return f"The strongest observed numeric relationship is {r['column_1']} vs {r['column_2']}, correlation {r['correlation']:.2f}. This does not establish causation."
        if "chart" in q or "visual" in q:
            return "I recommend: "+", ".join(x["label"] for x in self.chart_options()[:4])
        if "unusual" in q or "outlier" in q or "anomal" in q:
            a=self.engine.anomalies()
            if a.empty:return "No numeric anomaly candidates were detected with the IQR rule."
            r=a.sort_values("candidate_count",ascending=False).iloc[0]; return f"{r['column']} has {int(r['candidate_count'])} candidate outlier rows. Investigate them before removing anything."
        if "next" in q or "what should" in q:return "Suggested sequence: quality → distributions → relationships → segmentation → baseline model → validation."
        return "I can explain your data, check quality, find patterns, recommend charts, detect anomaly candidates and suggest next analyses."
    def chart_options(self): return self.engine.charts()
    def make_chart(self,s):
        if s["type"]=="hist":f=px.histogram(self.df,x=s["x"],marginal="box",title=s["label"])
        elif s["type"]=="scatter":f=px.scatter(self.df,x=s["x"],y=s["y"],trendline="ols",title=s["label"])
        else:
            t=self.df.groupby(s["x"],dropna=False)[s["y"]].mean().reset_index().sort_values(s["y"],ascending=False).head(20)
            f=px.bar(t,x=s["x"],y=s["y"],title=s["label"])
        f.update_layout(height=520); return f
