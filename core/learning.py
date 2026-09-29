class LearningProfile:
    def __init__(self): self.events=[]
    def record(self,kind,value):
        self.events.append((kind,str(value))); self.events=self.events[-200:]
    def summary(self):
        return {"analyses":sum(k=="analysis" for k,v in self.events),"questions":sum(k=="question" for k,v in self.events),"actions":len(self.events)}
