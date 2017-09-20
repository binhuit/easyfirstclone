class feat_record:
    def __init__(self):
        self.id = 0
        self.items = []
    def add(self,s, cls, f, c, p):
        item ={"score":s, "class":cls, "feats":f,"child":c,"parent":p}
        self.items.append(item)
    def save(self,filename):
        with open(filename+str(self.id),"w") as f:
            for item in self.items:
                f.write(str(item['score'])+"\n")
                f.write(str(item['class']) + "\n")
                f.write(str(item['child']['id']) + "\n")
                f.write(str(item['parent']['id']) + "\n")
                for feat in item['feats']:
                    f.write(feat+'\n')
        self.items = []
        self.id += 1
