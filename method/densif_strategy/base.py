class BaseDensifStrategy:
    def __init__(self, conf, model):
        self.conf = conf
        self.model = model
        self.densif_stats = {"cloned":0,"split":0,"pruned":0,"total":0}

    def densify(self, iteration: int, **kwargs):
        raise NotImplementedError