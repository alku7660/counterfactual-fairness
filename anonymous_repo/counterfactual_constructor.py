"""
Imports
"""
from counterfair import COUNTERFAIR

class Counterfactual:

    def __init__(self, data, model, method, alpha=1, dev=False, eff=False, type='L1_L0', percentage_close_train_cf=0.1, support_th=0.01, continuous_bins=10, cluster=None):
        self.data = data
        self.model = model
        self.method = method
        self.type = type
        self.percentage = percentage_close_train_cf
        self.support_th = support_th
        self.continuous_bins = continuous_bins
        self.alpha, self.dev, self.eff = alpha, dev, eff
        self.cluster = cluster
        self.cf_method = self.select_cf_method()

    def select_cf_method(self):
        """
        Selects the method to find the counterfactual and stores it in "normal_x_cf"
        ['CounterFair','ARES','FACTS']
        """
        if 'CounterFair' in self.method:
            cf_method = COUNTERFAIR(self)
        return cf_method