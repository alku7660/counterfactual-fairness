from nnt import NN
from mo import MO
from cchvae import CCHVAE
from ijuice import IJUICE
from fijuice import FIJUICE
from foce_constraint import FOCE_CONSTRAINT
# from bigrace import BIGRACE
from bigrace_nocluster import BIGRACE
from ares import ARES
from facts import FACTS

class Counterfactual:

    # def __init__(self, data, model, method, cluster, alpha=1, beta=1, gamma=1, delta1=1, delta2=1, delta3=1, type='L1_L0', percentage_close_train_cf=0.1, support_th=0.01):
    def __init__(self, data, model, method, alpha=1, dev=False, eff=False, type='L1_L0', percentage_close_train_cf=0.1, support_th=0.01, continuous_bins=10, cluster=None):
        self.data = data
        self.model = model
        self.method = method
        self.type = type
        self.percentage = percentage_close_train_cf
        self.support_th = support_th
        self.continuous_bins = continuous_bins
        # self.lagrange = lagrange
        # self.likelihood_factor = likelihood_factor
        self.alpha, self.dev, self.eff = alpha, dev, eff
        # self.t = t
        # self.k = k
        self.cluster = cluster
        self.cf_method = self.select_cf_method()

    # def calculate_rho_min(self):
    #     rho_min = (max(self.graph.rho.values()) - min(self.graph.rho.values()))*self.likelihood_factor + min(self.graph.rho.values())
    #     return rho_min

    def select_cf_method(self):
        """
        Selects the method to find the counterfactual and stores it in "normal_x_cf"
        ['BIGRACE','ARES','FACTS']
        """
        # if self.method == 'fijuice':
        #     cf_method = FIJUICE(self)
        # if self.method == 'fijuice_like_constraint':
        #     cf_method = FOCE_CONSTRAINT(self)
        if 'BIGRACE' in self.method:
            cf_method = BIGRACE(self)
        if self.method == 'ARES':
            cf_method = ARES(self)
        if self.method == 'FACTS':
            cf_method = FACTS(self)
        # elif self.method == 'nn':
        #     cf_method = NN(self)
        # elif self.method == 'mo':
        #     cf_method = MO(self)
        # elif self.method == 'cchvae':
        #     cf_method = CCHVAE(self)
        # elif self.method == 'ijuice':
        #     cf_method = IJUICE(self)
        # elif self.method == 'ft':
        #     cf_method = FT(self)
        # elif self.method == 'rt':
        #     cf_method = RT(self)
        # elif self.method == 'gs':
        #     cf_method = GS(self)
        # elif self.method == 'face':
        #     cf_method = FACE(self)
        # elif self.method == 'dice':
        #     cf_method = DICE(self)
        # elif self.method == 'mace':
        #     cf_method = MACE(self)
        # elif self.method == 'juice':
        #     cf_method = JUICE(self)
        return cf_method