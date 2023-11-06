from nnt import NN
from mo import MO
from cchvae import CCHVAE
from ijuice import IJUICE
from fijuice import FIJUICE
from fijuice_likelihood_constraint import FIJUICE_LIKE_CONSTRAINT
from fijuice_likelihood_optimize import FIJUICE_LIKE_OPTIMIZE
from ares import ARES

class Counterfactual:

    def __init__(self, data, model, method, cluster, lagrange, likelihood_factor=0.2, alpha=1, beta=1, gamma=1, type='L1_L0', t=100, k=10, graph=None):
        self.data = data
        self.model = model
        self.method = method
        self.cluster = cluster
        self.type = type
        self.lagrange = lagrange
        self.likelihood_factor = likelihood_factor
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.t = t
        self.k = k
        self.graph = graph
        self.rho_min = self.calculate_rho_min()
        self.cf_method = self.select_cf_method()

    def calculate_rho_min(self):
        rho_min = (max(self.graph.rho.values()) - min(self.graph.rho.values()))*self.likelihood_factor + min(self.graph.rho.values())
        return rho_min

    def select_cf_method(self):
        """
        Selects the method to find the counterfactual and stores it in "normal_x_cf"
        ['nn','mo','ft','rt','gs','face','dice','cchvae','juice','ijuice','fijuice_like_constraint','fijuice_like_optimize','ares']
        """
        if self.method == 'fijuice':
            cf_method = FIJUICE(self)
        if self.method == 'fijuice_like_constraint':
            cf_method = FIJUICE_LIKE_CONSTRAINT(self)
        if self.method == 'fijuice_like_optimize':
            cf_method = FIJUICE_LIKE_OPTIMIZE(self)
        if self.method == 'ares':
            cf_method = ARES(self)
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