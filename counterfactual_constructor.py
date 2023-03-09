from nnt import NN
from mocopy import MO
from cchvae import CCHVAE
from ijuice import IJUICE


class Counterfactual:

    def __init__(self, data, model, method, ioi, type='euclidean', lagrange=0.5):
        self.data = data
        self.model = model
        self.method = method
        self.ioi = ioi
        self.type = type
        self.lagrange = lagrange
        self.cf_method = self.select_cf_method()

    def select_cf_method(self):
        """
        Method that selects the method to find the counterfactual and stores it in "normal_x_cf"
        ['nn','mo','ft','rt','gs','face','dice','mace','cchvae','juice','ijuice']
        """
        if self.method == 'nn':
            cf_method = NN(self)
        elif self.method == 'mo':
            cf_method = MO(self)
        elif self.method == 'cchvae':
            cf_method = CCHVAE(self)
        elif self.method == 'ijuice':
            cf_method = IJUICE(self)
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