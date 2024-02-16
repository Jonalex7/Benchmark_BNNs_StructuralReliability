from .bnn_bpp import BNN_BayesBackProp
from .mc_droput import NeuralNetworkWithDropout
from .ensemble import Ensemble
from .sghmc import BNN_SGHMC

REGISTRY = {}

REGISTRY["bnnbpp"] = BNN_BayesBackProp
REGISTRY["dropout"] = NeuralNetworkWithDropout
REGISTRY["ensembles"] = Ensemble
REGISTRY["sghmc"] = BNN_SGHMC