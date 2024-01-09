# from .g1d_sinusoid import 
from .g2d_four_branch import g2D_four_branch
from .g2d_hmcmc_parabolic import g2d_parabolic
from .g2d_hmcmc_himmelblau import g2d_himmelblau
from .g100d_hmcmc_highdim import g100d_highdim
from .g11d_electric import g11d_electric
REGISTRY = {}

REGISTRY["g2d_four_branch"] = g2D_four_branch
REGISTRY["g2d_parabolic"] = g2d_parabolic
REGISTRY["g2d_himmelblau"] = g2d_himmelblau
REGISTRY["g100d_highdim"] = g100d_highdim
REGISTRY["g11d_electric"] = g11d_electric
