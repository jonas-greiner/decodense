import importlib

from .orbitals import set_adnp as orbitals_set_adnp
from .properties import set_adnp as properties_set_adnp
from .tools import set_adnp as tools_set_adnp


def load_adnp(ad):
    """
    this function loads the correct AD numpy package and sets it in the different
    modules
    """
    if not ad:
        adnp = importlib.import_module("numpy")
        addft = importlib.import_module("pyscf.dft")
        adiao = importlib.import_module("pyscf.lo.iao")
        adorth = importlib.import_module("pyscf.lo.orth")
    else:
        adnp = importlib.import_module("jax.numpy")
        addft = importlib.import_module("pyscfad.dft")
        adiao = importlib.import_module("pyscfad.lo.iao")
        adorth = importlib.import_module("pyscfad.lo.orth")

    orbitals_set_adnp(ad, adnp, adiao, adorth)
    properties_set_adnp(ad, adnp, addft)
    tools_set_adnp(adnp)
