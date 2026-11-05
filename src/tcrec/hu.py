"""Conversione tra unità Hounsfield (HU) e coefficiente di attenuazione lineare mu.

mu(HU) = mu_water * (1 + HU / 1000).

ATTENZIONE: mu_water dipende dall'energia effettiva del fascio. Il valore di default
va verificato sulla letteratura (tabelle NIST XCOM) e dichiarato nel report. Il valore
è un parametro di configurazione, non una costante fisica universale per questo progetto.
"""

from __future__ import annotations

import numpy as np

MU_WATER_DEFAULT_MM = 0.0192  # mm^-1, indicativo per un fascio diagnostico; da verificare


def hu_to_mu(hu, mu_water: float = MU_WATER_DEFAULT_MM):
    hu = np.asarray(hu, dtype=np.float32)
    mu = mu_water * (1.0 + hu / 1000.0)
    return np.clip(mu, 0.0, None)  # l'aria (HU ~ -1000) dà mu ~ 0, mai negativo


def mu_to_hu(mu, mu_water: float = MU_WATER_DEFAULT_MM):
    mu = np.asarray(mu, dtype=np.float32)
    return 1000.0 * (mu / mu_water - 1.0)
