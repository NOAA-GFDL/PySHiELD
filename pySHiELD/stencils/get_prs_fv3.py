from gt4py.cartesian.gtscript import PARALLEL, computation, interval

from ndsl.constants import ZVIR
from ndsl.dsl.typing import FloatField


def get_prs_fv3(
    phii: FloatField,
    prsi: FloatField,
    tgrs: FloatField,
    qgrs: FloatField,
    del_: FloatField,
    del_gz: FloatField,
):
    # Passing with integration, but zero padding is different from fortran for del_gz
    with computation(PARALLEL), interval(0, -1):
        del_ = prsi[0, 0, 1] - prsi[0, 0, 0]
        del_gz = (phii[0, 0, 0] - phii[0, 0, 1]) / (
            tgrs[0, 0, 0] * (1.0 + ZVIR * max(0.0, qgrs[0, 0, 0]))
        )
