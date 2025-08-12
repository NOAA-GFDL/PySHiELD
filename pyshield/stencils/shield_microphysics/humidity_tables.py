import numpy as np

import ndsl.constants as constants
import pyshield.constants as physcons


class HumiditySaturationTables:
    def __init__(
        self,
        length: int,
    ):
        self.length = length
        self.delt = 0.1
        self.n_min = 1600
        self._t0_init = False
        self._t2_init = False
        self.table0 = np.zeros(self.length)
        self.des0 = np.zeros(self.length)
        self.table2 = np.zeros(self.length)
        self.des2 = np.zeros(self.length)
        self._initialize_table0()
        self._initialize_table2()

    def _initialize_table0(self):
        # TODO: numpy-ify these init methods?
        tmin = physcons.TICE0 - 160.0

        for i in range(self.length):
            tem = tmin + self.delt * float(i)
            fac0 = (tem - physcons.TICE0) / (tem * physcons.TICE0)
            fac1 = fac0 * physcons.LV0
            fac2 = (
                constants.DC_VAP * np.log(tem / physcons.TICE0) + fac1
            ) / constants.RVGAS
            self.table0[i] = constants.E00 * np.exp(fac2)
            if i > 0:
                self.des0[i - 1] = max(0.0, self.table0[i] - self.table0[i - 1])

        self.des0[-1] = self.des0[-2]
        self._t0_init = True

    def _initialize_table2(self):
        if not self._t0_init:
            self._initialize_table0

        tmin = physcons.TICE0 - self.n_min * self.delt

        for i in range(self.length):
            if i < self.n_min:
                tem = tmin + self.delt * float(i)
                fac0 = (tem - physcons.TICE0) / (tem * physcons.TICE0)
                fac1 = fac0 * physcons.LI2
                fac2 = (
                    physcons.D2ICE * np.log(tem / physcons.TICE0) + fac1
                ) / constants.RVGAS
                self.table2[i] = constants.E00 * np.exp(fac2)
            else:
                self.table2[i] = self.table0[i - self.n_min]

            if i > 0:
                self.des2[i - 1] = max(0.0, self.table2[i] - self.table2[i - 1])

        self.des2[-1] = self.des2[-2]
        self._t2_init = True

    def _saturation(self, temperature, density, table: int):
        tmin = physcons.TICE0 - 160.0
        ap1 = 10.0 * np.maximum(0, temperature - tmin) + 1.0
        ap1 = np.minimum(self.length, ap1) - 1.0
        it = (ap1 - 0.5).astype(int)
        it2 = ap1.astype(int)

        if table not in [0, 2]:
            raise ValueError(f"table {table} not in [0, 2]")
        if table == 0:
            if not self._t0_init:
                self._initialize_table0
            qs_core = (self.table0[it2] + (ap1 - it2) * self.des0[it2]) / (
                constants.RVGAS * temperature * density
            )
            dqdt = (
                10.0
                * (self.des0[it] + (ap1 - it) * (self.des0[it + 1] - self.des0[it]))
                / (constants.RVGAS * temperature * density)
            )

        elif table == 2:
            if not self._t2_init:
                self._initialize_table2
            qs_core = (self.table2[it2] + (ap1 - it2) * self.des2[it2]) / (
                constants.RVGAS * temperature * density
            )
            dqdt = (
                10.0
                * (self.des2[it] + (ap1 - it) * (self.des2[it + 1] - self.des2[it]))
                / (constants.RVGAS * temperature * density)
            )

        else:
            raise ValueError(f"table {table} not in [0, 2]")

        return qs_core, dqdt

    def sat_water(self, temperature, density):
        wqs, dqdt = self._saturation(temperature, density, 0)
        return wqs, dqdt

    def sat_ice_water(self, temperature, density):
        iqs, dqdt = self._saturation(temperature, density, 2)
        return iqs, dqdt
