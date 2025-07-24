import dataclasses
from pathlib import Path

import numpy as np
import pyrte_rrtmgp as rad

from ndsl import (
    X_DIM,
    Y_DIM,
    Z_DIM,
    Bool,
    Float,
    FloatField,
    FloatFieldIJ,
    Int,
    QuantityFactory,
    StencilFactory,
)
from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, computation, interval, log
import ndsl.constants as constants

from .rad_astro import coszmn, sol_init, solar_update
from .rad_clouds import cld_init, progcld4, progcld5
from .rad_gases import co2_update, gas_init, get_gases_bottomup, get_gases_topdown
from .rad_sfc import set_albedo, set_sfcemis, sfc_init
from .radiation_state import RadiationState
from pySHiELD.physics_state import SurfaceState


GRAV = 9.80665
CP_DRY = 1004.64
QMIN=1.0e-10
QME5=1.0e-7
QME6=1.0e-7

def calc_tlvl(
    plyr: FloatField,
    plvl: FloatField,
    tgrs: FloatField,
    tskin: FloatFieldIJ,
    qvapor: FloatField,
    tvly: FloatField,
    tsfa: FloatFieldIJ,
    tlvl: FloatField,
):
    """
    Calculates interface(level) temperatures neede for radiation
    """
    with computation(FORWARD):
        with interval(0, 1):
            tsfa = tgrs
            tlvl = tskin
            tem2da = log(plyr)
            tem2db = log(plvl)
            qvapor = max(QME6, qvapor)
            tvly = tgrs * (1.0 + constants.ZVIR * qvapor)
        with interval(1, -1):
            qvapor = max(QME6, qvapor)
            tvly = tgrs * (1.0 + constants.ZVIR * qvapor)
            tlvl = tgrs[0, 0, -1] + (tgrs - tgrs[0, 0, -1]) * (
                log(plvl) - log(plyr[0, 0, -1])
            ) / (log(plyr) - log(plyr[0, 0, -1]))
        with interval(-1, None):
            tlvl = tgrs[0, 0, -1]

def calc_heating(
    flux_up: FloatField,
    flux_down: FloatField,
    p_lev: FloatField,
    heating_rate: FloatField,
):
    """
    Calculates heating rates based on pressures and fluxes,
    assuming k increases with height

    Args:
        flux_up: upward flux
        flux_down: downward flux
        p_lev: model interface pressure
        heating_rate: layer heating rate
    """
    with computation(PARALLEL), interval(0, -1):
        heating_rate = (
            (flux_up[0, 0, 1] - flux_up - flux_down[0, 0, 1] + flux_down[0, 0, 0])
            * GRAV
            / (CP_DRY * (p_lev[0, 0, 1] - p_lev))
        )


@dataclasses.dataclass
class RadiationConfig:
    dt_atmos: Float
    date: list
    fhswr: Float
    fhlwr: Float
    isolar: Int
    icmphys: Int
    ico2flg: Int
    ioznflg: Int
    ictmflg: Int
    ialbflg: Int
    iemsflg: Int
    ldisable_radiation_quasi_sea_ice: bool
    solar_constant_file: Path
    input_dir: Path
    aerosol_file: Path
    daily_mean: bool
    fixed_sollat: bool
    sollat: Float
    nstp: Int
    ivflip: Int
    lcnorm: bool
    lcrick: bool
    gfs_cloud_overlap: bool
    deltsw: Float
    delt_rad: Float

    def __post_init__(self):
        if self.ioznflg == 0:
            raise NotImplementedError(
                "climatological ozone (ioznflg = 0) is not supported"
            )
        if len(self.date) != 6:
            raise ValueError(
                "Initial date must be of format "
                f"[year, month, day, hour, minute, second], got {self.date}"
            )
        self.date = [int(digit) for digit in self.date]


class RTE_RRTMGPDriver:
    def __init__(
        self,
        config: RadiationConfig,
        gridlon: FloatFieldIJ,
        gridlat: FloatFieldIJ,
        sigma: np.ndarray,
        quantity_factory: QuantityFactory,
        stencil_factory: StencilFactory,
    ):
        grid_indexing = stencil_factory.grid_indexing
        iyear = config.date[0]
        imonth = config.date[1]
        iday = config.date[2]
        ihr = config.date[3]
        self.saved_iyear = iyear
        self.saved_imonth = imonth
        self.saved_iday = iday
        self.deltsw = config.deltsw
        self.delt_rad = config.delt_rad
        self.isolar = config.isolar
        self.ico2flg = config.ico2flg
        self.ictmflg = config.ictmflg
        self.ialbflg = config.ialbflg
        self.ldisable_radiation_quasi_sea_ice = config.ldisable_radiation_quasi_sea_ice
        self._first_step = True

        self.solhr = ihr
        self.slag = 0.0
        self.sdec = 0.0
        self.cdec = 0.0
        self.anginc = 0.0
        self.solcon = 0.0
        self.solc0 = 0.0
        self.nstp = 0

        # Allocate quantities
        self._coszdg = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            "radians",
            dtype=Float,
        )
        self._daymask = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            "",
            dtype=Bool,
        )
        self._co2_cyc = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            "",
            dtype=Bool,
        )
        self._co2_arr = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            "",
            dtype=Bool,
        )
        self._tvly = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            "degK",
            dtype=Float,
        )
        self._tsfca = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            "degK",
            dtype=Float,
        )
        self._cnvw = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            "",
            dtype=Float,
        )
        self._cnvc = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            "",
            dtype=Float,
        )

        self.gridlon = gridlon
        self.gridlat = gridlat
        # Init solar params
        self.isolflg, self._solar_constants, self.solc0 = sol_init(
            self.isolar,
            config.solar_constant_file,
            iyear,
        )
        # Here is where we will initialize aerosols once they're supported

        # Init gases
        (
            self.n2o,
            self.ch4,
            self.o2,
            self.co,
            self.n2,
            self.cfc11,
            self.cfc12,
            self.cfc22,
            self.ccl4,
            self.co2_glb,
            co2_arr,
            co2_cyc,
            self.co2_mvr_data,
            self.co2_glb_data,
            self.co2_cyc_data,
        ) = gas_init(
            config.input_dir,
            config.ico2flg,
            config.ioznflg,
            config.ictmflg,
            iyear,
            imonth,
            gridlon,
            gridlat,
        )

        self._co2_cyc.view[:] = co2_cyc
        self._co2_arr.view[:] = co2_arr

        # Init sfc albedo and emissivity
        self.albedo = np.zeros((gridlon.view[:].shape[0], gridlon.view[:].shape[1], 4))
        self.sfcemis = np.zeros((gridlon.view[:].shape[0], gridlon.view[:].shape[1]))
        sfcemis_datafile = config.input_dir.joinpath("sfc_emissivity_idx.txt")
        self.iemslw, self._sfcemis_map = sfc_init(
            config.ialbflg,
            config.iemsflg,
            config.ldisable_radiation_quasi_sea_ice,
            sfcemis_datafile,
        )

        # Init clouds:
        self._llyr = cld_init(sigma, config.ivflip)

        self._cloud_optics_lw = rad.rrtmgp_cloud_optics.load_cloud_optics(
            cloud_optics_file=rad.data_types.CloudOpticsFiles.LW_BND
        )
        self._gas_optics_lw = rad.rrtmgp_gas_optics.load_gas_optics(
            gas_optics_file=rad.data_typesGasOpticsFiles.LW_G256
        )

        self._cloud_optics_sw = rad.rrtmgp_cloud_optics.load_cloud_optics(
            cloud_optics_file=rad.data_types.CloudOpticsFiles.SW_BND
        )
        self._gas_optics_sw = rad.rrtmgp_gas_optics.load_gas_optics(
            gas_optics_file=rad.data_types.GasOpticsFiles.SW_G224
        )
        self._gas_mapping = {
            "h2o": "qvapor",
            "o3": "qo3mr",
            "co": "co",
            "n2o": "n2o",
            "o2": "o2",
            "co2": "co2",
            "n2": "n2",
        }
        self._var_mapping = {
            "pres_layer": "prsl",
            "pres_level": "prsi",
            "temp_layer": "tlyr",
            "temp_level": "tlvl",
            "surface_temperature": "tsfc",
            "solar_zenith_angle": "solar_zenith_angle",
            "surface_albedo": "surface_albedo",
            "surface_albedo_direct": "surface_albedo_direct",
            "surface_albedo_diffuse": "surface_albedo_diffuse",
            "surface_emissivity": "surface_emissivity",
            "surface_emissivity_jacobian": "surface_emissivity_jacobian",
        }
        self._atm_map = rad.data_validation.AtmosphericMapping(
            dim_mapping=rad.config.DEFAULT_DIM_MAPPING,
            var_mapping=rad.config.var_mapping,
        )

        self._calc_tlvl = stencil_factory.from_origin_domain(
            func=calc_tlvl,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._coszmn = stencil_factory.from_origin_domain(
            func=coszmn,
            externals={
                "daily_mean": config.daily_mean,
                "fixed_sollat": config.fixed_sollat,
                "nstp": config.nstp,
            },
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        if config.icmphys == 4:
            self._progcld = stencil_factory.from_origin_domain(
                func=progcld4,
                externals={
                    "ivflip": config.ivflip,
                    "lcrick": config.lcrick,
                    "lcnorm": config.lcnorm,
                },
                origin=grid_indexing.origin_compute(),
                domain=grid_indexing.domain_compute(),
            )
        elif config.icmphys == 5:
            self._progcld = stencil_factory.from_origin_domain(
                func=progcld5,
                externals={
                    "gfs_cloud_overlap": config.gfs_cloud_overlap,
                    "ivflip": config.ivflip,
                    "lcrick": config.lcrick,
                    "lcnorm": config.lcnorm,
                },
                origin=grid_indexing.origin_compute(),
                domain=grid_indexing.domain_compute(),
            )
            raise NotImplementedError(
                f"radiation cloud microphysics control flag {config.icmphys} "
                "does not have cnvw or cnvc yet"
            )
        else:
            raise NotImplementedError(
                f"radiation cloud microphysics control flag {config.icmphys} "
                "not implemented, please choose 4 or 5"
            )
        self._coszmn = stencil_factory.from_origin_domain(
            func=coszmn,
            externals={
                "daily_mean": config.daily_mean,
                "fixed_sollat": config.fixed_sollat,
                "nstp": config.nstp,
                "sollat": config.sollat,
            },
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        if config.ictmflg == -2:
            if config.ivflip == 0:
                self._get_gases = stencil_factory.from_origin_domain(
                    func=get_gases_topdown,
                    externals={
                        "ico2flg": config.ico2flg,
                    },
                    origin=grid_indexing.origin_compute(),
                    domain=grid_indexing.domain_compute(),
                )
            else:
                self._get_gases = stencil_factory.from_origin_domain(
                    func=get_gases_bottomup,
                    externals={
                        "ico2flg": config.ico2flg,
                    },
                    origin=grid_indexing.origin_compute(),
                    domain=grid_indexing.domain_compute(),
                )

        self._calc_heating = stencil_factory.from_origin_domain(
            func=calc_heating,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        pass

    def _accumulate_radiation_inputs(
        self,
        state: RadiationState,
        sfc_state: SurfaceState,
        sdate
    ):
        """
        For RTE-RRTMGP we need level and layer profiles of temperature and pressure,
        the species used for the spectral calculations:
            humidity, cloud water (and size), cloud ice (and size),
            CO2, O3, N2O, N2, O2, CH4, CO
        albedo and surface emissivities, the solar zenith angle
        Here we extract that info from the model state and time,
        and make sure units are correct.
        """

        self._update_inputs_if_needed(state, sdate)

        self._calc_tlvl(
            state.prsl,
            state.prsi,
            state.tlyr,
            state.tsfc,
            state.qvapor,
            self._tvly,
            self._tsfca,
            state.tlvl,
        )

        self._coszmn(
            self.gridlon.view[:],
            self.gridlat.view[:],
            sdate[3],
            self.slag,
            self.sdec,
            self.cdec,
            self.anginc,
            state.mu0,
            self._coszdg,
            self._daymask,
        )

        if self.ictmflg == -2:
            self._get_gases(
                state.co2,
                state.prsl,
                self.co2_glb,
                self._co2_cyc,
                self._co2_arr,
            )

        self._progcld(
            state.prsl,
            state.prsi,
            state.tlyr,
            self._tvly,
            state.qliquid,
            self._cnvw,
            self._cnvc,
            sfc_state.islmsk,
            state.qcld,
            state.clwp,
            state.clwr,
            state.cip,
            state.cir,
        )

        set_albedo(
            self.ialbflg,
            sfc_state.islmsk,
            sfc_state.snowd,
            sfc_state.sncovr,
            sfc_state.snoalb,
            sfc_state.zorl,
            state.mu0,
            sfc_state.tskin,
            sfc_state.hprim,
            sfc_state.alvsf,
            sfc_state.alnsf,
            sfc_state.alvwf,
            sfc_state.alnwf,
            sfc_state.facsf,
            sfc_state.facwf,
            sfc_state.fice,
            sfc_state.tisfc,
            sfc_state.albedo,
            self.albedo,
            self.ldisable_radiation_quasi_sea_ice,
        )
        state.albedo.view[:] = self.albedo
        set_sfcemis(
            self.gridlon.view[:],
            self.gridlat.view[:],
            sfc_state.islmsk,
            sfc_state.snowd,
            sfc_state.sncovr,
            sfc_state.zorl,
            sfc_state.tskin,
            sfc_state.hprim,
            self.iemslw,
            self.ialbflg,
            self.ldisable_radiation_quasi_sea_ice,
            self.sfcemis,
            self._sfcemis_map,
            sfc_state.sfcemis,
        )
        state.sfc_emis.view[:] = self.sfcemis
        pass

    def _update_inputs_if_needed(self, state: RadiationState, sdate):
        """
        Updates input data from external sources when model date differs
        from the saved date
        """
        lsol_chg = False
        if (self.isolflg not in [0, 10]) and sdate[0] != self.saved_iyear:
            lsol_chg = True
        (
            self.slag,
            self.sdec,
            self.cdec,
            self.anginc,
            self.solcon,
            self.solc0,
            self.nstp,
            self.saved_iyear,
        ) = solar_update(
            sdate,
            self.solc0,
            self.deltsw,
            self.delt_rad,
            lsol_chg,
            self.saved_iyear,
            self.isolflg,
            self._solar_constants,
        )

        # Here is where we update ozone and aerosols when enabled
        if (sdate[1] != self.saved_imonth) or (self._first_step):
            update_co2 = True
            self.saved_imonth = sdate[1]

        self.co2_glb, self._co2_arr.view[:], self._co2_cyc.view[:], = co2_update(
            sdate[0],
            sdate[1],
            self.ico2flg,
            update_co2,
            self.ictmflg,
            self.co2_glb,
            self._co2_arr.view[:],
            self._co2_cyc.view[:],
            self.gridlon.view[:],
            self.gridlat.view[:],
            self.co2_glb_data,
            self.co2_mvr_data,
            self.co2_cyc_data,
        )

    def _prep_outputs(self):
        pass

    def step_radiation(self, state: RadiationState, sfc_state, date):
        self._accumulate_radiation_inputs(state, sfc_state, date)
        radx = state.to_rterrtmgp_xr()
        is_day = state.mu0.data[:] > 0.0

        # Do SW fluxes:
        sw_optics = self._gas_optics_sw.compute_gas_optics(
            radx,
            problem_type=rad.data_types.OpticsProblemTypes.TWO_STREAM,
            add_to_input=False,
            gas_name_map=self._gas_mapping,
            variable_mapping=self._atm_map,
        )
        sw_optics["surface_albedo"] = radx["albedo"]
        sw_optics["mu0"] = radx["mu0"]
        clr_fluxes_sw = rad.rte_solver.rte_solve(sw_optics, add_to_input=False)

        sw_cloud_optical_props = self._cloud_optics_sw.compute_cloud_optics(
            radx,
            problem_type=rad.data_types.OpticsProblemTypes.ABSORPTION,
            add_to_input=False,
            gas_name_map=self._gas_mapping,
            variable_mapping=self._atm_map,
        )
        sw_cloud_optical_props.add_to(sw_optics)
        fluxes_sw = rad.rte_solver.rte_solve(sw_optics, add_to_input=False)

        # And do LW fluxes
        lw_optics = self._gas_optics_lw.compute_gas_optics(
            radx,
            problem_type=rad.data_types.OpticsProblemTypes.ABSORPTION,
            add_to_input=False,
            gas_name_map=self._gas_mapping,
            variable_mapping=self._atm_map,
        )
        lw_optics["surface_emissivity"] = radx["sfc_emis"]
        clr_fluxes_lw = rad.rte_solver.rte_solve(lw_optics, add_to_input=False)
        lw_cloud_optical_props = self._cloud_optics_lw.compute_cloud_optics(
            radx,
            problem_type=rad.data_types.OpticsProblemTypes.ABSORPTION,
            add_to_input=False,
            gas_name_map=self._gas_mapping,
            variable_mapping=self._atm_map,
        )
        lw_cloud_optical_props.add_to(lw_optics)
        fluxes_lw = rad.rte_solver.rte_solve(lw_optics, add_to_input=False)

        state.flwd.view[:] = fluxes_lw.lw_flux_down.data.reshape(
            state.flwd.view[:].shape
        )
        state.flwu.view[:] = fluxes_lw.lw_flux_up.data.reshape(state.flwu.view[:].shape)
        state.fswd.view[:] = fluxes_sw.sw_flux_down.data.reshape(
            state.fswd.view[:].shape
        )
        state.fswu.view[:] = fluxes_sw.sw_flux_up.data.reshape(state.fswu.view[:].shape)
        self._calc_heating(
            state.fswu,
            state.fswd,
            state.prsi,
            state.hrtsw,
        )
        self._calc_heating(
            state.flwu,
            state.flwd,
            state.prsi,
            state.hrtlw,
        )

        if self._first_step:
            self._first_step = False
        pass
