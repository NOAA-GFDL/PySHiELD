import dataclasses
from pathlib import Path

import numpy as np
import pyrte_rrtmgp as rad
import xarray as xr
from pyrte_rrtmgp import rrtmgp_cloud_optics, rrtmgp_gas_optics
from pyrte_rrtmgp.data_types import CloudOpticsFiles, GasOpticsFiles, OpticsProblemTypes

from ndsl import Float, Int, Quantity, QuantityFactory, StencilFactory

from .rad_astro import coszmn, sol_init, solar_update
from .rad_clouds import progcld4, progcld5
from .radiation_state import RadiationState


@dataclasses.dataclass
class RadiationConfig:
    dt_atmos: Float
    fhswr: Float
    fhlwr: Float
    isolar: Int
    icmphys: Int
    solar_constant_file: Path
    aerosol_file: Path
    daily_mean: bool
    fixed_sollat: bool
    nstp: Int
    ivflip: Int
    lcnorm: bool
    lcrick: bool
    gfs_cloud_overlap: bool


class RTE_RRTMGPDriver:
    def __init__(
        self,
        config: RadiationConfig,
        year: Int,
        stencil_factory: StencilFactory,
    ):
        grid_indexing = stencil_factory.grid_indexing
        self._isolar = config.isolar
        self._isolflg, self._solar_constants, self.solc0 = sol_init(
            self._isolar,
            config.solar_constant_file,
            year,
        )
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
        else:
            raise NotImplementedError(
                f"radiation cloud microphysics control flag {config.icmphys} "
                "not implemented, please choose 4 or 5"
            )
        pass

    def _accumulate_radiation_inputs(self, state: RadiationState):
        """
        For RTE-RRTMGP we need level and layer profiles of temperature and pressure,
        the species used for the spectral calculations:
            humidity, cloud water (and size), cloud ice (and size),
            CO2, O3, N2O, N2, O2, CH4, CO
        albedo and surface emissivities, the solar zenith angle
        Here we extract that info from the model state and time,
        and make sure units are correct.
        """
        pass

    def _get_ozone(self):
        """
        sets layer ozone mass-mixing ratio
        """
        pass

    def _get_gases(self):
        """
        Sets up non-prognostic gas volume mixing
        """
        pass

    def _set_aerosols(self):
        """
        set aerosol profiles
        """
        pass

    def _set_albedo(self):
        """
        sets surface albedos for SW calculations
        """
        pass

    def _set_sfcemis(self):
        """
        sets surface emissivities for LW calculations
        """
        pass

    def _prep_outputs(self):
        pass

    def step_radiation(self, state: RadiationState):
        self._accumulate_radiation_inputs(state)
        radx = state.to_rterrtmgp_xr()
        is_day = state.mu0.data[:] > 0.0

        # Do SW fluxes:
        sw_optics = self._gas_optics_sw.compute_gas_optics(
            radx,
            problem_type=OpticsProblemTypes.TWO_STREAM,
            add_to_input=False,
            gas_name_map=self._gas_mapping,
            variable_mapping=self._atm_map,
        )
        sw_optics["surface_albedo"] = radx["albedo"]
        sw_optics["mu0"] = radx["mu0"]
        clr_fluxes_sw = rad.rte_solver.rte_solve(sw_optics, add_to_input=False)

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

        pass
