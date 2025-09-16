from ndsl.dsl.stencil import StencilFactory
from ndsl.initialization.allocator import QuantityFactory
from ndsl.initialization.sizer import SubtileGridSizer
from ndsl.namelist import Namelist
from pyshield import PhysicsConfig
from pyshield.stencils.gfdl_cld_microphysics import GFDLCloudMicrophysics
from pyshield.stencils.gfdl_cld_microphysics.gfdl_cld_microphysics_state import (
    GFDLCloudMicrophysicsState,
)
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateMicrophysics3(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "mp_qv", "shield": True},
            "qliquid": {"serialname": "mp_ql", "shield": True},
            "qrain": {"serialname": "mp_qr", "shield": True},
            "qice": {"serialname": "mp_qi", "shield": True},
            "qsnow": {"serialname": "mp_qs", "shield": True},
            "qgraupel": {"serialname": "mp_qg", "shield": True},
            "qcld": {"serialname": "mp_qa", "shield": True},
            "qcloud_cond_nuclei": {"serialname": "mp_qnl", "shield": True},
            "qcloud_ice_nuclei": {"serialname": "mp_qni", "shield": True},
            "pt": {"serialname": "mp_pt", "shield": True},
            "ua": {"serialname": "mp_ua", "shield": True},
            "va": {"serialname": "mp_va", "shield": True},
            "wa": {"serialname": "mp_wa", "shield": True},
            "delz": {"serialname": "mp_delz", "shield": True},
            "delp": {"serialname": "mp_delp", "shield": True},
            "gsize": {"serialname": "mp_gsize", "shield": True},
            "geopotential_surface_height": {"serialname": "mp_hs", "shield": True},
            "column_water": {"serialname": "mp_water", "shield": True},
            "column_rain": {"serialname": "mp_rain", "shield": True},
            "column_ice": {"serialname": "mp_ice", "shield": True},
            "column_snow": {"serialname": "mp_snow", "shield": True},
            "column_graupel": {"serialname": "mp_graupel", "shield": True},
            "qcon": {"serialname": "mp_q_con", "shield": True},
            "cappa": {"serialname": "mp_cappa", "shield": True},
            "total_energy": {"serialname": "mp_te", "shield": True},
            "preflux_water": {"serialname": "mp_prefluxw", "shield": True},
            "preflux_rain": {"serialname": "mp_prefluxr", "shield": True},
            "preflux_ice": {"serialname": "mp_prefluxi", "shield": True},
            "preflux_snow": {"serialname": "mp_prefluxs", "shield": True},
            "preflux_graupel": {"serialname": "mp_prefluxg", "shield": True},
            "condensation": {"serialname": "mp_cond", "shield": True},
            "deposition": {"serialname": "mp_dep", "shield": True},
            "evaporation": {"serialname": "mp_reevap", "shield": True},
            "sublimation": {"serialname": "mp_sub", "shield": True},
        }
        self.in_vars["parameters"] = [
            "timestep",
            "consv_te",
            "last_step",
        ]

        self.out_vars = {
            "qvapor": {"serialname": "mp_qv", "kend": namelist.npz, "shield": True},
            "qliquid": {"serialname": "mp_ql", "kend": namelist.npz, "shield": True},
            "qrain": {"serialname": "mp_qr", "kend": namelist.npz, "shield": True},
            "qice": {"serialname": "mp_qi", "kend": namelist.npz, "shield": True},
            "qsnow": {"serialname": "mp_qs", "kend": namelist.npz, "shield": True},
            "qgraupel": {"serialname": "mp_qg", "kend": namelist.npz, "shield": True},
            "qcld": {"serialname": "mp_qa", "kend": namelist.npz, "shield": True},
            "pt": {"serialname": "mp_pt", "kend": namelist.npz, "shield": True},
            "ua": {"serialname": "mp_ua", "kend": namelist.npz, "shield": True},
            "va": {"serialname": "mp_va", "kend": namelist.npz, "shield": True},
            "wa": {"serialname": "mp_wa", "kend": namelist.npz, "shield": True},
            "delz": {"serialname": "mp_delz", "kend": namelist.npz, "shield": True},
            "delp": {"serialname": "mp_delp", "kend": namelist.npz, "shield": True},
            "column_water": {"serialname": "mp_water", "shield": True},
            "column_rain": {"serialname": "mp_rain", "shield": True},
            "column_ice": {"serialname": "mp_ice", "shield": True},
            "column_snow": {"serialname": "mp_snow", "shield": True},
            "column_graupel": {"serialname": "mp_graupel", "shield": True},
            "qcon": {"serialname": "mp_q_con", "kend": namelist.npz, "shield": True},
            "cappa": {"serialname": "mp_cappa", "kend": namelist.npz, "shield": True},
            "adj_vmr": {
                "serialname": "mp_adj_vmr",
                "kend": namelist.npz,
                "shield": True,
            },
            "total_energy": {
                "serialname": "mp_te",
                "kend": namelist.npz,
                "shield": True,
            },
            "column_energy_change": {"serialname": "mp_dte", "shield": True},
            "preflux_water": {
                "serialname": "mp_prefluxw",
                "kend": namelist.npz,
                "shield": True,
            },
            "preflux_rain": {
                "serialname": "mp_prefluxr",
                "kend": namelist.npz,
                "shield": True,
            },
            "preflux_ice": {
                "serialname": "mp_prefluxi",
                "kend": namelist.npz,
                "shield": True,
            },
            "preflux_snow": {
                "serialname": "mp_prefluxs",
                "kend": namelist.npz,
                "shield": True,
            },
            "preflux_graupel": {
                "serialname": "mp_prefluxg",
                "kend": namelist.npz,
                "shield": True,
            },
            "condensation": {"serialname": "mp_cond", "shield": True},
            "deposition": {"serialname": "mp_dep", "shield": True},
            "evaporation": {"serialname": "mp_reevap", "shield": True},
            "sublimation": {"serialname": "mp_sub", "shield": True},
            "particle_concentration_w": {
                "serialname": "mp_pcw",
                "kend": namelist.npz,
                "shield": True,
            },
            "effective_diameter_w": {
                "serialname": "mp_edw",
                "kend": namelist.npz,
                "shield": True,
            },
            "optical_extinction_w": {
                "serialname": "mp_oew",
                "kend": namelist.npz,
                "shield": True,
            },
            "radar_reflectivity_w": {
                "serialname": "mp_rrw",
                "kend": namelist.npz,
                "shield": True,
            },
            "terminal_velocity_w": {
                "serialname": "mp_tvw",
                "kend": namelist.npz,
                "shield": True,
            },
            "particle_concentration_r": {
                "serialname": "mp_pcr",
                "kend": namelist.npz,
                "shield": True,
            },
            "effective_diameter_r": {
                "serialname": "mp_edr",
                "kend": namelist.npz,
                "shield": True,
            },
            "optical_extinction_r": {
                "serialname": "mp_oer",
                "kend": namelist.npz,
                "shield": True,
            },
            "radar_reflectivity_r": {
                "serialname": "mp_rrr",
                "kend": namelist.npz,
                "shield": True,
            },
            "terminal_velocity_r": {
                "serialname": "mp_tvr",
                "kend": namelist.npz,
                "shield": True,
            },
            "particle_concentration_i": {
                "serialname": "mp_pci",
                "kend": namelist.npz,
                "shield": True,
            },
            "effective_diameter_i": {
                "serialname": "mp_edi",
                "kend": namelist.npz,
                "shield": True,
            },
            "optical_extinction_i": {
                "serialname": "mp_oei",
                "kend": namelist.npz,
                "shield": True,
            },
            "radar_reflectivity_i": {
                "serialname": "mp_rri",
                "kend": namelist.npz,
                "shield": True,
            },
            "terminal_velocity_i": {
                "serialname": "mp_tvi",
                "kend": namelist.npz,
                "shield": True,
            },
            "particle_concentration_s": {
                "serialname": "mp_pcs",
                "kend": namelist.npz,
                "shield": True,
            },
            "effective_diameter_s": {
                "serialname": "mp_eds",
                "kend": namelist.npz,
                "shield": True,
            },
            "optical_extinction_s": {
                "serialname": "mp_oes",
                "kend": namelist.npz,
                "shield": True,
            },
            "radar_reflectivity_s": {
                "serialname": "mp_rrs",
                "kend": namelist.npz,
                "shield": True,
            },
            "terminal_velocity_s": {
                "serialname": "mp_tvs",
                "kend": namelist.npz,
                "shield": True,
            },
            "particle_concentration_g": {
                "serialname": "mp_pcg",
                "kend": namelist.npz,
                "shield": True,
            },
            "effective_diameter_g": {
                "serialname": "mp_edg",
                "kend": namelist.npz,
                "shield": True,
            },
            "optical_extinction_g": {
                "serialname": "mp_oeg",
                "kend": namelist.npz,
                "shield": True,
            },
            "radar_reflectivity_g": {
                "serialname": "mp_rrg",
                "kend": namelist.npz,
                "shield": True,
            },
            "terminal_velocity_g": {
                "serialname": "mp_tvg",
                "kend": namelist.npz,
                "shield": True,
            },
        }

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        pconf = PhysicsConfig.from_namelist(namelist)
        self.config = pconf.microphysics

        self.sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npy - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        self.quantity_factory = QuantityFactory.from_backend(
            self.sizer, self.stencil_factory.backend
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        microphysics_state = GFDLCloudMicrophysicsState.init_from_storages(
            inputs,
            sizer=self.sizer,
            quantity_factory=self.quantity_factory,
        )

        microphysics = GFDLCloudMicrophysics(
            self.stencil_factory,
            self.quantity_factory,
            self.grid.grid_data,
            self.config,
            consv_te=inputs["consv_te"],
        )

        microphysics(
            microphysics_state,
            last_step=inputs["last_step"],
        )

        # copy microphysics state back to inputs
        inputs["qvapor"] = microphysics_state.qvapor
        inputs["qliquid"] = microphysics_state.qliquid
        inputs["qrain"] = microphysics_state.qrain
        inputs["qice"] = microphysics_state.qice
        inputs["qsnow"] = microphysics_state.qsnow
        inputs["qgraupel"] = microphysics_state.qgraupel
        inputs["qcld"] = microphysics_state.qcld
        inputs["pt"] = microphysics_state.pt
        inputs["ua"] = microphysics_state.ua
        inputs["va"] = microphysics_state.va
        inputs["wa"] = microphysics_state.wa
        inputs["delz"] = microphysics_state.delz
        inputs["delp"] = microphysics_state.delp
        inputs["column_water"][:] = microphysics_state.column_water
        inputs["column_rain"][:] = microphysics_state.column_rain
        inputs["column_ice"][:] = microphysics_state.column_ice
        inputs["column_snow"][:] = microphysics_state.column_snow
        inputs["column_graupel"] = microphysics_state.column_graupel
        inputs["condensation"][:] = microphysics_state.condensation
        inputs["deposition"][:] = microphysics_state.deposition
        inputs["sublimation"][:] = microphysics_state.sublimation
        inputs["evaporation"][:] = microphysics_state.evaporation
        inputs["column_energy_change"] = microphysics_state.column_energy_change
        inputs["adj_vmr"] = microphysics_state.adj_vmr
        inputs["particle_concentration_w"] = microphysics_state.particle_concentration_w
        inputs["effective_diameter_w"] = microphysics_state.effective_diameter_w
        inputs["optical_extinction_w"] = microphysics_state.optical_extinction_w
        inputs["radar_reflectivity_w"] = microphysics_state.radar_reflectivity_w
        inputs["terminal_velocity_w"] = microphysics_state.terminal_velocity_w
        inputs["particle_concentration_r"] = microphysics_state.particle_concentration_r
        inputs["effective_diameter_r"] = microphysics_state.effective_diameter_r
        inputs["optical_extinction_r"] = microphysics_state.optical_extinction_r
        inputs["radar_reflectivity_r"] = microphysics_state.radar_reflectivity_r
        inputs["terminal_velocity_r"] = microphysics_state.terminal_velocity_r
        inputs["particle_concentration_i"] = microphysics_state.particle_concentration_i
        inputs["effective_diameter_i"] = microphysics_state.effective_diameter_i
        inputs["optical_extinction_i"] = microphysics_state.optical_extinction_i
        inputs["radar_reflectivity_i"] = microphysics_state.radar_reflectivity_i
        inputs["terminal_velocity_i"] = microphysics_state.terminal_velocity_i
        inputs["particle_concentration_s"] = microphysics_state.particle_concentration_s
        inputs["effective_diameter_s"] = microphysics_state.effective_diameter_s
        inputs["optical_extinction_s"] = microphysics_state.optical_extinction_s
        inputs["radar_reflectivity_s"] = microphysics_state.radar_reflectivity_s
        inputs["terminal_velocity_s"] = microphysics_state.terminal_velocity_s
        inputs["particle_concentration_g"] = microphysics_state.particle_concentration_g
        inputs["effective_diameter_g"] = microphysics_state.effective_diameter_g
        inputs["optical_extinction_g"] = microphysics_state.optical_extinction_g
        inputs["radar_reflectivity_g"] = microphysics_state.radar_reflectivity_g
        inputs["terminal_velocity_g"] = microphysics_state.terminal_velocity_g

        return self.slice_output(inputs)
