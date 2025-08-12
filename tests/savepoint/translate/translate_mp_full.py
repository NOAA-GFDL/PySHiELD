from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.stencil import StencilFactory
from ndsl.initialization.sizer import SubtileGridSizer
from ndsl.initialization.allocator import QuantityFactory
from ndsl.namelist import Namelist
from pyshield import PhysicsConfig
from pyshield.stencils.shield_microphysics.ice_cloud import IceCloud
from pyshield.stencils.shield_microphysics.mp_full import (
    FullMicrophysics,
    accumulate_state_changes,
    add_fluxes_and_surface_tracers,
)
from pyshield.stencils.shield_microphysics.sedimentation import Sedimentation
from pyshield.stencils.shield_microphysics.subgrid_z_proc import (
    VerticalSubgridProcesses,
)
from pyshield.stencils.shield_microphysics.warm_rain import WarmRain
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class SubMicrophysics:
    def __init__(
        self,
        stencil_factory,
        quantity_factory,
        config,
        timestep: float,
        convert_mm_day: float,
    ):

        self._idx = stencil_factory.grid_indexing

        self._do_warm_rain = config.do_warm_rain_mp
        self._ntimes = config.ntimes

        def make_quantity():
            return quantity_factory.zeros([X_DIM, Y_DIM, Z_DIM], units="unknown")

        def make_quantity_2D():
            return quantity_factory.zeros([X_DIM, Y_DIM], units="unknown")

        self._fluxw = make_quantity()
        self._fluxr = make_quantity()
        self._fluxi = make_quantity()
        self._fluxs = make_quantity()
        self._fluxg = make_quantity()

        self._vtw = make_quantity()
        self._vtr = make_quantity()
        self._vti = make_quantity()
        self._vts = make_quantity()
        self._vtg = make_quantity()

        self._w1 = make_quantity_2D()
        self._r1 = make_quantity_2D()
        self._i1 = make_quantity_2D()
        self._s1 = make_quantity_2D()
        self._g1 = make_quantity_2D()

        self._cond = make_quantity_2D()
        self._dep = make_quantity_2D()
        self._reevap = make_quantity_2D()
        self._sub = make_quantity_2D()

        self._sedimentation = Sedimentation(
            stencil_factory,
            quantity_factory,
            config,
            timestep,
        )

        self._add_fluxes_and_surface_tracers = stencil_factory.from_origin_domain(
            func=add_fluxes_and_surface_tracers,
            externals={
                "convt": convert_mm_day,
            },
            origin=self._idx.origin_compute(),
            domain=self._idx.domain_compute(),
        )

        self._warm_rain = WarmRain(stencil_factory, config, timestep)

        if not self._do_warm_rain:
            self._ice_cloud = IceCloud(stencil_factory, config, timestep)

        self._subgrid_z_proc = VerticalSubgridProcesses(
            stencil_factory, config, timestep
        )

        self._accumulate_state_changes = stencil_factory.from_origin_domain(
            func=accumulate_state_changes,
            externals={
                "convt": convert_mm_day,
            },
            origin=self._idx.origin_compute(),
            domain=self._idx.domain_compute(),
        )

    def __call__(
        self,
        qvapor,
        qliquid,
        qrain,
        qice,
        qsnow,
        qgraupel,
        ua,
        va,
        wa,
        temperature,
        delp,
        delz,
        density,
        density_factor,
        cloud_condensation_nuclei,
        cloud_ice_nuclei,
        preflux_water,
        preflux_rain,
        preflux_ice,
        preflux_snow,
        preflux_graupel,
        h_var,
        rh_adj,
        column_energy_change,
        surface_water,
        surface_rain,
        surface_ice,
        surface_snow,
        surface_graupel,
        condensation,
        deposition,
        evaporation,
        sublimation,
    ):
        """
        Full Microphysics Loop
        executes ntimes:
        """
        for i in range(self._ntimes):
            # self._sedimentation(
            #     qvapor,
            #     qliquid,
            #     qrain,
            #     qice,
            #     qsnow,
            #     qgraupel,
            #     ua,
            #     va,
            #     wa,
            #     temperature,
            #     delp,
            #     delz,
            #     density,
            #     density_factor,
            #     self._fluxw,
            #     self._fluxr,
            #     self._fluxi,
            #     self._fluxs,
            #     self._fluxg,
            #     self._vtw,
            #     self._vtr,
            #     self._vti,
            #     self._vts,
            #     self._vtg,
            #     column_energy_change,
            #     self._w1,
            #     self._r1,
            #     self._i1,
            #     self._s1,
            #     self._g1,
            # )

            self._warm_rain(
                qvapor,
                qliquid,
                qrain,
                qice,
                qsnow,
                qgraupel,
                temperature,
                delp,
                density,
                density_factor,
                self._vtw,
                self._vtr,
                cloud_condensation_nuclei,
                self._reevap,
                h_var,
            )

            self._add_fluxes_and_surface_tracers(
                preflux_water,
                preflux_rain,
                preflux_ice,
                preflux_snow,
                preflux_graupel,
                self._fluxw,
                self._fluxr,
                self._fluxi,
                self._fluxs,
                self._fluxg,
                surface_water,
                surface_rain,
                surface_ice,
                surface_snow,
                surface_graupel,
                evaporation,
                self._w1,
                self._r1,
                self._i1,
                self._s1,
                self._g1,
                self._reevap,
            )

            if not self._do_warm_rain:
                self._ice_cloud(
                    qvapor,
                    qliquid,
                    qrain,
                    qice,
                    qsnow,
                    qgraupel,
                    temperature,
                    density,
                    density_factor,
                    self._vtw,
                    self._vtr,
                    self._vti,
                    self._vts,
                    self._vtg,
                    h_var,
                )

            # self._subgrid_z_proc(
            #     qvapor,
            #     qliquid,
            #     qrain,
            #     qice,
            #     qsnow,
            #     qgraupel,
            #     temperature,
            #     density,
            #     density_factor,
            #     delp,
            #     cloud_condensation_nuclei,
            #     cloud_ice_nuclei,
            #     self._cond,
            #     self._dep,
            #     self._reevap,
            #     self._sub,
            #     rh_adj,
            # )

            # self._accumulate_state_changes(
            #     self._cond,
            #     self._dep,
            #     self._reevap,
            #     self._sub,
            #     condensation,
            #     deposition,
            #     evaporation,
            #     sublimation,
            # )


class TranslateMPFull(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "mpf_qv", "mp3": True},
            "qliquid": {"serialname": "mpf_ql", "mp3": True},
            "qrain": {"serialname": "mpf_qr", "mp3": True},
            "qice": {"serialname": "mpf_qi", "mp3": True},
            "qsnow": {"serialname": "mpf_qs", "mp3": True},
            "qgraupel": {"serialname": "mpf_qg", "mp3": True},
            "ua": {"serialname": "mpf_u", "mp3": True},
            "va": {"serialname": "mpf_v", "mp3": True},
            "wa": {"serialname": "mpf_w", "mp3": True},
            "temperature": {"serialname": "mpf_pt", "mp3": True},
            "delp": {"serialname": "mpf_delp", "mp3": True},
            "delz": {"serialname": "mpf_delz", "mp3": True},
            "density": {"serialname": "mpf_den", "mp3": True},
            "density_factor": {"serialname": "mpf_denfac", "mp3": True},
            "cloud_condensation_nuclei": {"serialname": "mpf_ccn", "mp3": True},
            "cloud_ice_nuclei": {"serialname": "mpf_cin", "mp3": True},
            "preflux_water": {"serialname": "mpf_pfw", "mp3": True},
            "preflux_rain": {"serialname": "mpf_pfr", "mp3": True},
            "preflux_ice": {"serialname": "mpf_pfi", "mp3": True},
            "preflux_snow": {"serialname": "mpf_pfs", "mp3": True},
            "preflux_graupel": {"serialname": "mpf_pfg", "mp3": True},
            "h_var": {"serialname": "mpf_h_var", "mp3": True},
            "rh_adj": {"serialname": "mpf_rh_adj", "mp3": True},
            "column_energy_change": {"serialname": "mpf_dte", "mp3": True},
            "surface_water": {"serialname": "mpf_water", "mp3": True},
            "surface_rain": {"serialname": "mpf_rain", "mp3": True},
            "surface_ice": {"serialname": "mpf_ice", "mp3": True},
            "surface_snow": {"serialname": "mpf_snow", "mp3": True},
            "surface_graupel": {"serialname": "mpf_graupel", "mp3": True},
            "condensation": {"serialname": "mpf_cond", "mp3": True},
            "deposition": {"serialname": "mpf_dep", "mp3": True},
            "evaporation": {"serialname": "mpf_evap", "mp3": True},
            "sublimation": {"serialname": "mpf_sub", "mp3": True},
        }

        self.in_vars["parameters"] = ["convt", "dt"]

        self.out_vars = {
            "qvapor": {"serialname": "mpf_qv", "kend": namelist.npz, "mp3": True},
            "qliquid": {"serialname": "mpf_ql", "kend": namelist.npz, "mp3": True},
            "qrain": {"serialname": "mpf_qr", "kend": namelist.npz, "mp3": True},
            "qice": {"serialname": "mpf_qi", "kend": namelist.npz, "mp3": True},
            "qsnow": {"serialname": "mpf_qs", "kend": namelist.npz, "mp3": True},
            "qgraupel": {"serialname": "mpf_qg", "kend": namelist.npz, "mp3": True},
            "ua": {"serialname": "mpf_u", "kend": namelist.npz, "mp3": True},
            "va": {"serialname": "mpf_v", "kend": namelist.npz, "mp3": True},
            "wa": {"serialname": "mpf_w", "kend": namelist.npz, "mp3": True},
            "temperature": {"serialname": "mpf_pt", "kend": namelist.npz, "mp3": True},
            "delp": {"serialname": "mpf_delp", "kend": namelist.npz, "mp3": True},
            "delz": {"serialname": "mpf_delz", "kend": namelist.npz, "mp3": True},
            "density": {"serialname": "mpf_den", "kend": namelist.npz, "mp3": True},
            "density_factor": {
                "serialname": "mpf_denfac",
                "kend": namelist.npz,
                "mp3": True,
            },
            "cloud_condensation_nuclei": {
                "serialname": "mpf_ccn",
                "kend": namelist.npz,
                "mp3": True,
            },
            "cloud_ice_nuclei": {
                "serialname": "mpf_cin",
                "kend": namelist.npz,
                "mp3": True,
            },
            "preflux_water": {
                "serialname": "mpf_pfw",
                "kend": namelist.npz,
                "mp3": True,
            },
            "preflux_rain": {
                "serialname": "mpf_pfr",
                "kend": namelist.npz,
                "mp3": True,
            },
            "preflux_ice": {"serialname": "mpf_pfi", "kend": namelist.npz, "mp3": True},
            "preflux_snow": {
                "serialname": "mpf_pfs",
                "kend": namelist.npz,
                "mp3": True,
            },
            "preflux_graupel": {
                "serialname": "mpf_pfg",
                "kend": namelist.npz,
                "mp3": True,
            },
            "column_energy_change": {"serialname": "mpf_dte", "mp3": True},
            "surface_water": {"serialname": "mpf_water", "mp3": True},
            "surface_rain": {"serialname": "mpf_rain", "mp3": True},
            "surface_ice": {"serialname": "mpf_ice", "mp3": True},
            "surface_snow": {"serialname": "mpf_snow", "mp3": True},
            "surface_graupel": {"serialname": "mpf_graupel", "mp3": True},
            "condensation": {"serialname": "mpf_cond", "mp3": True},
            "deposition": {"serialname": "mpf_dep", "mp3": True},
            "evaporation": {"serialname": "mpf_evap", "mp3": True},
            "sublimation": {"serialname": "mpf_sub", "mp3": True},
        }

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        pconf = PhysicsConfig.from_namelist(namelist)
        self.config = pconf.microphysics
        self.config.do_mp_table_emulation = True

        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npy - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        self.quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        compute_func = FullMicrophysics(
            self.stencil_factory,
            self.quantity_factory,
            self.config,
            inputs.pop("dt"),
            inputs.pop("convt"),
        )

        compute_func(**inputs)

        return self.slice_output(inputs)


class TranslateMPSub(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "mpsub_qv", "mp3": True},
            "qliquid": {"serialname": "mpsub_ql", "mp3": True},
            "qrain": {"serialname": "mpsub_qr", "mp3": True},
            "qice": {"serialname": "mpsub_qi", "mp3": True},
            "qsnow": {"serialname": "mpsub_qs", "mp3": True},
            "qgraupel": {"serialname": "mpsub_qg", "mp3": True},
            "ua": {"serialname": "mpsub_u", "mp3": True},
            "va": {"serialname": "mpsub_v", "mp3": True},
            "wa": {"serialname": "mpsub_w", "mp3": True},
            "temperature": {"serialname": "mpsub_pt", "mp3": True},
            "delp": {"serialname": "mpsub_delp", "mp3": True},
            "delz": {"serialname": "mpsub_delz", "mp3": True},
            "density": {"serialname": "mpsub_den", "mp3": True},
            "density_factor": {"serialname": "mpsub_denfac", "mp3": True},
            "cloud_condensation_nuclei": {"serialname": "mpsub_ccn", "mp3": True},
            "cloud_ice_nuclei": {"serialname": "mpsub_cin", "mp3": True},
            "preflux_water": {"serialname": "mpsub_pfw", "mp3": True},
            "preflux_rain": {"serialname": "mpsub_pfr", "mp3": True},
            "preflux_ice": {"serialname": "mpsub_pfi", "mp3": True},
            "preflux_snow": {"serialname": "mpsub_pfs", "mp3": True},
            "preflux_graupel": {"serialname": "mpsub_pfg", "mp3": True},
            "h_var": {"serialname": "mpsub_h_var", "mp3": True},
            "rh_adj": {"serialname": "mpsub_rh_adj", "mp3": True},
            "column_energy_change": {"serialname": "mpsub_dte", "mp3": True},
            "surface_water": {"serialname": "mpsub_water", "mp3": True},
            "surface_rain": {"serialname": "mpsub_rain", "mp3": True},
            "surface_ice": {"serialname": "mpsub_ice", "mp3": True},
            "surface_snow": {"serialname": "mpsub_snow", "mp3": True},
            "surface_graupel": {"serialname": "mpsub_graupel", "mp3": True},
            "condensation": {"serialname": "mpsub_cond", "mp3": True},
            "deposition": {"serialname": "mpsub_dep", "mp3": True},
            "evaporation": {"serialname": "mpsub_evap", "mp3": True},
            "sublimation": {"serialname": "mpsub_sub", "mp3": True},
        }

        self.in_vars["parameters"] = ["convt", "dt"]

        self.out_vars = {
            "qvapor": {"serialname": "mpsub_qv", "kend": namelist.npz, "mp3": True},
            "qliquid": {"serialname": "mpsub_ql", "kend": namelist.npz, "mp3": True},
            "qrain": {"serialname": "mpsub_qr", "kend": namelist.npz, "mp3": True},
            "qice": {"serialname": "mpsub_qi", "kend": namelist.npz, "mp3": True},
            "qsnow": {"serialname": "mpsub_qs", "kend": namelist.npz, "mp3": True},
            "qgraupel": {"serialname": "mpsub_qg", "kend": namelist.npz, "mp3": True},
            "ua": {"serialname": "mpsub_u", "kend": namelist.npz, "mp3": True},
            "va": {"serialname": "mpsub_v", "kend": namelist.npz, "mp3": True},
            "wa": {"serialname": "mpsub_w", "kend": namelist.npz, "mp3": True},
            "temperature": {
                "serialname": "mpsub_pt",
                "kend": namelist.npz,
                "mp3": True,
            },
            "cloud_condensation_nuclei": {
                "serialname": "mpsub_ccn",
                "kend": namelist.npz,
                "mp3": True,
            },
            "cloud_ice_nuclei": {
                "serialname": "mpsub_cin",
                "kend": namelist.npz,
                "mp3": True,
            },
            "preflux_water": {
                "serialname": "mpsub_pfw",
                "kend": namelist.npz,
                "mp3": True,
            },
            "preflux_rain": {
                "serialname": "mpsub_pfr",
                "kend": namelist.npz,
                "mp3": True,
            },
            "preflux_ice": {
                "serialname": "mpsub_pfi",
                "kend": namelist.npz,
                "mp3": True,
            },
            "preflux_snow": {
                "serialname": "mpsub_pfs",
                "kend": namelist.npz,
                "mp3": True,
            },
            "preflux_graupel": {
                "serialname": "mpsub_pfg",
                "kend": namelist.npz,
                "mp3": True,
            },
            "column_energy_change": {"serialname": "mpsub_dte", "mp3": True},
            "surface_water": {"serialname": "mpsub_water", "mp3": True},
            "surface_rain": {"serialname": "mpsub_rain", "mp3": True},
            "surface_ice": {"serialname": "mpsub_ice", "mp3": True},
            "surface_snow": {"serialname": "mpsub_snow", "mp3": True},
            "surface_graupel": {"serialname": "mpsub_graupel", "mp3": True},
            "condensation": {"serialname": "mpsub_cond", "mp3": True},
            "deposition": {"serialname": "mpsub_dep", "mp3": True},
            "evaporation": {"serialname": "mpsub_evap", "mp3": True},
            "sublimation": {"serialname": "mpsub_sub", "mp3": True},
        }

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        pconf = PhysicsConfig.from_namelist(namelist)
        self.config = pconf.microphysics
        self.config.do_mp_table_emulation = True

        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npy - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        self.quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        compute_func = SubMicrophysics(
            self.stencil_factory,
            self.quantity_factory,
            self.config,
            inputs.pop("dt"),
            inputs.pop("convt"),
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
