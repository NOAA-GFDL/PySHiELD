from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.stencil import StencilFactory
from ndsl.initialization import SubtileGridSizer
from ndsl.initialization.allocator import QuantityFactory
from ndsl.namelist import Namelist
from ndsl.quantity import Quantity
from pyshield import PhysicsConfig
from pyshield.stencils.shield_microphysics.sedimentation import (
    Sedimentation,
    calc_terminal_velocity_ice,
    calc_terminal_velocity_rsg,
    init_zeros_heat_cap_latent_heat_precip,
    sedi_melt,
)
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class InitSed:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        config,
    ):
        self._idx = stencil_factory.grid_indexing

        self._init_zeros_heat_cap_latent_heat_precip = (
            stencil_factory.from_origin_domain(
                func=init_zeros_heat_cap_latent_heat_precip,
                externals={
                    "c1_vap": config.c1_vap,
                    "c1_liq": config.c1_liq,
                    "c1_ice": config.c1_ice,
                    "lv00": config.lv00,
                    "li00": config.li00,
                    "li20": config.li20,
                    "d1_vap": config.d1_vap,
                    "d1_ice": config.d1_ice,
                    "t_wfr": config.t_wfr,
                },
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )
        )

    def __call__(
        self,
        qvapor,
        qliquid,
        qrain,
        qice,
        qsnow,
        qgraupel,
        temperature,
        icpk,
        preflux_water,
        preflux_rain,
        preflux_ice,
        preflux_snow,
        preflux_graupel,
        vterminal_water,
        vterminal_rain,
        vterminal_ice,
        vterminal_snow,
        vterminal_graupel,
        column_water,
        column_rain,
        column_ice,
        column_snow,
        column_graupel,
    ):
        self._init_zeros_heat_cap_latent_heat_precip(
            qvapor,
            qliquid,
            qrain,
            qice,
            qsnow,
            qgraupel,
            temperature,
            icpk,
            preflux_water,
            preflux_rain,
            preflux_ice,
            preflux_snow,
            preflux_graupel,
            vterminal_water,
            vterminal_rain,
            vterminal_ice,
            vterminal_snow,
            vterminal_graupel,
            column_water,
            column_rain,
            column_ice,
            column_snow,
            column_graupel,
        )


class CalcVT:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        config,
    ):
        self._idx = stencil_factory.grid_indexing
        self.config = config
        if config.do_psd_ice_fall is False:
            self._calc_terminal_ice_velocity = stencil_factory.from_origin_domain(
                func=calc_terminal_velocity_ice,
                externals={
                    "ifflag": config.ifflag,
                    "constant_v": config.const_vi,
                    "v_fac": config.vi_fac,
                    "v_max": config.vi_max,
                    "aa": -4.14122e-5,
                    "bb": -0.00538922,
                    "cc": -0.0516344,
                    "dd": 0.00216078,
                    "ee": 1.9714,
                },
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )

        if False in [
            config.const_vr,
            config.const_vi,
            config.const_vs,
            config.const_vg,
        ]:
            self._calc_terminal_rsg_velocity = stencil_factory.from_origin_domain(
                func=calc_terminal_velocity_rsg,
                externals={"const_v": False},
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )

        if True in [config.const_vr, config.const_vi, config.const_vs, config.const_vg]:
            self._calc_terminal_rsg_velocity_const = stencil_factory.from_origin_domain(
                func=calc_terminal_velocity_rsg,
                externals={"const_v": True},
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )

    def __call__(
        self,
        qfall,
        density,
        temperature,
        vterminal,
        density_factor,
        mode,
    ):
        if mode == "ice":
            if self.config.do_psd_ice_fall:
                if self.config.const_vi is False:
                    self._calc_terminal_rsg_velocity(
                        qfall,
                        density,
                        density_factor,
                        vterminal,
                        self.config.vi_fac,
                        self.config.tvai,
                        self.config.tvbi,
                        self.config.mui,
                        self.config.blini,
                        self.config.vi_max,
                    )
                else:
                    self._calc_terminal_rsg_velocity_const(
                        qfall,
                        density,
                        density_factor,
                        vterminal,
                        self.config.vi_fac,
                        self.config.tvai,
                        self.config.tvbi,
                        self.config.mui,
                        self.config.blini,
                        self.config.vi_max,
                    )
            else:
                self._calc_terminal_ice_velocity(qfall, temperature, density, vterminal)
        elif mode == "snow":
            if self.config.const_vs is False:
                self._calc_terminal_rsg_velocity(
                    qfall,
                    density,
                    density_factor,
                    vterminal,
                    self.config.vs_fac,
                    self.config.tvas,
                    self.config.tvbs,
                    self.config.mus,
                    self.config.blins,
                    self.config.vs_max,
                )
            else:
                self._calc_terminal_rsg_velocity_const(
                    qfall,
                    density,
                    density_factor,
                    vterminal,
                    self.config.vs_fac,
                    self.config.tvas,
                    self.config.tvbs,
                    self.config.mus,
                    self.config.blins,
                    self.config.vs_max,
                )
        elif mode == "rain":
            if self.config.const_vr is False:
                self._calc_terminal_rsg_velocity(
                    qfall,
                    density,
                    density_factor,
                    vterminal,
                    self.config.vr_fac,
                    self.config.tvar,
                    self.config.tvbr,
                    self.config.mur,
                    self.config.blinr,
                    self.config.vr_max,
                )
            else:
                self._calc_terminal_rsg_velocity_const(
                    qfall,
                    density,
                    density_factor,
                    vterminal,
                    self.config.vr_fac,
                    self.config.tvar,
                    self.config.tvbr,
                    self.config.mur,
                    self.config.blinr,
                    self.config.vr_max,
                )
        else:
            raise ValueError(f"calc_vt mode {mode} not ice or snow")


class SediMelt:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        config,
        timestep,
    ):
        self._idx = stencil_factory.grid_indexing
        self.config = config

        self.li00 = config.li00
        self._timestep = timestep

        self._idx = stencil_factory.grid_indexing
        self._is_ = self._idx.isc
        self._ie = self._idx.iec
        self._js = self._idx.jsc
        self._je = self._idx.jec
        self._ks = 0
        self._ke = config.npz - 1
        self.c1_vap = config.c1_vap
        self.c1_liq = config.c1_liq
        self.c1_ice = config.c1_ice

    def __call__(
        self,
        qvapor,
        qliquid,
        qrain,
        qice,
        qsnow,
        qgraupel,
        cvm,
        temperature,
        delp,
        z_edge,
        z_terminal,
        z_surface,
        icpk,
        vterminal,
        column_rain,
        mode: str,
    ):
        if self.config.do_sedi_melt:
            if mode == "ice":
                sedi_melt(
                    qvapor,
                    qliquid,
                    qrain,
                    qice,
                    qsnow,
                    qgraupel,
                    cvm,
                    temperature,
                    delp,
                    z_edge,
                    z_terminal,
                    z_surface,
                    self._timestep,
                    vterminal,
                    column_rain,
                    self.config.tau_imlt,
                    icpk,
                    self.li00,
                    self.c1_vap,
                    self.c1_liq,
                    self.c1_ice,
                    self._ks,
                    self._ke,
                    self._is_,
                    self._ie,
                    self._js,
                    self._je,
                    mode,
                )
            elif mode == "snow":
                sedi_melt(
                    qvapor,
                    qliquid,
                    qrain,
                    qice,
                    qsnow,
                    qgraupel,
                    cvm,
                    temperature,
                    delp,
                    z_edge,
                    z_terminal,
                    z_surface,
                    self._timestep,
                    vterminal,
                    column_rain,
                    self.config.tau_smlt,
                    icpk,
                    self.li00,
                    self.c1_vap,
                    self.c1_liq,
                    self.c1_ice,
                    self._ks,
                    self._ke,
                    self._is_,
                    self._ie,
                    self._js,
                    self._je,
                    mode,
                )
            elif mode == "graupel":
                sedi_melt(
                    qvapor,
                    qliquid,
                    qrain,
                    qice,
                    qsnow,
                    qgraupel,
                    cvm,
                    temperature,
                    delp,
                    z_edge,
                    z_terminal,
                    z_surface,
                    self._timestep,
                    vterminal,
                    column_rain,
                    self.config.tau_gmlt,
                    icpk,
                    self.li00,
                    self.c1_vap,
                    self.c1_liq,
                    self.c1_ice,
                    self._ks,
                    self._ke,
                    self._is_,
                    self._ie,
                    self._js,
                    self._je,
                    mode,
                )
            else:
                raise ValueError(f"sedi_melt mode {mode} not ice, snow, or graupel")


class TranslateSedimentation(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "sd_qv", "mp3": True},
            "qliquid": {"serialname": "sd_ql", "mp3": True},
            "qrain": {"serialname": "sd_qr", "mp3": True},
            "qice": {"serialname": "sd_qi", "mp3": True},
            "qsnow": {"serialname": "sd_qs", "mp3": True},
            "qgraupel": {"serialname": "sd_qg", "mp3": True},
            "temperature": {"serialname": "sd_pt", "mp3": True},
            "delp": {"serialname": "sd_delp", "mp3": True},
            "delz": {"serialname": "sd_delz", "mp3": True},
            "density": {"serialname": "sd_den", "mp3": True},
            "density_factor": {"serialname": "sd_denfac", "mp3": True},
            "ua": {"serialname": "sd_u", "mp3": True},
            "va": {"serialname": "sd_v", "mp3": True},
            "wa": {"serialname": "sd_w", "mp3": True},
            "column_energy_change": {"serialname": "sd_dte", "mp3": True},
            "preflux_water": {"serialname": "sd_pfw", "mp3": True},
            "preflux_rain": {"serialname": "sd_pfr", "mp3": True},
            "preflux_ice": {"serialname": "sd_pfi", "mp3": True},
            "preflux_snow": {"serialname": "sd_pfs", "mp3": True},
            "preflux_graupel": {"serialname": "sd_pfg", "mp3": True},
            "vterminal_water": {"serialname": "sd_vtw", "mp3": True},
            "vterminal_rain": {"serialname": "sd_vtr", "mp3": True},
            "vterminal_ice": {"serialname": "sd_vti", "mp3": True},
            "vterminal_snow": {"serialname": "sd_vts", "mp3": True},
            "vterminal_graupel": {"serialname": "sd_vtg", "mp3": True},
            "column_water": {"serialname": "sd_w1", "mp3": True},
            "column_rain": {"serialname": "sd_r1", "mp3": True},
            "column_ice": {"serialname": "sd_i1", "mp3": True},
            "column_snow": {"serialname": "sd_s1", "mp3": True},
            "column_graupel": {"serialname": "sd_g1", "mp3": True},
        }

        self.in_vars["parameters"] = ["dt"]

        self.out_vars = {
            "qvapor": {"serialname": "sd_qv", "kend": namelist.npz, "mp3": True},
            "qliquid": {"serialname": "sd_ql", "kend": namelist.npz, "mp3": True},
            "qrain": {"serialname": "sd_qr", "kend": namelist.npz, "mp3": True},
            "qice": {"serialname": "sd_qi", "kend": namelist.npz, "mp3": True},
            "qsnow": {"serialname": "sd_qs", "kend": namelist.npz, "mp3": True},
            "qgraupel": {"serialname": "sd_qg", "kend": namelist.npz, "mp3": True},
            "temperature": {"serialname": "sd_pt", "kend": namelist.npz, "mp3": True},
            "ua": {"serialname": "sd_u", "kend": namelist.npz, "mp3": True},
            "va": {"serialname": "sd_v", "kend": namelist.npz, "mp3": True},
            "wa": {"serialname": "sd_w", "kend": namelist.npz, "mp3": True},
            "preflux_water": {
                "serialname": "sd_pfw",
                "kend": namelist.npz,
                "mp3": True,
            },
            "preflux_rain": {"serialname": "sd_pfr", "kend": namelist.npz, "mp3": True},
            "preflux_ice": {"serialname": "sd_pfi", "kend": namelist.npz, "mp3": True},
            "preflux_snow": {"serialname": "sd_pfs", "kend": namelist.npz, "mp3": True},
            "preflux_graupel": {
                "serialname": "sd_pfg",
                "kend": namelist.npz,
                "mp3": True,
            },
            "vterminal_water": {
                "serialname": "sd_vtw",
                "kend": namelist.npz,
                "mp3": True,
            },
            "vterminal_rain": {
                "serialname": "sd_vtr",
                "kend": namelist.npz,
                "mp3": True,
            },
            "vterminal_ice": {
                "serialname": "sd_vti",
                "kend": namelist.npz,
                "mp3": True,
            },
            "vterminal_snow": {
                "serialname": "sd_vts",
                "kend": namelist.npz,
                "mp3": True,
            },
            "vterminal_graupel": {
                "serialname": "sd_vtg",
                "kend": namelist.npz,
                "mp3": True,
            },
            "column_water": {"serialname": "sd_w1", "mp3": True},
            "column_rain": {"serialname": "sd_r1", "mp3": True},
            "column_ice": {"serialname": "sd_i1", "mp3": True},
            "column_snow": {"serialname": "sd_s1", "mp3": True},
            "column_graupel": {"serialname": "sd_g1", "mp3": True},
            "column_energy_change": {
                "serialname": "sd_dte",
                "kend": namelist.npz,
                "mp3": True,
            },
        }

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        pconf = PhysicsConfig.from_namelist(namelist)
        self.config = pconf.microphysics

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

        compute_func = Sedimentation(
            self.stencil_factory,
            self.quantity_factory,
            self.config,
            timestep=inputs.pop("dt"),
        )

        for var in inputs.keys():
            if len(inputs[var].shape) == 3:
                inputs[var] = Quantity(
                    inputs[var],
                    dims=[X_DIM, Y_DIM, Z_DIM],
                    units="unknown",
                )
            elif len(inputs[var].shape) == 2:
                inputs[var] = Quantity(
                    inputs[var],
                    dims=[X_DIM, Y_DIM],
                    units="unknown",
                )
            else:
                raise TypeError(
                    f"input data with strange len: {len(inputs[var].shape)}"
                )

        compute_func(**inputs)

        return self.slice_output(inputs)


class TranslateSediMelt(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "sm_qv", "kend": namelist.npz, "mp3": True},
            "qliquid": {"serialname": "sm_ql", "kend": namelist.npz, "mp3": True},
            "qrain": {"serialname": "sm_qr", "kend": namelist.npz, "mp3": True},
            "qice": {"serialname": "sm_qi", "kend": namelist.npz, "mp3": True},
            "qsnow": {"serialname": "sm_qs", "kend": namelist.npz, "mp3": True},
            "qgraupel": {"serialname": "sm_qg", "kend": namelist.npz, "mp3": True},
            "cvm": {"serialname": "sm_cv", "kend": namelist.npz, "mp3": True},
            "temperature": {"serialname": "sm_pt", "kend": namelist.npz, "mp3": True},
            "delp": {"serialname": "sm_dp", "kend": namelist.npz, "mp3": True},
            "z_edge": {"serialname": "sm_ze", "kend": namelist.npz + 1, "mp3": True},
            "z_terminal": {
                "serialname": "sm_zt",
                "kend": namelist.npz + 1,
                "mp3": True,
            },
            "z_surface": {"serialname": "sm_zs", "mp3": True},
            "icpk": {"serialname": "sm_ic", "kend": namelist.npz, "mp3": True},
            "vterminal": {"serialname": "sm_vt", "kend": namelist.npz, "mp3": True},
            "column_rain": {"serialname": "sm_r1", "mp3": True},
        }
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = {
            "qrain": {"serialname": "sm_qr", "kend": namelist.npz, "mp3": True},
            "qice": {"serialname": "sm_qi", "kend": namelist.npz, "mp3": True},
            "qsnow": {"serialname": "sm_qs", "kend": namelist.npz, "mp3": True},
            "qgraupel": {"serialname": "sm_qg", "kend": namelist.npz, "mp3": True},
            "temperature": {"serialname": "sm_pt", "kend": namelist.npz, "mp3": True},
            "cvm": {"serialname": "sm_cv", "kend": namelist.npz, "mp3": True},
            "column_rain": {"serialname": "sm_r1", "mp3": True},
        }

        self.stencil_factory = stencil_factory
        pconf = PhysicsConfig.from_namelist(namelist)
        self.config = pconf.microphysics

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        inputs["mode"] = "snow"

        compute_func = SediMelt(
            self.stencil_factory,
            self.config,
            timestep=inputs.pop("dt"),
        )

        compute_func(**inputs)

        return self.slice_output(inputs)


class TranslateCalcVTIce(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "qfall": {"serialname": "vti_qi", "kend": namelist.npz, "mp3": True},
            "density": {"serialname": "vti_den", "mp3": True},
            "density_factor": {"serialname": "vti_denfac", "mp3": True},
            "temperature": {"serialname": "vti_pt", "kend": namelist.npz, "mp3": True},
            "vterminal": {"serialname": "vti_vti", "kend": namelist.npz, "mp3": True},
        }

        self.out_vars = {
            "vterminal": {"serialname": "vti_vti", "kend": namelist.npz, "mp3": True},
        }

        self.stencil_factory = stencil_factory
        pconf = PhysicsConfig.from_namelist(namelist)
        self.config = pconf.microphysics

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        inputs["mode"] = "ice"

        compute_func = CalcVT(
            self.stencil_factory,
            self.config,
        )

        compute_func(**inputs)

        return self.slice_output(inputs)


class TranslateCalcVTSnow(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "qfall": {"serialname": "vts_qs", "kend": namelist.npz, "mp3": True},
            "density": {"serialname": "vts_den", "mp3": True},
            "density_factor": {"serialname": "vts_denfac", "mp3": True},
            "temperature": {"serialname": "vts_pt", "kend": namelist.npz, "mp3": True},
            "vterminal": {"serialname": "vts_vts", "kend": namelist.npz, "mp3": True},
        }

        self.out_vars = {
            "vterminal": {"serialname": "vts_vts", "kend": namelist.npz, "mp3": True},
        }

        self.stencil_factory = stencil_factory
        pconf = PhysicsConfig.from_namelist(namelist)
        self.config = pconf.microphysics

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        inputs["mode"] = "snow"

        compute_func = CalcVT(
            self.stencil_factory,
            self.config,
        )

        compute_func(**inputs)

        return self.slice_output(inputs)


class TranslateInitSed(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "is_qv", "kend": namelist.npz, "mp3": True},
            "qliquid": {"serialname": "is_ql", "kend": namelist.npz, "mp3": True},
            "qrain": {"serialname": "is_qr", "kend": namelist.npz, "mp3": True},
            "qice": {"serialname": "is_qi", "kend": namelist.npz, "mp3": True},
            "qsnow": {"serialname": "is_qs", "kend": namelist.npz, "mp3": True},
            "qgraupel": {"serialname": "is_qg", "kend": namelist.npz, "mp3": True},
            "temperature": {"serialname": "is_pt", "kend": namelist.npz, "mp3": True},
            "icpk": {"serialname": "is_icpk", "kend": namelist.npz, "mp3": True},
            "preflux_water": {
                "serialname": "is_pfw",
                "kend": namelist.npz,
                "mp3": True,
            },
            "preflux_rain": {"serialname": "is_pfr", "kend": namelist.npz, "mp3": True},
            "preflux_ice": {"serialname": "is_pfi", "kend": namelist.npz, "mp3": True},
            "preflux_snow": {"serialname": "is_pfs", "kend": namelist.npz, "mp3": True},
            "preflux_graupel": {
                "serialname": "is_pfg",
                "kend": namelist.npz,
                "mp3": True,
            },
            "vterminal_water": {
                "serialname": "is_vtw",
                "kend": namelist.npz,
                "mp3": True,
            },
            "vterminal_rain": {
                "serialname": "is_vtr",
                "kend": namelist.npz,
                "mp3": True,
            },
            "vterminal_ice": {
                "serialname": "is_vti",
                "kend": namelist.npz,
                "mp3": True,
            },
            "vterminal_snow": {
                "serialname": "is_vts",
                "kend": namelist.npz,
                "mp3": True,
            },
            "vterminal_graupel": {
                "serialname": "is_vtg",
                "kend": namelist.npz,
                "mp3": True,
            },
            "column_water": {"serialname": "is_w1", "mp3": True},
            "column_rain": {"serialname": "is_r1", "mp3": True},
            "column_ice": {"serialname": "is_i1", "mp3": True},
            "column_snow": {"serialname": "is_s1", "mp3": True},
            "column_graupel": {"serialname": "is_g1", "mp3": True},
        }

        self.out_vars = {
            "icpk": {"serialname": "is_icpk", "kend": namelist.npz, "mp3": True},
            "preflux_water": {
                "serialname": "is_pfw",
                "kend": namelist.npz,
                "mp3": True,
            },
            "preflux_rain": {"serialname": "is_pfr", "kend": namelist.npz, "mp3": True},
            "preflux_ice": {"serialname": "is_pfi", "kend": namelist.npz, "mp3": True},
            "preflux_snow": {"serialname": "is_pfs", "kend": namelist.npz, "mp3": True},
            "preflux_graupel": {
                "serialname": "is_pfg",
                "kend": namelist.npz,
                "mp3": True,
            },
            "vterminal_water": {
                "serialname": "is_vtw",
                "kend": namelist.npz,
                "mp3": True,
            },
            "vterminal_rain": {
                "serialname": "is_vtr",
                "kend": namelist.npz,
                "mp3": True,
            },
            "vterminal_ice": {
                "serialname": "is_vti",
                "kend": namelist.npz,
                "mp3": True,
            },
            "vterminal_snow": {
                "serialname": "is_vts",
                "kend": namelist.npz,
                "mp3": True,
            },
            "vterminal_graupel": {
                "serialname": "is_vtg",
                "kend": namelist.npz,
                "mp3": True,
            },
            "column_water": {"serialname": "is_w1", "mp3": True},
            "column_rain": {"serialname": "is_r1", "mp3": True},
            "column_ice": {"serialname": "is_i1", "mp3": True},
            "column_snow": {"serialname": "is_s1", "mp3": True},
            "column_graupel": {"serialname": "is_g1", "mp3": True},
        }

        self.stencil_factory = stencil_factory
        pconf = PhysicsConfig.from_namelist(namelist)
        self.config = pconf.microphysics

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        compute_func = InitSed(
            self.stencil_factory,
            self.config,
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
