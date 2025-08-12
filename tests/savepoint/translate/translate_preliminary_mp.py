import pyfv3.stencils.basic_operations as basic

from ndsl.constants import X_DIM, Y_DIM
from ndsl.dsl.stencil import StencilFactory
from ndsl.initialization import SubtileGridSizer
from ndsl.initialization.allocator import QuantityFactory
from ndsl.namelist import Namelist
from pyshield import PhysicsConfig
from pyshield.stencils.shield_microphysics.gfdl_cld_mp_driver import (
    calculate_density_factor,
    cloud_nuclei_subgrid_and_relative_humidity,
    convert_specific_to_mass_mixing_ratios_and_calculate_densities,
    convert_virtual_to_true_temperature_and_calc_total_energy,
    moist_total_energy_and_water,
    reset_initial_values_and_make_copies,
)
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class PrelimCalcs:
    def __init__(
        self,
        stencil_factory,
        quantity_factory,
        config,
        consv_te,
    ):
        self.config = config
        self._idx = stencil_factory.grid_indexing
        self.consv_te = consv_te
        self.do_qa = config.do_qa
        self.hydrostatic = config.hydrostatic
        self.consv_checker = config.consv_checker
        self.do_inline_mp = config.do_inline_mp
        self.fix_negative = config.fix_negative
        self._te_err = config.te_err
        self._tw_err = config.tw_err

        def make_quantity2d(**kwargs):
            return quantity_factory.zeros(dims=[X_DIM, Y_DIM], units="unknown")

        self._tot_energy_change = make_quantity2d()
        self._bottom_density = make_quantity2d()

        self._copy_stencil = stencil_factory.from_origin_domain(
            basic.copy_defn,
            origin=self._idx.origin_compute(),
            domain=self._idx.domain_compute(),
        )

        self._reset_initial_values_and_make_copies = stencil_factory.from_origin_domain(
            func=reset_initial_values_and_make_copies,
            origin=self._idx.origin_compute(),
            domain=self._idx.domain_compute(),
        )

        self._convert_virtual_to_true_temperature_and_calc_total_energy = (
            stencil_factory.from_origin_domain(
                func=convert_virtual_to_true_temperature_and_calc_total_energy,
                externals={
                    "consv_te": consv_te,
                    "do_inline_mp": self.do_inline_mp,
                    "hydrostatic": self.config.hydrostatic,
                    "c_air": self.config.c_air,
                    "c1_vap": self.config.c1_vap,
                    "c1_liq": self.config.c1_liq,
                    "c1_ice": self.config.c1_ice,
                },
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )
        )

        if self.config.consv_checker:
            self._moist_total_energy_and_water_mq = stencil_factory.from_origin_domain(
                func=moist_total_energy_and_water,
                externals={
                    "hydrostatic": self.config.hydrostatic,
                    "moist_q": True,
                    "c1_vap": self.config.c1_vap,
                    "c1_liq": self.config.c1_liq,
                    "c1_ice": self.config.c1_ice,
                    "lv00": self.config.lv00,
                    "li00": self.config.li00,
                    "c_air": self.config.c_air,
                    "timestep": self.config.dt_full,
                },
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )

            self._moist_total_energy_and_water = stencil_factory.from_origin_domain(
                func=moist_total_energy_and_water,
                externals={
                    "hydrostatic": self.config.hydrostatic,
                    "moist_q": False,
                    "c1_vap": self.config.c1_vap,
                    "c1_liq": self.config.c1_liq,
                    "c1_ice": self.config.c1_ice,
                    "lv00": self.config.lv00,
                    "li00": self.config.li00,
                    "c_air": self.config.c_air,
                    "timestep": self.config.dt_full,
                },
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )

        self._convert_specific_to_mass_mixing_ratios_and_calculate_densities = (
            stencil_factory.from_origin_domain(
                func=convert_specific_to_mass_mixing_ratios_and_calculate_densities,
                externals={
                    "do_inline_mp": self.config.do_inline_mp,
                },
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )
        )

        self._calculate_density_factor = stencil_factory.from_origin_domain(
            func=calculate_density_factor,
            origin=self._idx.origin_compute(),
            domain=self._idx.domain_compute(),
        )

        self._cloud_nuclei_subgrid_and_relative_humidity = (
            stencil_factory.from_origin_domain(
                func=cloud_nuclei_subgrid_and_relative_humidity,
                externals={
                    "prog_ccn": self.config.prog_ccn,
                    "ccn_l": self.config.ccn_l,
                    "ccn_o": self.config.ccn_o,
                    "dw_land": self.config.dw_land,
                    "dw_ocean": self.config.dw_ocean,
                    "rh_inc": self.config.rh_inc,
                    "rh_inr": self.config.rh_inr,
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
        delp,
        delz,
        density,
        pz,
        density_factor,
        pt,
        ua,
        va,
        wa,
        qvapor0,
        qliquid0,
        qrain0,
        qice0,
        qsnow0,
        qgraupel0,
        dp0,
        pt0,
        u0,
        v0,
        w0,
        column_energy_change,
        cond,
        adj_vmr,
        gsize,
        hs,
        qcloud_cond_nuclei,
        qcloud_ice_nuclei,
        column_vapor,
        column_water,
        column_rain,
        column_ice,
        column_snow,
        column_graupel,
        total_energy_wet_begin,
        total_water_wet_begin,
        total_energy_bot_wet_begin,
        total_water_bot_wet_begin,
        total_energy_dry_begin,
        total_water_dry_begin,
        total_energy_bot_dry_begin,
        total_water_bot_dry_begin,
        cloud_condensation_nuclei,
        cloud_ice_nuclei,
        h_var,
        rh_adj,
        rh_rain,
    ):
        self._reset_initial_values_and_make_copies(
            adj_vmr,
            qvapor,
            qliquid,
            qrain,
            qice,
            qsnow,
            qgraupel,
            delp,
            pt,
            ua,
            va,
            qvapor0,
            qliquid0,
            qrain0,
            qice0,
            qsnow0,
            qgraupel0,
            dp0,
            pt0,
            u0,
            v0,
            column_energy_change,
            cond,
        )

        if not self.hydrostatic:
            self._copy_stencil(wa, w0)

        if self.do_inline_mp or self.consv_te:
            self._convert_virtual_to_true_temperature_and_calc_total_energy(
                qvapor,
                qliquid,
                qrain,
                qice,
                qsnow,
                qgraupel,
                pt,
            )

        if self.consv_checker:
            self._moist_total_energy_and_water_mq(
                qvapor,
                qliquid,
                qrain,
                qice,
                qsnow,
                qgraupel,
                pt,
                ua,
                va,
                wa,
                delp,
                gsize,
                column_energy_change,
                column_vapor,
                column_water,
                column_rain,
                column_ice,
                column_snow,
                column_graupel,
                0.0,
                0.0,
                total_energy_wet_begin,
                total_water_wet_begin,
                total_energy_bot_wet_begin,
                total_water_bot_wet_begin,
            )

        self._convert_specific_to_mass_mixing_ratios_and_calculate_densities(
            qvapor,
            qliquid,
            qrain,
            qice,
            qsnow,
            qgraupel,
            delp,
            delz,
            pt,
            density,
            pz,
            self._bottom_density,
        )

        self._calculate_density_factor(
            density,
            density_factor,
            self._bottom_density,
        )

        if self.consv_checker:
            self._moist_total_energy_and_water(
                qvapor,
                qliquid,
                qrain,
                qice,
                qsnow,
                qgraupel,
                pt,
                ua,
                va,
                wa,
                delp,
                gsize,
                column_energy_change,
                column_vapor,
                column_water,
                column_rain,
                column_ice,
                column_snow,
                column_graupel,
                0.0,
                0.0,
                total_energy_dry_begin,
                total_water_dry_begin,
                total_energy_bot_dry_begin,
                total_water_bot_dry_begin,
            )

        self._cloud_nuclei_subgrid_and_relative_humidity(
            hs,
            qcloud_cond_nuclei,
            qcloud_ice_nuclei,
            density,
            cloud_condensation_nuclei,
            cloud_ice_nuclei,
            gsize,
            h_var,
            rh_adj,
            rh_rain,
        )


class TranslatePreliminaryCalculations(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "pre_qv", "mp3": True},
            "qliquid": {"serialname": "pre_ql", "mp3": True},
            "qrain": {"serialname": "pre_qr", "mp3": True},
            "qice": {"serialname": "pre_qi", "mp3": True},
            "qsnow": {"serialname": "pre_qs", "mp3": True},
            "qgraupel": {"serialname": "pre_qg", "mp3": True},
            "density": {"serialname": "pre_den", "mp3": True},
            "delp": {"serialname": "pre_delp", "mp3": True},
            "delz": {"serialname": "pre_delz", "mp3": True},
            "pz": {"serialname": "pre_pz", "mp3": True},
            "density_factor": {"serialname": "pre_denfac", "mp3": True},
            "pt": {"serialname": "pre_pt", "mp3": True},
            "ua": {"serialname": "pre_ua", "mp3": True},
            "va": {"serialname": "pre_va", "mp3": True},
            "wa": {"serialname": "pre_wa", "mp3": True},
            "qvapor0": {"serialname": "pre_qv0", "mp3": True},
            "qliquid0": {"serialname": "pre_ql0", "mp3": True},
            "qrain0": {"serialname": "pre_qr0", "mp3": True},
            "qice0": {"serialname": "pre_qi0", "mp3": True},
            "qsnow0": {"serialname": "pre_qs0", "mp3": True},
            "qgraupel0": {"serialname": "pre_qg0", "mp3": True},
            "dp0": {"serialname": "pre_dp0", "mp3": True},
            "pt0": {"serialname": "pre_pt0", "mp3": True},
            "u0": {"serialname": "pre_u0", "mp3": True},
            "v0": {"serialname": "pre_v0", "mp3": True},
            "w0": {"serialname": "pre_w0", "mp3": True},
            "column_energy_change": {"serialname": "pre_dte", "mp3": True},
            "cond": {"serialname": "pre_cond", "mp3": True},
            "adj_vmr": {"serialname": "pre_adj_vmr", "mp3": True},
            "gsize": {"serialname": "pre_gsize", "mp3": True},
            "hs": {"serialname": "pre_hs", "mp3": True},
            "qcloud_cond_nuclei": {"serialname": "pre_qnl", "mp3": True},
            "qcloud_ice_nuclei": {"serialname": "pre_qni", "mp3": True},
            "column_vapor": {"serialname": "pre_vapor", "mp3": True},
            "column_water": {"serialname": "pre_water", "mp3": True},
            "column_rain": {"serialname": "pre_rain", "mp3": True},
            "column_ice": {"serialname": "pre_ice", "mp3": True},
            "column_snow": {"serialname": "pre_snow", "mp3": True},
            "column_graupel": {"serialname": "pre_graupel", "mp3": True},
            "total_energy_wet_begin": {"serialname": "pre_ew0", "mp3": True},
            "total_water_wet_begin": {"serialname": "pre_ww0", "mp3": True},
            "total_energy_bot_wet_begin": {"serialname": "pre_bew0", "mp3": True},
            "total_water_bot_wet_begin": {"serialname": "pre_bww0", "mp3": True},
            "total_energy_dry_begin": {"serialname": "pre_ed0", "mp3": True},
            "total_water_dry_begin": {"serialname": "pre_wd0", "mp3": True},
            "total_energy_bot_dry_begin": {"serialname": "pre_bed0", "mp3": True},
            "total_water_bot_dry_begin": {"serialname": "pre_bwd0", "mp3": True},
            "cloud_condensation_nuclei": {"serialname": "pre_ccn", "mp3": True},
            "cloud_ice_nuclei": {"serialname": "pre_cin", "mp3": True},
            "h_var": {"serialname": "pre_h_var", "mp3": True},
            "rh_adj": {"serialname": "pre_rh_adj", "mp3": True},
            "rh_rain": {"serialname": "pre_rh_rain", "mp3": True},
        }

        self.out_vars = {
            "qvapor": {"serialname": "pre_qv", "kend": namelist.npz, "mp3": True},
            "qliquid": {"serialname": "pre_ql", "kend": namelist.npz, "mp3": True},
            "qrain": {"serialname": "pre_qr", "kend": namelist.npz, "mp3": True},
            "qice": {"serialname": "pre_qi", "kend": namelist.npz, "mp3": True},
            "qsnow": {"serialname": "pre_qs", "kend": namelist.npz, "mp3": True},
            "qgraupel": {"serialname": "pre_qg", "kend": namelist.npz, "mp3": True},
            "density": {"serialname": "pre_den", "kend": namelist.npz, "mp3": True},
            "delp": {"serialname": "pre_delp", "kend": namelist.npz, "mp3": True},
            "delz": {"serialname": "pre_delz", "kend": namelist.npz, "mp3": True},
            "pz": {"serialname": "pre_pz", "kend": namelist.npz, "mp3": True},
            "density_factor": {
                "serialname": "pre_denfac",
                "kend": namelist.npz,
                "mp3": True,
            },
            "pt": {"serialname": "pre_pt", "kend": namelist.npz, "mp3": True},
            "ua": {"serialname": "pre_ua", "kend": namelist.npz, "mp3": True},
            "va": {"serialname": "pre_va", "kend": namelist.npz, "mp3": True},
            "wa": {"serialname": "pre_wa", "kend": namelist.npz, "mp3": True},
            "qvapor0": {"serialname": "pre_qv0", "kend": namelist.npz, "mp3": True},
            "qliquid0": {"serialname": "pre_ql0", "kend": namelist.npz, "mp3": True},
            "qrain0": {"serialname": "pre_qr0", "kend": namelist.npz, "mp3": True},
            "qice0": {"serialname": "pre_qi0", "kend": namelist.npz, "mp3": True},
            "qsnow0": {"serialname": "pre_qs0", "kend": namelist.npz, "mp3": True},
            "qgraupel0": {"serialname": "pre_qg0", "kend": namelist.npz, "mp3": True},
            "dp0": {"serialname": "pre_dp0", "kend": namelist.npz, "mp3": True},
            "pt0": {"serialname": "pre_pt0", "kend": namelist.npz, "mp3": True},
            "u0": {"serialname": "pre_u0", "kend": namelist.npz, "mp3": True},
            "v0": {"serialname": "pre_v0", "kend": namelist.npz, "mp3": True},
            "w0": {"serialname": "pre_w0", "kend": namelist.npz, "mp3": True},
            "column_energy_change": {"serialname": "pre_dte", "mp3": True},
            "cond": {"serialname": "pre_cond", "mp3": True},
            "adj_vmr": {"serialname": "pre_adj_vmr", "kend": namelist.npz, "mp3": True},
            "total_energy_wet_begin": {
                "serialname": "pre_ew0",
                "kend": namelist.npz,
                "mp3": True,
            },
            "total_water_wet_begin": {
                "serialname": "pre_ww0",
                "kend": namelist.npz,
                "mp3": True,
            },
            "total_energy_bot_wet_begin": {"serialname": "pre_bew0", "mp3": True},
            "total_water_bot_wet_begin": {"serialname": "pre_bww0", "mp3": True},
            "total_energy_dry_begin": {
                "serialname": "pre_ed0",
                "kend": namelist.npz,
                "mp3": True,
            },
            "total_water_dry_begin": {
                "serialname": "pre_wd0",
                "kend": namelist.npz,
                "mp3": True,
            },
            "total_energy_bot_dry_begin": {"serialname": "pre_bed0", "mp3": True},
            "total_water_bot_dry_begin": {"serialname": "pre_bwd0", "mp3": True},
            "cloud_condensation_nuclei": {
                "serialname": "pre_ccn",
                "kend": namelist.npz,
                "mp3": True,
            },
            "cloud_ice_nuclei": {
                "serialname": "pre_cin",
                "kend": namelist.npz,
                "mp3": True,
            },
            "h_var": {"serialname": "pre_h_var", "mp3": True},
            "rh_adj": {"serialname": "pre_rh_adj", "mp3": True},
            "rh_rain": {"serialname": "pre_rh_rain", "mp3": True},
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

        compute_func = PrelimCalcs(
            self.stencil_factory, self.quantity_factory, self.config, consv_te=False
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
