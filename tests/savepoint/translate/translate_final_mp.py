from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.stencil import StencilFactory
from ndsl.initialization import SubtileGridSizer
from ndsl.initialization.allocator import QuantityFactory
from ndsl.namelist import Namelist
from pySHiELD import PhysicsConfig
from pySHiELD.stencils.SHiELD_microphysics.cloud_fraction import CloudFraction
from pySHiELD.stencils.SHiELD_microphysics.gfdl_cld_mp_driver import (
    calc_sedimentation_energy_loss,
    calculate_particle_properties,
    calculate_total_energy_change_and_convert_temp,
    convert_mass_mixing_to_specific_ratios_and_update_temperatures,
    moist_total_energy_and_water,
    total_energy_check,
    update_temperature_pre_delp_q,
)
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class PostMP:
    def __init__(
        self,
        stencil_factory,
        config,
        quantity_factory,
        consv_te,
    ):
        self._idx = stencil_factory.grid_indexing
        self.consv_te = consv_te
        self.do_qa = config.do_qa
        self.hydrostatic = config.hydrostatic
        self.consv_checker = config.consv_checker
        self.do_inline_mp = config.do_inline_mp
        self.fix_negative = config.fix_negative
        self._te_err = config.te_err
        self._tw_err = config.tw_err
        self.do_sedi_uv = config.do_sedi_uv
        self.do_sedi_w = config.do_sedi_w

        self._tzuv = quantity_factory.zeros(dims=[X_DIM, Y_DIM, Z_DIM], units="unknown")
        self._tzw = quantity_factory.zeros(dims=[X_DIM, Y_DIM, Z_DIM], units="unknown")

        if config.do_hail:
            pcag = config.pcah
            pcbg = config.pcbh
            edag = config.edah
            edbg = config.edbh
            oeag = config.oeah
            oebg = config.oebh
            rrag = config.rrah
            rrbg = config.rrbh
            tvag = config.tvah
            tvbg = config.tvbh
            mug = config.muh
            bling = config.blinh
        else:
            pcag = config.pcag
            pcbg = config.pcbg
            edag = config.edag
            edbg = config.edbg
            oeag = config.oeag
            oebg = config.oebg
            rrag = config.rrag
            rrbg = config.rrbg
            tvag = config.tvag
            tvbg = config.tvbg
            mug = config.mug
            bling = config.bling

        self._calculate_particle_properties = stencil_factory.from_origin_domain(
            func=calculate_particle_properties,
            externals={
                "pcaw": config.pcaw,
                "pcbw": config.pcbw,
                "pcai": config.pcai,
                "pcbi": config.pcbi,
                "pcar": config.pcar,
                "pcbr": config.pcbr,
                "pcas": config.pcas,
                "pcbs": config.pcbs,
                "pcag": pcag,
                "pcbg": pcbg,
                "edaw": config.edaw,
                "edbw": config.edbw,
                "edai": config.edai,
                "edbi": config.edbi,
                "edar": config.edar,
                "edbr": config.edbr,
                "edas": config.edas,
                "edbs": config.edbs,
                "edag": edag,
                "edbg": edbg,
                "oeaw": config.oeaw,
                "oebw": config.oebw,
                "oeai": config.oeai,
                "oebi": config.oebi,
                "oear": config.oear,
                "oebr": config.oebr,
                "oeas": config.oeas,
                "oebs": config.oebs,
                "oeag": oeag,
                "oebg": oebg,
                "rraw": config.rraw,
                "rrbw": config.rrbw,
                "rrai": config.rrai,
                "rrbi": config.rrbi,
                "rrar": config.rrar,
                "rrbr": config.rrbr,
                "rras": config.rras,
                "rrbs": config.rrbs,
                "rrag": rrag,
                "rrbg": rrbg,
                "tvaw": config.tvaw,
                "tvbw": config.tvbw,
                "tvai": config.tvai,
                "tvbi": config.tvbi,
                "tvar": config.tvar,
                "tvbr": config.tvbr,
                "tvas": config.tvas,
                "tvbs": config.tvbs,
                "tvag": tvag,
                "tvbg": tvbg,
                "muw": config.muw,
                "mui": config.mui,
                "mur": config.mur,
                "mus": config.mus,
                "mug": mug,
                "blinw": config.blinw,
                "blini": config.blini,
                "blinr": config.blinr,
                "blins": config.blins,
                "bling": bling,
            },
            origin=self._idx.origin_compute(),
            domain=self._idx.domain_compute(),
        )

        if config.consv_checker:
            self._moist_total_energy_and_water_mq = stencil_factory.from_origin_domain(
                func=moist_total_energy_and_water,
                externals={
                    "hydrostatic": config.hydrostatic,
                    "moist_q": True,
                    "c1_vap": config.c1_vap,
                    "c1_liq": config.c1_liq,
                    "c1_ice": config.c1_ice,
                    "lv00": config.lv00,
                    "li00": config.li00,
                    "c_air": config.c_air,
                    "timestep": config.dt_full,
                },
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )

            self._moist_total_energy_and_water = stencil_factory.from_origin_domain(
                func=moist_total_energy_and_water,
                externals={
                    "hydrostatic": config.hydrostatic,
                    "moist_q": False,
                    "c1_vap": config.c1_vap,
                    "c1_liq": config.c1_liq,
                    "c1_ice": config.c1_ice,
                    "lv00": config.lv00,
                    "li00": config.li00,
                    "c_air": config.c_air,
                    "timestep": config.dt_full,
                },
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )

            self._calc_sedimentation_energy_loss = stencil_factory.from_origin_domain(
                func=calc_sedimentation_energy_loss,
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )

        if (self.do_sedi_uv) or (self.do_sedi_w):
            self._update_temperature_pre_delp_q = stencil_factory.from_origin_domain(
                func=update_temperature_pre_delp_q,
                externals={
                    "do_sedi_uv": self.do_sedi_uv,
                    "do_sedi_w": self.do_sedi_w,
                    "c_air": config.c_air,
                    "c1_vap": config.c1_vap,
                    "c1_liq": config.c1_liq,
                    "c1_ice": config.c1_ice,
                },
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )

        self._convert_mass_mixing_to_specific_ratios_and_update_temperatures = (
            stencil_factory.from_origin_domain(
                func=convert_mass_mixing_to_specific_ratios_and_update_temperatures,
                externals={
                    "do_inline_mp": config.do_inline_mp,
                    "c_air": config.c_air,
                    "c1_vap": config.c1_vap,
                    "c1_liq": config.c1_liq,
                    "c1_ice": config.c1_ice,
                    "do_sedi_uv": config.do_sedi_uv,
                    "do_sedi_w": config.do_sedi_w,
                },
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )
        )

        self._calculate_total_energy_change_and_convert_temp = (
            stencil_factory.from_origin_domain(
                func=calculate_total_energy_change_and_convert_temp,
                externals={
                    "consv_te": consv_te,
                    "hydrostatic": self.hydrostatic,
                    "c_air": config.c_air,
                    "do_inline_mp": config.do_inline_mp,
                    "cp_heating": config.cp_heating,
                    "c1_vap": config.c1_vap,
                    "c1_liq": config.c1_liq,
                    "c1_ice": config.c1_ice,
                },
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )
        )

        if self.do_qa:
            self._cloud_fraction = CloudFraction(stencil_factory, config)

    def __call__(
        self,
        qvapor,
        qliquid,
        qrain,
        qice,
        qsnow,
        qgraupel,
        qcld,
        pt,
        density,
        delp,
        delz,
        pz,
        ua,
        va,
        wa,
        total_energy,
        particle_concentration_w,
        particle_concentration_r,
        particle_concentration_i,
        particle_concentration_s,
        particle_concentration_g,
        effective_diameter_w,
        effective_diameter_r,
        effective_diameter_i,
        effective_diameter_s,
        effective_diameter_g,
        optical_extinction_w,
        optical_extinction_r,
        optical_extinction_i,
        optical_extinction_s,
        optical_extinction_g,
        radar_reflectivity_w,
        radar_reflectivity_r,
        radar_reflectivity_i,
        radar_reflectivity_s,
        radar_reflectivity_g,
        terminal_velocity_w,
        terminal_velocity_r,
        terminal_velocity_i,
        terminal_velocity_s,
        terminal_velocity_g,
        qvapor0,
        qliquid0,
        qrain0,
        qice0,
        qsnow0,
        qgraupel0,
        u0,
        v0,
        w0,
        dp0,
        pt0,
        column_energy_change,
        column_energy_loss,
        column_vapor,
        column_water,
        column_rain,
        column_ice,
        column_snow,
        column_graupel,
        total_energy_dry_begin,
        total_energy_moist_begin,
        total_water_dry_begin,
        total_water_moist_begin,
        total_energy_bot_dry_begin,
        total_energy_bot_moist_begin,
        total_water_bot_dry_begin,
        total_water_bot_moist_begin,
        total_energy_dry_end,
        total_water_dry_end,
        total_energy_bot_dry_end,
        total_water_bot_dry_end,
        total_energy_moist_end,
        total_water_moist_end,
        total_energy_bot_moist_end,
        total_water_bot_moist_end,
        adj_vmr,
        h_var,
        gsize,
        last_step: bool,
    ):

        if (self.do_qa) and last_step:
            self._cloud_fraction(
                qvapor,
                qliquid,
                qrain,
                qice,
                qsnow,
                qgraupel,
                qcld,
                pt,
                density,
                pz,
                h_var,
                gsize,
            )

        self._calculate_particle_properties(
            qliquid,
            qrain,
            qice,
            qsnow,
            qgraupel,
            density,
            particle_concentration_w,
            particle_concentration_r,
            particle_concentration_i,
            particle_concentration_s,
            particle_concentration_g,
            effective_diameter_w,
            effective_diameter_r,
            effective_diameter_i,
            effective_diameter_s,
            effective_diameter_g,
            optical_extinction_w,
            optical_extinction_r,
            optical_extinction_i,
            optical_extinction_s,
            optical_extinction_g,
            radar_reflectivity_w,
            radar_reflectivity_r,
            radar_reflectivity_i,
            radar_reflectivity_s,
            radar_reflectivity_g,
            terminal_velocity_w,
            terminal_velocity_r,
            terminal_velocity_i,
            terminal_velocity_s,
            terminal_velocity_g,
        )

        if (self.do_sedi_uv) or (self.do_sedi_w):
            self._update_temperature_pre_delp_q(
                qvapor,
                qliquid,
                qrain,
                qice,
                qsnow,
                qgraupel,
                ua,
                va,
                wa,
                u0,
                v0,
                w0,
                pt,
                self._tzuv,
                self._tzw,
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
                total_energy_dry_end,
                total_water_dry_end,
                total_energy_bot_dry_end,
                total_water_bot_dry_end,
            )

            self._calc_sedimentation_energy_loss(
                column_energy_loss, column_energy_change, gsize
            )

        self._convert_mass_mixing_to_specific_ratios_and_update_temperatures(
            qvapor,
            qliquid,
            qrain,
            qice,
            qsnow,
            qgraupel,
            ua,
            va,
            wa,
            delp,
            qvapor0,
            qliquid0,
            qrain0,
            qice0,
            qsnow0,
            qgraupel0,
            u0,
            v0,
            w0,
            dp0,
            pt,
            self._tzuv,
            self._tzw,
            adj_vmr,
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
                total_energy_moist_end,
                total_water_moist_end,
                total_energy_bot_moist_end,
                total_water_bot_moist_end,
            )

        self._calculate_total_energy_change_and_convert_temp(
            qvapor,
            qliquid,
            qrain,
            qice,
            qsnow,
            qgraupel,
            total_energy,
            pt,
            pt0,
            delp,
            delz,
        )

        if self.consv_checker:
            total_energy_check(
                total_energy_dry_end,
                total_energy_moist_end,
                total_water_dry_end,
                total_water_moist_end,
                total_energy_dry_begin,
                total_energy_moist_begin,
                total_water_dry_begin,
                total_water_moist_begin,
                total_energy_bot_dry_end,
                total_energy_bot_moist_end,
                total_water_bot_dry_end,
                total_water_bot_moist_end,
                total_energy_bot_dry_begin,
                total_energy_bot_moist_begin,
                total_water_bot_dry_begin,
                total_water_bot_moist_begin,
                self._idx.isc,
                self._idx.iec,
                self._idx.jsc,
                self._idx.jec,
                self._te_err,
                self._tw_err,
            )


class FinalCalcs:
    def __init__(
        self,
        stencil_factory,
        config,
        consv_te,
    ):
        self._idx = stencil_factory.grid_indexing
        self.consv_te = consv_te
        self.do_qa = config.do_qa
        self.hydrostatic = config.hydrostatic
        self.consv_checker = config.consv_checker
        self.do_inline_mp = config.do_inline_mp
        self.fix_negative = config.fix_negative
        self._te_err = config.te_err
        self._tw_err = config.tw_err
        self.do_sedi_uv = config.do_sedi_uv
        self.do_sedi_w = config.do_sedi_w

        if config.consv_checker:
            self._moist_total_energy_and_water_mq = stencil_factory.from_origin_domain(
                func=moist_total_energy_and_water,
                externals={
                    "hydrostatic": config.hydrostatic,
                    "moist_q": True,
                    "c1_vap": config.c1_vap,
                    "c1_liq": config.c1_liq,
                    "c1_ice": config.c1_ice,
                    "lv00": config.lv00,
                    "li00": config.li00,
                    "c_air": config.c_air,
                    "timestep": config.dt_full,
                },
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )

            self._moist_total_energy_and_water = stencil_factory.from_origin_domain(
                func=moist_total_energy_and_water,
                externals={
                    "hydrostatic": config.hydrostatic,
                    "moist_q": False,
                    "c1_vap": config.c1_vap,
                    "c1_liq": config.c1_liq,
                    "c1_ice": config.c1_ice,
                    "lv00": config.lv00,
                    "li00": config.li00,
                    "c_air": config.c_air,
                    "timestep": config.dt_full,
                },
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )

            self._calc_sedimentation_energy_loss = stencil_factory.from_origin_domain(
                func=calc_sedimentation_energy_loss,
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )

        if (self.do_sedi_uv) or (self.do_sedi_w):
            self._update_temperature_pre_delp_q = stencil_factory.from_origin_domain(
                func=update_temperature_pre_delp_q,
                externals={
                    "do_sedi_uv": self.do_sedi_uv,
                    "do_sedi_w": self.do_sedi_w,
                    "c_air": config.c_air,
                    "c1_vap": config.c1_vap,
                    "c1_liq": config.c1_liq,
                    "c1_ice": config.c1_ice,
                },
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )

        self._convert_mass_mixing_to_specific_ratios_and_update_temperatures = (
            stencil_factory.from_origin_domain(
                func=convert_mass_mixing_to_specific_ratios_and_update_temperatures,
                externals={
                    "do_inline_mp": config.do_inline_mp,
                    "c_air": config.c_air,
                    "c1_vap": config.c1_vap,
                    "c1_liq": config.c1_liq,
                    "c1_ice": config.c1_ice,
                    "do_sedi_uv": config.do_sedi_uv,
                    "do_sedi_w": config.do_sedi_w,
                },
                origin=self._idx.origin_compute(),
                domain=self._idx.domain_compute(),
            )
        )

        self._calculate_total_energy_change_and_convert_temp = (
            stencil_factory.from_origin_domain(
                func=calculate_total_energy_change_and_convert_temp,
                externals={
                    "consv_te": consv_te,
                    "hydrostatic": self.hydrostatic,
                    "c_air": config.c_air,
                    "do_inline_mp": config.do_inline_mp,
                    "cp_heating": config.cp_heating,
                    "c1_vap": config.c1_vap,
                    "c1_liq": config.c1_liq,
                    "c1_ice": config.c1_ice,
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
        pt,
        ua,
        va,
        wa,
        total_energy,
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
        tzuv,
        tzw,
        column_energy_change,
        adj_vmr,
        gsize,
        column_vapor,
        column_water,
        column_rain,
        column_ice,
        column_snow,
        column_graupel,
        total_energy_moist_begin,
        total_water_moist_begin,
        total_energy_bot_moist_begin,
        total_water_bot_moist_begin,
        total_energy_dry_begin,
        total_water_dry_begin,
        total_energy_bot_dry_begin,
        total_water_bot_dry_begin,
        total_energy_dry_end,
        total_energy_moist_end,
        total_water_dry_end,
        total_water_moist_end,
        total_energy_bot_dry_end,
        total_energy_bot_moist_end,
        total_water_bot_dry_end,
        total_water_bot_moist_end,
        column_energy_loss,
    ):
        if (self.do_sedi_uv) or (self.do_sedi_w):
            self._update_temperature_pre_delp_q(
                qvapor,
                qliquid,
                qrain,
                qice,
                qsnow,
                qgraupel,
                ua,
                va,
                wa,
                u0,
                v0,
                w0,
                pt,
                tzuv,
                tzw,
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
                total_energy_dry_end,
                total_water_dry_end,
                total_energy_bot_dry_end,
                total_water_bot_dry_end,
            )

            self._calc_sedimentation_energy_loss(
                column_energy_loss, column_energy_change, gsize
            )

        self._convert_mass_mixing_to_specific_ratios_and_update_temperatures(
            qvapor,
            qliquid,
            qrain,
            qice,
            qsnow,
            qgraupel,
            ua,
            va,
            wa,
            delp,
            qvapor0,
            qliquid0,
            qrain0,
            qice0,
            qsnow0,
            qgraupel0,
            u0,
            v0,
            w0,
            dp0,
            pt,
            tzuv,
            tzw,
            adj_vmr,
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
                total_energy_moist_end,
                total_water_moist_end,
                total_energy_bot_moist_end,
                total_water_bot_moist_end,
            )

        self._calculate_total_energy_change_and_convert_temp(
            qvapor,
            qliquid,
            qrain,
            qice,
            qsnow,
            qgraupel,
            total_energy,
            pt,
            pt0,
            delp,
            delz,
        )

        if self.consv_checker:
            total_energy_check(
                total_energy_dry_end,
                total_energy_moist_end,
                total_water_dry_end,
                total_water_moist_end,
                total_energy_dry_begin,
                total_energy_moist_begin,
                total_water_dry_begin,
                total_water_moist_begin,
                total_energy_bot_dry_end,
                total_energy_bot_moist_end,
                total_water_bot_dry_end,
                total_water_bot_moist_end,
                total_energy_bot_dry_begin,
                total_energy_bot_moist_begin,
                total_water_bot_dry_begin,
                total_water_bot_moist_begin,
                self._idx.isc,
                self._idx.iec,
                self._idx.jsc,
                self._idx.jec,
                self._te_err,
                self._tw_err,
            )


class TranslateFinalCalculations(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "fin_qv", "mp3": True},
            "qliquid": {"serialname": "fin_ql", "mp3": True},
            "qrain": {"serialname": "fin_qr", "mp3": True},
            "qice": {"serialname": "fin_qi", "mp3": True},
            "qsnow": {"serialname": "fin_qs", "mp3": True},
            "qgraupel": {"serialname": "fin_qg", "mp3": True},
            "delp": {"serialname": "fin_delp", "mp3": True},
            "delz": {"serialname": "fin_delz", "mp3": True},
            "pt": {"serialname": "fin_pt", "mp3": True},
            "ua": {"serialname": "fin_ua", "mp3": True},
            "va": {"serialname": "fin_va", "mp3": True},
            "wa": {"serialname": "fin_wa", "mp3": True},
            "qvapor0": {"serialname": "fin_qv0", "mp3": True},
            "qliquid0": {"serialname": "fin_ql0", "mp3": True},
            "qrain0": {"serialname": "fin_qr0", "mp3": True},
            "qice0": {"serialname": "fin_qi0", "mp3": True},
            "qsnow0": {"serialname": "fin_qs0", "mp3": True},
            "qgraupel0": {"serialname": "fin_qg0", "mp3": True},
            "dp0": {"serialname": "fin_dp0", "mp3": True},
            "pt0": {"serialname": "fin_pt0", "mp3": True},
            "u0": {"serialname": "fin_u0", "mp3": True},
            "v0": {"serialname": "fin_v0", "mp3": True},
            "w0": {"serialname": "fin_w0", "mp3": True},
            "column_energy_change": {"serialname": "fin_dte", "mp3": True},
            "adj_vmr": {"serialname": "fin_adj_vmr", "mp3": True},
            "gsize": {"serialname": "fin_gsize", "mp3": True},
            "column_vapor": {"serialname": "fin_vapor", "mp3": True},
            "column_water": {"serialname": "fin_water", "mp3": True},
            "column_rain": {"serialname": "fin_rain", "mp3": True},
            "column_ice": {"serialname": "fin_ice", "mp3": True},
            "column_snow": {"serialname": "fin_snow", "mp3": True},
            "column_graupel": {"serialname": "fin_graupel", "mp3": True},
            "total_energy_moist_begin": {"serialname": "fin_ew0", "mp3": True},
            "total_water_moist_begin": {"serialname": "fin_ww0", "mp3": True},
            "total_energy_bot_moist_begin": {"serialname": "fin_bew0", "mp3": True},
            "total_water_bot_moist_begin": {"serialname": "fin_bww0", "mp3": True},
            "total_energy_dry_begin": {"serialname": "fin_ed0", "mp3": True},
            "total_water_dry_begin": {"serialname": "fin_wd0", "mp3": True},
            "total_energy_bot_dry_begin": {"serialname": "fin_bed0", "mp3": True},
            "total_water_bot_dry_begin": {"serialname": "fin_bwd0", "mp3": True},
            "total_energy_moist_end": {"serialname": "fin_ew", "mp3": True},
            "total_water_moist_end": {"serialname": "fin_ww", "mp3": True},
            "total_energy_bot_moist_end": {"serialname": "fin_bew", "mp3": True},
            "total_water_bot_moist_end": {"serialname": "fin_bww", "mp3": True},
            "total_energy_dry_end": {"serialname": "fin_ed", "mp3": True},
            "total_water_dry_end": {"serialname": "fin_wd", "mp3": True},
            "total_energy_bot_dry_end": {"serialname": "fin_bed", "mp3": True},
            "total_water_bot_dry_end": {"serialname": "fin_bwd", "mp3": True},
            "tzuv": {"serialname": "fin_tzuv", "mp3": True},
            "tzw": {"serialname": "fin_tzw", "mp3": True},
            "total_energy": {"serialname": "fin_te", "mp3": True},
            "column_energy_loss": {"serialname": "fin_te_loss", "mp3": True},
        }

        self.out_vars = {
            "qvapor": {"serialname": "fin_qv", "kend": namelist.npz, "mp3": True},
            "qliquid": {"serialname": "fin_ql", "kend": namelist.npz, "mp3": True},
            "qrain": {"serialname": "fin_qr", "kend": namelist.npz, "mp3": True},
            "qice": {"serialname": "fin_qi", "kend": namelist.npz, "mp3": True},
            "qsnow": {"serialname": "fin_qs", "kend": namelist.npz, "mp3": True},
            "qgraupel": {"serialname": "fin_qg", "kend": namelist.npz, "mp3": True},
            "delp": {"serialname": "fin_delp", "kend": namelist.npz, "mp3": True},
            "delz": {"serialname": "fin_delz", "kend": namelist.npz, "mp3": True},
            "pt": {"serialname": "fin_pt", "kend": namelist.npz, "mp3": True},
            "ua": {"serialname": "fin_ua", "kend": namelist.npz, "mp3": True},
            "va": {"serialname": "fin_va", "kend": namelist.npz, "mp3": True},
            "wa": {"serialname": "fin_wa", "kend": namelist.npz, "mp3": True},
            "column_energy_change": {"serialname": "fin_dte", "mp3": True},
            "adj_vmr": {"serialname": "fin_adj_vmr", "kend": namelist.npz, "mp3": True},
            "total_energy_moist_end": {
                "serialname": "fin_ew",
                "kend": namelist.npz,
                "mp3": True,
            },
            "total_water_moist_end": {
                "serialname": "fin_ww",
                "kend": namelist.npz,
                "mp3": True,
            },
            "total_energy_bot_moist_end": {"serialname": "fin_bew", "mp3": True},
            "total_water_bot_moist_end": {"serialname": "fin_bww", "mp3": True},
            "total_energy_dry_end": {
                "serialname": "fin_ed",
                "kend": namelist.npz,
                "mp3": True,
            },
            "total_water_dry_end": {
                "serialname": "fin_wd",
                "kend": namelist.npz,
                "mp3": True,
            },
            "total_energy_bot_dry_end": {"serialname": "fin_bed", "mp3": True},
            "total_water_bot_dry_end": {"serialname": "fin_bwd", "mp3": True},
            "total_energy": {"serialname": "fin_te", "kend": namelist.npz, "mp3": True},
            "column_energy_loss": {"serialname": "fin_te_loss", "mp3": True},
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

        compute_func = FinalCalcs(self.stencil_factory, self.config, consv_te=False)

        compute_func(**inputs)

        return self.slice_output(inputs)


class TranslatePostMP(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "post_qv", "mp3": True},
            "qliquid": {"serialname": "post_ql", "mp3": True},
            "qrain": {"serialname": "post_qr", "mp3": True},
            "qice": {"serialname": "post_qi", "mp3": True},
            "qsnow": {"serialname": "post_qs", "mp3": True},
            "qgraupel": {"serialname": "post_qg", "mp3": True},
            "delp": {"serialname": "post_delp", "mp3": True},
            "delz": {"serialname": "post_delz", "mp3": True},
            "pt": {"serialname": "post_pt", "mp3": True},
            "ua": {"serialname": "post_ua", "mp3": True},
            "va": {"serialname": "post_va", "mp3": True},
            "wa": {"serialname": "post_wa", "mp3": True},
            "pz": {"serialname": "post_pz", "mp3": True},
            "qvapor0": {"serialname": "post_qv0", "mp3": True},
            "qliquid0": {"serialname": "post_ql0", "mp3": True},
            "qrain0": {"serialname": "post_qr0", "mp3": True},
            "qice0": {"serialname": "post_qi0", "mp3": True},
            "qsnow0": {"serialname": "post_qs0", "mp3": True},
            "qgraupel0": {"serialname": "post_qg0", "mp3": True},
            "dp0": {"serialname": "post_dp0", "mp3": True},
            "pt0": {"serialname": "post_pt0", "mp3": True},
            "u0": {"serialname": "post_u0", "mp3": True},
            "v0": {"serialname": "post_v0", "mp3": True},
            "w0": {"serialname": "post_w0", "mp3": True},
            "column_energy_change": {"serialname": "post_dte", "mp3": True},
            "adj_vmr": {"serialname": "post_adj_vmr", "mp3": True},
            "h_var": {"serialname": "post_h_var", "mp3": True},
            "gsize": {"serialname": "post_gsize", "mp3": True},
            "column_vapor": {"serialname": "post_vapor", "mp3": True},
            "column_water": {"serialname": "post_water", "mp3": True},
            "column_rain": {"serialname": "post_rain", "mp3": True},
            "column_ice": {"serialname": "post_ice", "mp3": True},
            "column_snow": {"serialname": "post_snow", "mp3": True},
            "column_graupel": {"serialname": "post_graupel", "mp3": True},
            "total_energy_moist_begin": {"serialname": "post_ew0", "mp3": True},
            "total_water_moist_begin": {"serialname": "post_ww0", "mp3": True},
            "total_energy_bot_moist_begin": {"serialname": "post_bew0", "mp3": True},
            "total_water_bot_moist_begin": {"serialname": "post_bww0", "mp3": True},
            "total_energy_dry_begin": {"serialname": "post_ed0", "mp3": True},
            "total_water_dry_begin": {"serialname": "post_wd0", "mp3": True},
            "total_energy_bot_dry_begin": {"serialname": "post_bed0", "mp3": True},
            "total_water_bot_dry_begin": {"serialname": "post_bwd0", "mp3": True},
            "total_energy_moist_end": {"serialname": "post_ew", "mp3": True},
            "total_water_moist_end": {"serialname": "post_ww", "mp3": True},
            "total_energy_bot_moist_end": {"serialname": "post_bew", "mp3": True},
            "total_water_bot_moist_end": {"serialname": "post_bww", "mp3": True},
            "total_energy_dry_end": {"serialname": "post_ed", "mp3": True},
            "total_water_dry_end": {"serialname": "post_wd", "mp3": True},
            "total_energy_bot_dry_end": {"serialname": "post_bed", "mp3": True},
            "total_water_bot_dry_end": {"serialname": "post_bwd", "mp3": True},
            "total_energy": {"serialname": "post_te", "mp3": True},
            "column_energy_loss": {"serialname": "post_te_loss", "mp3": True},
            "density": {"serialname": "post_den", "mp3": True},
            "particle_concentration_w": {
                "serialname": "post_pcw",
                "mp3": True,
            },
            "effective_diameter_w": {
                "serialname": "post_edw",
                "mp3": True,
            },
            "optical_extinction_w": {
                "serialname": "post_oew",
                "mp3": True,
            },
            "radar_reflectivity_w": {
                "serialname": "post_rrw",
                "mp3": True,
            },
            "terminal_velocity_w": {
                "serialname": "post_tvw",
                "mp3": True,
            },
            "particle_concentration_r": {
                "serialname": "post_pcr",
                "mp3": True,
            },
            "effective_diameter_r": {
                "serialname": "post_edr",
                "mp3": True,
            },
            "optical_extinction_r": {
                "serialname": "post_oer",
                "mp3": True,
            },
            "radar_reflectivity_r": {
                "serialname": "post_rrr",
                "mp3": True,
            },
            "terminal_velocity_r": {
                "serialname": "post_tvr",
                "mp3": True,
            },
            "particle_concentration_i": {
                "serialname": "post_pci",
                "mp3": True,
            },
            "effective_diameter_i": {
                "serialname": "post_edi",
                "mp3": True,
            },
            "optical_extinction_i": {
                "serialname": "post_oei",
                "mp3": True,
            },
            "radar_reflectivity_i": {
                "serialname": "post_rri",
                "mp3": True,
            },
            "terminal_velocity_i": {
                "serialname": "post_tvi",
                "mp3": True,
            },
            "particle_concentration_s": {
                "serialname": "post_pcs",
                "mp3": True,
            },
            "effective_diameter_s": {
                "serialname": "post_eds",
                "mp3": True,
            },
            "optical_extinction_s": {
                "serialname": "post_oes",
                "mp3": True,
            },
            "radar_reflectivity_s": {
                "serialname": "post_rrs",
                "mp3": True,
            },
            "terminal_velocity_s": {
                "serialname": "post_tvs",
                "mp3": True,
            },
            "particle_concentration_g": {
                "serialname": "post_pcg",
                "mp3": True,
            },
            "effective_diameter_g": {
                "serialname": "post_edg",
                "mp3": True,
            },
            "optical_extinction_g": {
                "serialname": "post_oeg",
                "mp3": True,
            },
            "radar_reflectivity_g": {
                "serialname": "post_rrg",
                "mp3": True,
            },
            "terminal_velocity_g": {
                "serialname": "post_tvg",
                "mp3": True,
            },
            "qcld": {"serialname": "post_qa", "mp3": True},
        }

        self.out_vars = {
            "qvapor": {"serialname": "post_qv", "kend": namelist.npz, "mp3": True},
            "qliquid": {"serialname": "post_ql", "kend": namelist.npz, "mp3": True},
            "qrain": {"serialname": "post_qr", "kend": namelist.npz, "mp3": True},
            "qice": {"serialname": "post_qi", "kend": namelist.npz, "mp3": True},
            "qsnow": {"serialname": "post_qs", "kend": namelist.npz, "mp3": True},
            "qgraupel": {"serialname": "post_qg", "kend": namelist.npz, "mp3": True},
            "qcld": {"serialname": "post_qa", "kend": namelist.npz, "mp3": True},
            "delp": {"serialname": "post_delp", "kend": namelist.npz, "mp3": True},
            "delz": {"serialname": "post_delz", "kend": namelist.npz, "mp3": True},
            "pt": {"serialname": "post_pt", "kend": namelist.npz, "mp3": True},
            "ua": {"serialname": "post_ua", "kend": namelist.npz, "mp3": True},
            "va": {"serialname": "post_va", "kend": namelist.npz, "mp3": True},
            "wa": {"serialname": "post_wa", "kend": namelist.npz, "mp3": True},
            "column_energy_change": {"serialname": "post_dte", "mp3": True},
            "adj_vmr": {
                "serialname": "post_adj_vmr",
                "kend": namelist.npz,
                "mp3": True,
            },
            "total_energy_moist_end": {
                "serialname": "post_ew",
                "kend": namelist.npz,
                "mp3": True,
            },
            "total_water_moist_end": {
                "serialname": "post_ww",
                "kend": namelist.npz,
                "mp3": True,
            },
            "total_energy_bot_moist_end": {"serialname": "post_bew", "mp3": True},
            "total_water_bot_moist_end": {"serialname": "post_bww", "mp3": True},
            "total_energy_dry_end": {
                "serialname": "post_ed",
                "kend": namelist.npz,
                "mp3": True,
            },
            "total_water_dry_end": {
                "serialname": "post_wd",
                "kend": namelist.npz,
                "mp3": True,
            },
            "total_energy_bot_dry_end": {"serialname": "post_bed", "mp3": True},
            "total_water_bot_dry_end": {"serialname": "post_bwd", "mp3": True},
            "total_energy": {
                "serialname": "post_te",
                "kend": namelist.npz,
                "mp3": True,
            },
            "column_energy_loss": {"serialname": "post_te_loss", "mp3": True},
            "particle_concentration_w": {
                "serialname": "post_pcw",
                "kend": namelist.npz,
                "mp3": True,
            },
            "effective_diameter_w": {
                "serialname": "post_edw",
                "kend": namelist.npz,
                "mp3": True,
            },
            "optical_extinction_w": {
                "serialname": "post_oew",
                "kend": namelist.npz,
                "mp3": True,
            },
            "radar_reflectivity_w": {
                "serialname": "post_rrw",
                "kend": namelist.npz,
                "mp3": True,
            },
            "terminal_velocity_w": {
                "serialname": "post_tvw",
                "kend": namelist.npz,
                "mp3": True,
            },
            "particle_concentration_r": {
                "serialname": "post_pcr",
                "kend": namelist.npz,
                "mp3": True,
            },
            "effective_diameter_r": {
                "serialname": "post_edr",
                "kend": namelist.npz,
                "mp3": True,
            },
            "optical_extinction_r": {
                "serialname": "post_oer",
                "kend": namelist.npz,
                "mp3": True,
            },
            "radar_reflectivity_r": {
                "serialname": "post_rrr",
                "kend": namelist.npz,
                "mp3": True,
            },
            "terminal_velocity_r": {
                "serialname": "post_tvr",
                "kend": namelist.npz,
                "mp3": True,
            },
            "particle_concentration_i": {
                "serialname": "post_pci",
                "kend": namelist.npz,
                "mp3": True,
            },
            "effective_diameter_i": {
                "serialname": "post_edi",
                "kend": namelist.npz,
                "mp3": True,
            },
            "optical_extinction_i": {
                "serialname": "post_oei",
                "kend": namelist.npz,
                "mp3": True,
            },
            "radar_reflectivity_i": {
                "serialname": "post_rri",
                "kend": namelist.npz,
                "mp3": True,
            },
            "terminal_velocity_i": {
                "serialname": "post_tvi",
                "kend": namelist.npz,
                "mp3": True,
            },
            "particle_concentration_s": {
                "serialname": "post_pcs",
                "kend": namelist.npz,
                "mp3": True,
            },
            "effective_diameter_s": {
                "serialname": "post_eds",
                "kend": namelist.npz,
                "mp3": True,
            },
            "optical_extinction_s": {
                "serialname": "post_oes",
                "kend": namelist.npz,
                "mp3": True,
            },
            "radar_reflectivity_s": {
                "serialname": "post_rrs",
                "kend": namelist.npz,
                "mp3": True,
            },
            "terminal_velocity_s": {
                "serialname": "post_tvs",
                "kend": namelist.npz,
                "mp3": True,
            },
            "particle_concentration_g": {
                "serialname": "post_pcg",
                "kend": namelist.npz,
                "mp3": True,
            },
            "effective_diameter_g": {
                "serialname": "post_edg",
                "kend": namelist.npz,
                "mp3": True,
            },
            "optical_extinction_g": {
                "serialname": "post_oeg",
                "kend": namelist.npz,
                "mp3": True,
            },
            "radar_reflectivity_g": {
                "serialname": "post_rrg",
                "kend": namelist.npz,
                "mp3": True,
            },
            "terminal_velocity_g": {
                "serialname": "post_tvg",
                "kend": namelist.npz,
                "mp3": True,
            },
        }

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        pconf = PhysicsConfig.from_namelist(namelist)
        self.config = pconf.microphysics
        self.config.do_mp_table_emulation = True

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

        inputs["last_step"] = True

        compute_func = PostMP(
            self.stencil_factory, self.config, self.quantity_factory, consv_te=False
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
