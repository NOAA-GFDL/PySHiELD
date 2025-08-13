from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import FloatField
from ndsl.initialization.sizer import SubtileGridSizer
from ndsl.initialization.allocator import QuantityFactory
from ndsl.namelist import Namelist
from pyshield._config import MicroPhysicsConfig, PhysicsConfig
from pyshield.stencils.shield_microphysics.gfdl_cld_mp_driver import (
    calculate_particle_properties,
)
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class CalcParticleProperties:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        config: MicroPhysicsConfig,
    ):
        self._idx = stencil_factory.grid_indexing

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

    def __call__(
        self,
        qliquid: FloatField,
        qrain: FloatField,
        qice: FloatField,
        qsnow: FloatField,
        qgraupel: FloatField,
        density: FloatField,
        particle_concentration_w: FloatField,
        particle_concentration_r: FloatField,
        particle_concentration_i: FloatField,
        particle_concentration_s: FloatField,
        particle_concentration_g: FloatField,
        effective_diameter_w: FloatField,
        effective_diameter_r: FloatField,
        effective_diameter_i: FloatField,
        effective_diameter_s: FloatField,
        effective_diameter_g: FloatField,
        optical_extinction_w: FloatField,
        optical_extinction_r: FloatField,
        optical_extinction_i: FloatField,
        optical_extinction_s: FloatField,
        optical_extinction_g: FloatField,
        radar_reflectivity_w: FloatField,
        radar_reflectivity_r: FloatField,
        radar_reflectivity_i: FloatField,
        radar_reflectivity_s: FloatField,
        radar_reflectivity_g: FloatField,
        terminal_velocity_w: FloatField,
        terminal_velocity_r: FloatField,
        terminal_velocity_i: FloatField,
        terminal_velocity_s: FloatField,
        terminal_velocity_g: FloatField,
    ):
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


class TranslateParticleProperties(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qliquid": {"serialname": "pp_ql", "shield": True},
            "qrain": {"serialname": "pp_qr", "shield": True},
            "qice": {"serialname": "pp_qi", "shield": True},
            "qsnow": {"serialname": "pp_qs", "shield": True},
            "qgraupel": {"serialname": "pp_qg", "shield": True},
            "density": {"serialname": "pp_den", "shield": True},
            "particle_concentration_w": {
                "serialname": "pp_pcw",
                "shield": True,
            },
            "effective_diameter_w": {
                "serialname": "pp_edw",
                "shield": True,
            },
            "optical_extinction_w": {
                "serialname": "pp_oew",
                "shield": True,
            },
            "radar_reflectivity_w": {
                "serialname": "pp_rrw",
                "shield": True,
            },
            "terminal_velocity_w": {
                "serialname": "pp_tvw",
                "shield": True,
            },
            "particle_concentration_r": {
                "serialname": "pp_pcr",
                "shield": True,
            },
            "effective_diameter_r": {
                "serialname": "pp_edr",
                "shield": True,
            },
            "optical_extinction_r": {
                "serialname": "pp_oer",
                "shield": True,
            },
            "radar_reflectivity_r": {
                "serialname": "pp_rrr",
                "shield": True,
            },
            "terminal_velocity_r": {
                "serialname": "pp_tvr",
                "shield": True,
            },
            "particle_concentration_i": {
                "serialname": "pp_pci",
                "shield": True,
            },
            "effective_diameter_i": {
                "serialname": "pp_edi",
                "shield": True,
            },
            "optical_extinction_i": {
                "serialname": "pp_oei",
                "shield": True,
            },
            "radar_reflectivity_i": {
                "serialname": "pp_rri",
                "shield": True,
            },
            "terminal_velocity_i": {
                "serialname": "pp_tvi",
                "shield": True,
            },
            "particle_concentration_s": {
                "serialname": "pp_pcs",
                "shield": True,
            },
            "effective_diameter_s": {
                "serialname": "pp_eds",
                "shield": True,
            },
            "optical_extinction_s": {
                "serialname": "pp_oes",
                "shield": True,
            },
            "radar_reflectivity_s": {
                "serialname": "pp_rrs",
                "shield": True,
            },
            "terminal_velocity_s": {
                "serialname": "pp_tvs",
                "shield": True,
            },
            "particle_concentration_g": {
                "serialname": "pp_pcg",
                "shield": True,
            },
            "effective_diameter_g": {
                "serialname": "pp_edg",
                "shield": True,
            },
            "optical_extinction_g": {
                "serialname": "pp_oeg",
                "shield": True,
            },
            "radar_reflectivity_g": {
                "serialname": "pp_rrg",
                "shield": True,
            },
            "terminal_velocity_g": {
                "serialname": "pp_tvg",
                "shield": True,
            },
        }

        self.out_vars = {
            "particle_concentration_w": {
                "serialname": "pp_pcw",
                "kend": namelist.npz,
                "shield": True,
            },
            "effective_diameter_w": {
                "serialname": "pp_edw",
                "kend": namelist.npz,
                "shield": True,
            },
            "optical_extinction_w": {
                "serialname": "pp_oew",
                "kend": namelist.npz,
                "shield": True,
            },
            "radar_reflectivity_w": {
                "serialname": "pp_rrw",
                "kend": namelist.npz,
                "shield": True,
            },
            "terminal_velocity_w": {
                "serialname": "pp_tvw",
                "kend": namelist.npz,
                "shield": True,
            },
            "particle_concentration_r": {
                "serialname": "pp_pcr",
                "kend": namelist.npz,
                "shield": True,
            },
            "effective_diameter_r": {
                "serialname": "pp_edr",
                "kend": namelist.npz,
                "shield": True,
            },
            "optical_extinction_r": {
                "serialname": "pp_oer",
                "kend": namelist.npz,
                "shield": True,
            },
            "radar_reflectivity_r": {
                "serialname": "pp_rrr",
                "kend": namelist.npz,
                "shield": True,
            },
            "terminal_velocity_r": {
                "serialname": "pp_tvr",
                "kend": namelist.npz,
                "shield": True,
            },
            "particle_concentration_i": {
                "serialname": "pp_pci",
                "kend": namelist.npz,
                "shield": True,
            },
            "effective_diameter_i": {
                "serialname": "pp_edi",
                "kend": namelist.npz,
                "shield": True,
            },
            "optical_extinction_i": {
                "serialname": "pp_oei",
                "kend": namelist.npz,
                "shield": True,
            },
            "radar_reflectivity_i": {
                "serialname": "pp_rri",
                "kend": namelist.npz,
                "shield": True,
            },
            "terminal_velocity_i": {
                "serialname": "pp_tvi",
                "kend": namelist.npz,
                "shield": True,
            },
            "particle_concentration_s": {
                "serialname": "pp_pcs",
                "kend": namelist.npz,
                "shield": True,
            },
            "effective_diameter_s": {
                "serialname": "pp_eds",
                "kend": namelist.npz,
                "shield": True,
            },
            "optical_extinction_s": {
                "serialname": "pp_oes",
                "kend": namelist.npz,
                "shield": True,
            },
            "radar_reflectivity_s": {
                "serialname": "pp_rrs",
                "kend": namelist.npz,
                "shield": True,
            },
            "terminal_velocity_s": {
                "serialname": "pp_tvs",
                "kend": namelist.npz,
                "shield": True,
            },
            "particle_concentration_g": {
                "serialname": "pp_pcg",
                "kend": namelist.npz,
                "shield": True,
            },
            "effective_diameter_g": {
                "serialname": "pp_edg",
                "kend": namelist.npz,
                "shield": True,
            },
            "optical_extinction_g": {
                "serialname": "pp_oeg",
                "kend": namelist.npz,
                "shield": True,
            },
            "radar_reflectivity_g": {
                "serialname": "pp_rrg",
                "kend": namelist.npz,
                "shield": True,
            },
            "terminal_velocity_g": {
                "serialname": "pp_tvg",
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

        compute_func = CalcParticleProperties(
            self.stencil_factory,
            self.config,
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
