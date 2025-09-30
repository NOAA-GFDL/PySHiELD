from ndsl.initialization.allocator import QuantityFactory
from ndsl.initialization.sizer import SubtileGridSizer
from pyshield.stencils.shallow_convection import ScaleAwareMassFluxShallowConvection, ShallowConvectionConfig
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateShalConv(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "delp": {"serialname": "sc_delta", "shield": True},
            "prslp": {"serialname": "sc_prsl", "shield": True},
            "psp": {"serialname": "sc_pgr", "shield": True},
            "phil": {"serialname": "sc_phil", "shield": True},
            "qtr": {"serialname": "sc_clw", "shield": True},
            "q1": {"serialname": "sc_gq0", "shield": True},
            "t1": {"serialname": "sc_gt0", "shield": True},
            "u1": {"serialname": "sc_gu0", "shield": True},
            "v1": {"serialname": "sc_gv0", "shield": True},
            "rn": {"serialname": "sc_rain1", "shield": True},
            "kbot": {"serialname": "sc_kbot", "shield": True, "index_variable": True},
            "ktop": {"serialname": "sc_ktop", "shield": True, "index_variable": True},
            "kcnv": {"serialname": "sc_kcnv", "shield": True},
            "islimsk": {"serialname": "sc_islmsk", "shield": True},
            "garea": {"serialname": "sc_garea", "shield": True},
            "dot": {"serialname": "sc_vvl", "shield": True},
            "hpbl": {"serialname": "sc_hpbl", "shield": True},
            "ud_mf": {"serialname": "sc_ud_mf", "shield": True},
            "dt_mf": {"serialname": "sc_dt_mf", "shield": True},
            "cnvw": {"serialname": "sc_cnvw", "shield": True},
            "cnvc": {"serialname": "sc_cnvc", "shield": True},
        }
        self.in_vars["parameters"] = [
            "sc_clam_shal",
            "sc_c0s_shal",
            "sc_c1_shal",
            "sc_ncld",
            "sc_pgcon_shal",
            "sc_asolfac_shal",
            "sc_dtp",
            "sc_itc",
            "sc_ntchm",
            "sc_ntk",
            "sc_nsamftrac",
            "sc_ser_fscav",
        ]
        self.out_vars = {
            "delp": {"serialname": "sc_delta", "shield": True},  #
            "prslp": {"serialname": "sc_prsl", "shield": True},  #
            "psp": {"serialname": "sc_pgr", "shield": True},  #
            "phil": {"serialname": "sc_phil", "shield": True},  #
            "q1": {"serialname": "sc_gq0", "shield": True},
            "t1": {"serialname": "sc_gt0", "shield": True},
            "u1": {"serialname": "sc_gu0", "shield": True},
            "v1": {"serialname": "sc_gv0", "shield": True},
            "qtr": {"serialname": "sc_clw", "shield": True},
            "rn": {"serialname": "sc_rain1", "shield": True},
            "kbot": {"serialname": "sc_kbot", "shield": True, "index_variable": True},
            "ktop": {"serialname": "sc_ktop", "shield": True, "index_variable": True},
            "kcnv": {"serialname": "sc_kcnv", "shield": True},  #
            "dot": {"serialname": "sc_vvl", "shield": True},  #
            "hpbl": {"serialname": "sc_hpbl", "shield": True},  #
            "ud_mf": {"serialname": "sc_ud_mf", "shield": True},
            "dt_mf": {"serialname": "sc_dt_mf", "shield": True},
            "cnvw": {"serialname": "sc_cnvw", "shield": True},
            "cnvc": {"serialname": "sc_cnvc", "shield": True},
        }
        self.stencil_factory = stencil_factory

        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npx - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        self.quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

        self.grid_indexing = stencil_factory.grid_indexing

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        config = ShallowConvectionConfig(
            dt_atmos=inputs.pop("sc_dtp"),
            ntke=int(inputs.pop("sc_ntk") - 1),
            nsamftrac=int(inputs.pop("sc_nsamftrac")),
            ncld=int(inputs.pop("sc_ncld")),
            ntchm=int(inputs.pop("sc_ntchm")),
            ntiw=0,
            ntcw=1,
            itc=int(inputs.pop("sc_itc") - 1),
            clam_shal=inputs.pop("sc_clam_shal"),
            c0s_shal=inputs.pop("sc_c0s_shal"),
            c1_shal=inputs.pop("sc_c1_shal"),
            pgcon_shal=inputs.pop("sc_pgcon_shal"),
            asolfac_shal=inputs.pop("sc_asolfac_shal"),
            fscav=inputs.pop("sc_ser_fscav"),
        )
        self.compute_func = ScaleAwareMassFluxShallowConvection(
            self.stencil_factory,
            self.quantity_factory,
            config,
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)
