from ndsl.initialization.allocator import QuantityFactory
from ndsl.initialization.sizer import SubtileGridSizer
from pySHiELD._config import ShallowConvectionConfig
from pySHiELD.stencils.shallow_convection import ScaleAwareMassFluxShallowConvection
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateShalConv(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "delp": {"serialname": "delta", "shield": True},
            "prslp": {"serialname": "prsl", "shield": True},
            "psp": {"serialname": "pgr", "shield": True},
            "phil": {"serialname": "phil", "shield": True},
            "qtr": {"serialname": "clw", "shield": True},
            "q1": {"serialname": "gq0", "shield": True},
            "t1": {"serialname": "gt0", "shield": True},
            "u1": {"serialname": "gu0", "shield": True},
            "v1": {"serialname": "gv0", "shield": True},
            "rn": {"serialname": "rain1", "shield": True},
            "kbot": {"serialname": "kbot", "shield": True},
            "ktop": {"serialname": "ktop", "shield": True},
            "kcnv": {"serialname": "kcnv", "shield": True},
            "islimsk": {"serialname": "islmsk", "shield": True},
            "garea": {"serialname": "garea", "shield": True},
            "dot": {"serialname": "vvl", "shield": True},
            "ncld": {"serialname": "ncld", "shield": True},
            "hpbl": {"serialname": "hpbl", "shield": True},
            "ud_mf": {"serialname": "ud_mf", "shield": True},
            "dt_mf": {"serialname": "dt_mf", "shield": True},
            "cnvw": {"serialname": "cnvw", "shield": True},
            "cnvc": {"serialname": "cnvc", "shield": True},
        }
        self.in_vars["parameters"] = [
            "clam_shal",
            "c0s_shal",
            "c1_shal",
            "ncld",
            "pgcon_shal",
            "asolfac_shal",
            "dtp",
            "itc",
            "ntchm",
            "ntk",
            "nsamftrac",
            "ser_fscav",
        ]
        self.out_vars = {
            "delp": {"serialname": "delta", "shield": True},
            "prslp": {"serialname": "prsl", "shield": True},
            "psp": {"serialname": "pgr", "shield": True},
            "phil": {"serialname": "phil", "shield": True},
            "q1": {"serialname": "gq0", "shield": True},
            "t1": {"serialname": "gt0", "shield": True},
            "u1": {"serialname": "gu0", "shield": True},
            "v1": {"serialname": "gv0", "shield": True},
            "qtr": {"serialname": "clw", "shield": True},
            "rn": {"serialname": "rain1", "shield": True},
            "kbot": {"serialname": "kbot", "shield": True},
            "ktop": {"serialname": "ktop", "shield": True},
            "kcnv": {"serialname": "kcnv", "shield": True},
            "dot": {"serialname": "vvl", "shield": True},
            "hpbl": {"serialname": "hpbl", "shield": True},
            "ud_mf": {"serialname": "ud_mf", "shield": True},
            "dt_mf": {"serialname": "dt_mf", "shield": True},
            "cnvw": {"serialname": "cnvw", "shield": True},
            "cnvc": {"serialname": "cnvc", "shield": True},
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
            dt_atmos=inputs.pop("dtp"),
            ntke=inputs.pop("ntk"),
            ntr=inputs.pop("nsamftrac"),
            ncld=inputs.pop("ncld"),
            ntchm=inputs.pop("ntchm"),
            itc=inputs.pop("itc"),
            clam_shal=inputs.pop("clam_shal"),
            c0s_shal=inputs.pop("c0s_shal"),
            c1_shal=inputs.pop("c1_shal"),
            pgcon_shal=inputs.pop("pgcon_shal"),
            asolfac_shal=inputs.pop("asolfac_shal"),
            fscav=inputs.pop("ser_fscav"),
        )
        self.compute_func = ScaleAwareMassFluxShallowConvection(
            self.stencil_factory,
            self.quantity_factory,
            config,
        )
        assert self._daily_mean == inputs.pop("daily_mean")
        self.compute_func(**inputs)
        return self.slice_output(inputs)
