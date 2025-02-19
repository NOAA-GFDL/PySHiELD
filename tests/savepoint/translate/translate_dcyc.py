from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py
from pySHiELD.stencils.physics import interpolate_radiation

class RadInterp:
    def __init__(self):
        pass
    def __call__(self, *args, **kwds):
        pass

class TranslateRadInterp(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "pe": {"serialname": "IPD_prsi"},
            "q": {"serialname": "IPD_gq0"},
        }
        self.in_vars["parameters"] = ["rdt"]
        self.out_vars = {
            "q": {"serialname": "IPD_qvapor", "kend": namelist.npz - 1},
        }
        self.grid_indexing = stencil_factory.grid_indexing
        self.compute_func = stencil_factory.from_origin_domain(
            interpolate_radiation,
            externals={
                "daily_mean": namelist.daily_mean
            },
            origin=self.grid_indexing.origin_full(),
            domain=self.grid_indexing.domain_full(),
        )
    
    def compute(self, inputs):
        self.compute_func(**inputs)
        return self.slice_output(inputs)