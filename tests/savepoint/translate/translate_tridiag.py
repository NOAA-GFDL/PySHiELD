from ndsl.dsl.stencil import StencilFactory
from ndsl import copy_defn
from pySHiELD.stencils.pbl.tridiag import tridit, tridi2, tridin
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py

class TridiT:
    def __init__(
        self,
        stencil_factory: StencilFactory,
    ):
        idx = stencil_factory.grid_indexing
        self._tridit = stencil_factory.from_origin_domain(
            func=tridit,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )
        self._copy = stencil_factory.from_origin_domain(
            func=copy_defn,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

    def __call__(
        self,
        au,
        ad,
        al,
        f1,
        cu,
        at,
    ):
        self._tridit(
            au,
            ad,
            al,
            f1,
        )
        self._copy(
            au,
            cu,
        )
        self._copy(
            f1,
            at,
        )

class Tridi2:
    def __init__(
        self,
        stencil_factory: StencilFactory,
    ):
        idx = stencil_factory.grid_indexing
        self._tridi2 = stencil_factory.from_origin_domain(
            func=tridi2,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )
        self._copy = stencil_factory.from_origin_domain(
            func=copy_defn,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

    def __call__(
        self,
        au,
        ad,
        al,
        f1,
        f2,
        cu,
        a1,
        a2,
    ):
        self._tridi2(
            f1,
            f2,
            au,
            al,
            ad,
            au,
            f1,
            f2,
        )
        self._copy(au, cu)
        self._copy(f1, a1)
        self._copy(f2, a2)

class TridiN:
    def __init__(
        self,
        stencil_factory: StencilFactory,
    ):
        idx = stencil_factory.grid_indexing
        self._tridin = stencil_factory.from_origin_domain(
            func=tridin,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )
        self._copy = stencil_factory.from_origin_domain(
            func=copy_defn,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

    def __call__(
        self,
        au,
        ad,
        al,
        f1,
        f2,
        cu,
        a1,
        a2,
        nt,
    ):
        self._tridin(
            al,
            ad,
            au,
            f1,
            f2,
            au,
            f1,
            f2,
            nt,
        )
        self._copy(au, cu)
        self._copy(f1, a1)
        self._copy(f2, a2)


class TranslateTridit(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "al": {"shield": True, "serialname": "cl"},
            "au": {"shield": True, "kend": namelist.npz - 1},
            "ad": {"shield": True, "serialname": "cd"},
            "f1": {"shield": True, "serialname": "rt"},
            "cu": {"shield": True, "kend": namelist.npz - 1},
            "at": {"shield": True},
        }
        self.out_vars = {
            "al": {"shield": True, "serialname": "cl"},
            "au": {"shield": True, "kend": namelist.npz - 1},
            "ad": {"shield": True, "serialname": "cd"},
            "f1": {"shield": True, "serialname": "rt"},
            "cu": {"shield": True, "kend": namelist.npz - 1},
            "at": {"shield": True},
        }
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        compute_func = TridiT(self.stencil_factory,)

        compute_func(**inputs)

        return self.slice_output(inputs)

class TranslateTridi2(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "al": {"shield": True, "serialname": "cl"},
            "au": {"shield": True, "kend": namelist.npz - 1},
            "ad": {"shield": True, "serialname": "cm"},
            "f1": {"shield": True, "serialname": "r1"},
            "f2": {"shield": True, "serialname": "r2"},
            "cu": {"shield": True, "kend": namelist.npz - 1},
            "a1": {"shield": True},
            "a2": {"shield": True},
        }
        self.out_vars = {
            "al": {"shield": True, "serialname": "cl"},
            "au": {"shield": True, "kend": namelist.npz - 1},
            "ad": {"shield": True, "serialname": "cm"},
            "f1": {"shield": True, "serialname": "r1"},
            "f2": {"shield": True, "serialname": "r2"},
            "cu": {"shield": True, "kend": namelist.npz - 1},
            "a1": {"shield": True},
            "a2": {"shield": True},
        }
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        compute_func = Tridi2(self.stencil_factory,)

        compute_func(**inputs)

        return self.slice_output(inputs)

class TranslateTridin(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "al": {"shield": True, "serialname": "cl"},
            "au": {"shield": True, "kend": namelist.npz - 1},
            "ad": {"shield": True, "serialname": "cm"},
            "f1": {"shield": True, "serialname": "r1"},
            "f2": {"shield": True, "serialname": "r2"},
            "cu": {"shield": True, "kend": namelist.npz - 1},
            "a1": {"shield": True},
            "a2": {"shield": True},
        }
        self.in_vars.parameters = [
            "nt"
        ]
        self.out_vars = {
            "al": {"shield": True, "serialname": "cl"},
            "au": {"shield": True, "kend": namelist.npz - 1},
            "ad": {"shield": True, "serialname": "cm"},
            "f1": {"shield": True, "serialname": "r1"},
            "f2": {"shield": True, "serialname": "r2"},
            "cu": {"shield": True, "kend": namelist.npz - 1},
            "a1": {"shield": True},
            "a2": {"shield": True},
        }
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        compute_func = Tridi2(self.stencil_factory,)

        compute_func(**inputs)

        return self.slice_output(inputs)
