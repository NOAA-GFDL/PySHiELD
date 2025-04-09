from ndsl import QuantityFactory, StencilFactory
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.typing import (
    Bool,
    BoolField,
    BoolFieldIJ,
    Float,
    FloatField,
    FloatFieldIJ,
    Int,
    IntField,
    IntFieldIJ,
)
from ndsl.initialization.sizer import SubtileGridSizer
from gt4py.cartesian.gtscript import FORWARD, computation, interval
from pySHiELD.stencils.shallow_convection import (
    stencil_static1, stencil_update_kbcon1_cnvflg,
    stencil_static9, stencil_static12, feedback_control_update_mass_flux
)
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py

def set_pfld_kbcon(
    kbcon: IntFieldIJ,
    k_mask: IntField,
    pfld: FloatField,
    pfld_kbcon: FloatFieldIJ
):
    with computation(FORWARD), interval(...):
        if k_mask == kbcon:
            pfld_kbcon = pfld

class Static1:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ):
        grid_indexing = stencil_factory.grid_indexing

        self._heo_kb = quantity_factory.zeros([X_DIM, Y_DIM], units="unknown", dtype=Float)
        self._k_mask = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Int,
        )
        for k in range(grid_indexing.domain[2]):
            self._k_mask.data[:, :, k] = k
        
        self._static1 = stencil_factory.from_origin_domain(
            func=stencil_static1,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
    
    def __call__(
        self,
        cnvflg: BoolFieldIJ,
        flg: BoolFieldIJ,
        kbcon: IntFieldIJ,
        kmax: IntFieldIJ,
        kbm: IntFieldIJ,
        kb: IntFieldIJ,
        heo: FloatField,
        heso: FloatField,
    ):
        self._static1(
            cnvflg,
            flg,
            kbcon,
            kmax,
            self._k_mask,
            kbm,
            kb,
            self._heo_kb,
            heo,
            heso,
        )

class UpdateKB9:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ):
        grid_indexing = stencil_factory.grid_indexing

        self._pfld_kbcon = quantity_factory.zeros([X_DIM, Y_DIM], units="unknown", dtype=Float)
        self._pfld_kbcon1 = quantity_factory.zeros([X_DIM, Y_DIM], units="unknown", dtype=Float)
        self._k_mask = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Int,
        )
        for k in range(grid_indexing.domain[2]):
            self._k_mask.data[:, :, k] = k
        
        self._set_pfld_kbcon = stencil_factory.from_origin_domain(
            func=set_pfld_kbcon,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._stencil_update_kbcon1_cnvflg = stencil_factory.from_origin_domain(
            func=stencil_update_kbcon1_cnvflg,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._stencil_static9 = stencil_factory.from_origin_domain(
            func=stencil_static9,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
    
    def __call__(
        self,
        dbyo: FloatField,
        cnvflg: BoolFieldIJ,
        kmax: IntFieldIJ,
        kbm: IntFieldIJ,
        kbcon: IntFieldIJ,
        kbcon1: IntFieldIJ,
        flg: BoolFieldIJ,
        pfld: FloatField,
    ):
        self._set_pfld_kbcon(
            kbcon,
            self._k_mask,
            pfld,
            self._pfld_kbcon
        )
        self._stencil_update_kbcon1_cnvflg(
            dbyo,
            cnvflg,
            kmax,
            kbm,
            kbcon,
            kbcon1,
            flg,
            self._k_mask,
        )
        self._stencil_static9(
            cnvflg,
            pfld,
            self._pfld_kbcon,
            self._pfld_kbcon1,
            self._k_mask,
            kbcon1,
        )

class Static12:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        c1: Float,
        ncloud: Int,
    ):
        grid_indexing = stencil_factory.grid_indexing

        self._heo_kb = quantity_factory.zeros([X_DIM, Y_DIM], units="unknown", dtype=Float)
        self._k_mask = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Int,
        )
        for k in range(grid_indexing.domain[2]):
            self._k_mask.data[:, :, k] = k
        
        self._static12 = stencil_factory.from_origin_domain(
            func=stencil_static12,
            externals={"c1": c1, "ncloud": ncloud},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
    
    def __call__(
        self,
        cnvflg: BoolFieldIJ,
        aa1: FloatFieldIJ,
        flg: BoolFieldIJ,
        ktcon1: IntFieldIJ,
        kbm: IntFieldIJ,
        ktcon: IntFieldIJ,
        zo: FloatField,
        qeso: FloatField,
        to: FloatField,
        dbyo: FloatField,
        zi: FloatField,
        xlamue: FloatField,
        xlamud: FloatFieldIJ,
        qcko: FloatField,
        qrcko: FloatField,
        qo: FloatField,
        eta: FloatField,
        del0: FloatField,
        c0t: FloatField,
        pwo: FloatField,
        cnvwt: FloatField,
        buo: FloatField,
        wu2: FloatField,
        wc: FloatFieldIJ,
        sumx: FloatFieldIJ,
        kbcon1: IntFieldIJ,
        drag: FloatField,
        dellal: FloatField,
    ):
        self._static12(
            cnvflg,
            aa1,
            flg,
            ktcon1,
            kbm,
            self._k_mask,
            ktcon,
            zo,
            qeso,
            to,
            dbyo,
            zi,
            xlamue,
            xlamud,
            qcko,
            qrcko,
            qo,
            eta,
            del0,
            c0t,
            pwo,
            cnvwt,
            buo,
            wu2,
            wc,
            sumx,
            kbcon1,
            drag,
            dellal,
        )

class FeedbackCtrl:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        dt2: Float,
    ):
        grid_indexing = stencil_factory.grid_indexing

        self._ud_mf = quantity_factory.zeros(
                [X_DIM, Y_DIM, Z_DIM],
                units="unknown",
                dtype=Float,
            )
        self._dt_mf = quantity_factory.zeros(
                [X_DIM, Y_DIM, Z_DIM],
                units="unknown",
                dtype=Float,
            )
        self._k_mask = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Int,
        )
        for k in range(grid_indexing.domain[2]):
            self._k_mask.data[:, :, k] = k
        
        self._feedback_control_update_mass_flux = stencil_factory.from_origin_domain(
            func=feedback_control_update_mass_flux,
            externals={"dt2": dt2},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
    
    def __call__(
        self,
        cnvflg: BoolFieldIJ,
        kmax: IntFieldIJ,
        kb: IntFieldIJ,
        ktcon: IntFieldIJ,
        flg: BoolFieldIJ,
        islimsk: IntFieldIJ,
        ktop: IntFieldIJ,
        kbot: IntFieldIJ,
        kbcon: IntFieldIJ,
        kcnv: IntFieldIJ,
        qeso: FloatField,
        pfld: FloatField,
        delhbar: FloatFieldIJ,
        delqbar: FloatFieldIJ,
        deltbar: FloatFieldIJ,
        delubar: FloatFieldIJ,
        delvbar: FloatFieldIJ,
        qcond: FloatFieldIJ,
        dellah: FloatField,
        dellaq: FloatField,
        t1: FloatField,
        xmb: FloatFieldIJ,
        q1: FloatField,
        u1: FloatField,
        dellau: FloatField,
        v1: FloatField,
        dellav: FloatField,
        del0: FloatField,
        rntot: FloatFieldIJ,
        delqev: FloatFieldIJ,
        delq2: FloatFieldIJ,
        pwo: FloatField,
        deltv: FloatFieldIJ,
        delq: FloatFieldIJ,
        qevap: FloatFieldIJ,
        rn: FloatFieldIJ,
        edt: FloatFieldIJ,
        cnvw: FloatField,
        cnvwt: FloatField,
        cnvc: FloatField,
        eta: FloatField,
    ):
        self._feedback_control_update_mass_flux(
            cnvflg,
            self._k_mask,
            kmax,
            kb,
            ktcon,
            flg,
            islimsk,
            ktop,
            kbot,
            kbcon,
            kcnv,
            qeso,
            pfld,
            delhbar,
            delqbar,
            deltbar,
            delubar,
            delvbar,
            qcond,
            dellah,
            dellaq,
            t1,
            xmb,
            q1,
            u1,
            dellau,
            v1,
            dellav,
            del0,
            rntot,
            delqev,
            delq2,
            pwo,
            deltv,
            delq,
            qevap,
            rn,
            edt,
            cnvw,
            cnvwt,
            cnvc,
            self._ud_mf,
            self._dt_mf,
            eta,
        )

class TranslateStatic1(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "cnvflg": {"serialname": "sc1_cnvflg", "shield": True},
            "flg": {"serialname": "sc1_flg", "shield": True},
            "kbcon": {"serialname": "sc1_kbcon", "shield": True},
            "kmax": {"serialname": "sc1_kmax", "shield": True},
            "kbm": {"serialname": "sc1_kbm", "shield": True},
            "kb": {"serialname": "sc1_kb", "shield": True},
            "heo": {"serialname": "sc1_heo", "shield": True},
            "heso": {"serialname": "sc1_heso", "shield": True},
        }
        self.out_vars = {
            "cnvflg": {"serialname": "sc1_cnvflg", "shield": True},
            "flg": {"serialname": "sc1_flg", "shield": True},
            "kbcon": {"serialname": "sc1_kbcon", "shield": True},
            "kmax": {"serialname": "sc1_kmax", "shield": True},
            "kbm": {"serialname": "sc1_kbm", "shield": True},
            "kb": {"serialname": "sc1_kb", "shield": True},
            "heo": {"serialname": "sc1_heo", "shield": True},
            "heso": {"serialname": "sc1_heso", "shield": True},
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
        self.compute_func = Static1(
            self.stencil_factory,
            self.quantity_factory,
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)

class TranslateUpdateKb9(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "dbyo": {"serialname": "uk9_dbyo", "shield": True},
            "cnvflg": {"serialname": "uk9_cnvflg", "shield": True},
            "kmax": {"serialname": "uk9_kmax", "shield": True},
            "kbm": {"serialname": "uk9_kbm", "shield": True},
            "kbcon": {"serialname": "uk9_kbcon", "shield": True},
            "kbcon1": {"serialname": "uk9_kbcon1", "shield": True},
            "flg": {"serialname": "uk9_flg", "shield": True},
            "pfld": {"serialname": "uk9_pfld", "shield": True},
        }
        self.out_vars = {
            "dbyo": {"serialname": "uk9_dbyo", "shield": True},
            "cnvflg": {"serialname": "uk9_cnvflg", "shield": True},
            "kmax": {"serialname": "uk9_kmax", "shield": True},
            "kbm": {"serialname": "uk9_kbm", "shield": True},
            "kbcon": {"serialname": "uk9_kbcon", "shield": True},
            "kbcon1": {"serialname": "uk9_kbcon1", "shield": True},
            "flg": {"serialname": "uk9_flg", "shield": True},
            "pfld": {"serialname": "uk9_pfld", "shield": True},
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
        self.compute_func = UpdateKB9(
            self.stencil_factory,
            self.quantity_factory,
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)

class TranslateStatic12(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "cnvflg": {"serialname": "s12_cnvflg", "shield": True},
            "aa1": {"serialname": "s12_aa1", "shield": True},
            "flg": {"serialname": "s12_flg", "shield": True},
            "ktcon1": {"serialname": "s12_ktcon1", "shield": True},
            "kbm": {"serialname": "s12_kbm", "shield": True},
            "ktcon": {"serialname": "s12_ktcon", "shield": True},
            "zo": {"serialname": "s12_zo", "shield": True},
            "qeso": {"serialname": "s12_qeso", "shield": True},
            "to": {"serialname": "s12_to", "shield": True},
            "dbyo": {"serialname": "s12_dbyo", "shield": True},
            "zi": {"serialname": "s12_zi", "shield": True},
            "xlamue": {"serialname": "s12_xlamue", "shield": True},
            "xlamud": {"serialname": "s12_xlamud", "shield": True},
            "qcko": {"serialname": "s12_qcko", "shield": True},
            "qrcko": {"serialname": "s12_qrcko", "shield": True},
            "qo": {"serialname": "s12_qo", "shield": True},
            "eta": {"serialname": "s12_eta", "shield": True},
            "del0": {"serialname": "s12_del0", "shield": True},
            "c0t": {"serialname": "s12_c0t", "shield": True},
            "pwo": {"serialname": "s12_pwo", "shield": True},
            "cnvwt": {"serialname": "s12_cnvwt", "shield": True},
            "buo": {"serialname": "s12_buo", "shield": True},
            "wu2": {"serialname": "s12_wu2", "shield": True},
            "wc": {"serialname": "s12_wc", "shield": True},
            "sumx": {"serialname": "s12_sumx", "shield": True},
            "kbcon1": {"serialname": "s12_kbcon1", "shield": True},
            "drag": {"serialname": "s12_drag", "shield": True},
            "dellal": {"serialname": "s12_dellal", "shield": True},
        }
        self.in_vars["parameters"] = [
            "s12_c1",
            "s12_ncloud",
        ]
        self.out_vars = {
            "cnvflg": {"serialname": "s12_cnvflg", "shield": True},
            "aa1": {"serialname": "s12_aa1", "shield": True},
            "flg": {"serialname": "s12_flg", "shield": True},
            "ktcon1": {"serialname": "s12_ktcon1", "shield": True},
            "kbm": {"serialname": "s12_kbm", "shield": True},
            "ktcon": {"serialname": "s12_ktcon", "shield": True},
            "zo": {"serialname": "s12_zo", "shield": True},
            "qeso": {"serialname": "s12_qeso", "shield": True},
            "to": {"serialname": "s12_to", "shield": True},
            "dbyo": {"serialname": "s12_dbyo", "shield": True},
            "zi": {"serialname": "s12_zi", "shield": True},
            "xlamue": {"serialname": "s12_xlamue", "shield": True},
            "xlamud": {"serialname": "s12_xlamud", "shield": True},
            "qcko": {"serialname": "s12_qcko", "shield": True},
            "qrcko": {"serialname": "s12_qrcko", "shield": True},
            "qo": {"serialname": "s12_qo", "shield": True},
            "eta": {"serialname": "s12_eta", "shield": True},
            "del0": {"serialname": "s12_del0", "shield": True},
            "c0t": {"serialname": "s12_c0t", "shield": True},
            "pwo": {"serialname": "s12_pwo", "shield": True},
            "cnvwt": {"serialname": "s12_cnvwt", "shield": True},
            "buo": {"serialname": "s12_buo", "shield": True},
            "wu2": {"serialname": "s12_wu2", "shield": True},
            "wc": {"serialname": "s12_wc", "shield": True},
            "sumx": {"serialname": "s12_sumx", "shield": True},
            "kbcon1": {"serialname": "s12_kbcon1", "shield": True},
            "drag": {"serialname": "s12_drag", "shield": True},
            "dellal": {"serialname": "s12_dellal", "shield": True},
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
        self.compute_func = Static12(
            self.stencil_factory,
            self.quantity_factory,
            inputs.pop("s12_c1"),
            int(inputs.pop("s12_ncloud")),
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)

class TranslateFeedbackCtrl(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "cnvflg": {"serialname": "fc_cnvflg", "shield": True},
            "kmax": {"serialname": "fc_kmax", "shield": True},
            "kb": {"serialname": "fc_kb", "shield": True},
            "ktcon": {"serialname": "fc_ktcon", "shield": True},
            "flg": {"serialname": "fc_flg", "shield": True},
            "islimsk": {"serialname": "fc_islimsk", "shield": True},
            "ktop": {"serialname": "fc_ktop", "shield": True},
            "kbot": {"serialname": "fc_kbot", "shield": True},
            "kbcon": {"serialname": "fc_kbcon", "shield": True},
            "kcnv": {"serialname": "fc_kcnv", "shield": True},
            "qeso": {"serialname": "fc_qeso", "shield": True},
            "pfld": {"serialname": "fc_pfld", "shield": True},
            "delhbar": {"serialname": "fc_delhbar", "shield": True},
            "delqbar": {"serialname": "fc_delqbar", "shield": True},
            "deltbar": {"serialname": "fc_deltbar", "shield": True},
            "delubar": {"serialname": "fc_delubar", "shield": True},
            "delvbar": {"serialname": "fc_delvbar", "shield": True},
            "qcond": {"serialname": "fc_qcond", "shield": True},
            "dellah": {"serialname": "fc_dellah", "shield": True},
            "dellaq": {"serialname": "fc_dellaq", "shield": True},
            "t1": {"serialname": "fc_t1", "shield": True},
            "xmb": {"serialname": "fc_xmb", "shield": True},
            "q1": {"serialname": "fc_q1", "shield": True},
            "u1": {"serialname": "fc_u1", "shield": True},
            "dellau": {"serialname": "fc_dellau", "shield": True},
            "v1": {"serialname": "fc_v1", "shield": True},
            "dellav": {"serialname": "fc_dellav", "shield": True},
            "del0": {"serialname": "fc_del", "shield": True},
            "rntot": {"serialname": "fc_rntot", "shield": True},
            "delqev": {"serialname": "fc_delqev", "shield": True},
            "delq2": {"serialname": "fc_delq2", "shield": True},
            "pwo": {"serialname": "fc_pwo", "shield": True},
            "deltv": {"serialname": "fc_deltv", "shield": True},
            "delq": {"serialname": "fc_delq", "shield": True},
            "qevap": {"serialname": "fc_qevap", "shield": True},
            "rn": {"serialname": "fc_rn", "shield": True},
            "edt": {"serialname": "fc_edt", "shield": True},
            "cnvw": {"serialname": "fc_cnvw", "shield": True},
            "cnvwt": {"serialname": "fc_cnvwt", "shield": True},
            "cnvc": {"serialname": "fc_cnvc", "shield": True},
            "eta": {"serialname": "fc_eta", "shield": True},
        }
        self.in_vars["parameters"] = [
            "fc_dt2",
        ]
        self.out_vars = {
            "cnvflg": {"serialname": "s12_cnvflg", "shield": True},
            "aa1": {"serialname": "s12_aa1", "shield": True},
            "flg": {"serialname": "s12_flg", "shield": True},
            "ktcon1": {"serialname": "s12_ktcon1", "shield": True},
            "kbm": {"serialname": "s12_kbm", "shield": True},
            "ktcon": {"serialname": "s12_ktcon", "shield": True},
            "zo": {"serialname": "s12_zo", "shield": True},
            "qeso": {"serialname": "s12_qeso", "shield": True},
            "to": {"serialname": "s12_to", "shield": True},
            "dbyo": {"serialname": "s12_dbyo", "shield": True},
            "zi": {"serialname": "s12_zi", "shield": True},
            "xlamue": {"serialname": "s12_xlamue", "shield": True},
            "xlamud": {"serialname": "s12_xlamud", "shield": True},
            "qcko": {"serialname": "s12_qcko", "shield": True},
            "qrcko": {"serialname": "s12_qrcko", "shield": True},
            "qo": {"serialname": "s12_qo", "shield": True},
            "eta": {"serialname": "s12_eta", "shield": True},
            "del0": {"serialname": "s12_del0", "shield": True},
            "c0t": {"serialname": "s12_c0t", "shield": True},
            "pwo": {"serialname": "s12_pwo", "shield": True},
            "cnvwt": {"serialname": "s12_cnvwt", "shield": True},
            "buo": {"serialname": "s12_buo", "shield": True},
            "wu2": {"serialname": "s12_wu2", "shield": True},
            "wc": {"serialname": "s12_wc", "shield": True},
            "sumx": {"serialname": "s12_sumx", "shield": True},
            "kbcon1": {"serialname": "s12_kbcon1", "shield": True},
            "drag": {"serialname": "s12_drag", "shield": True},
            "dellal": {"serialname": "s12_dellal", "shield": True},
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
        self.compute_func = FeedbackCtrl(
            self.stencil_factory,
            self.quantity_factory,
            inputs.pop("fc_dt2"),
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)
