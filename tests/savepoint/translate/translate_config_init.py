from ndsl.dsl.stencil import StencilFactory
from ndsl.namelist import Namelist
from pyshield import PhysicsConfig
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class PassVars:
    def __init__(self, config):
        self.config = config
        self.l1 = len(config.acc)
        self.l2 = len(config.acco[1])

    def __call__(
        self,
        c_air,
        c_vap,
        d0_vap,
        lv00,
        li00,
        li20,
        d1_vap,
        d1_ice,
        c1_vap,
        c1_liq,
        c1_ice,
        t_wfr,
        pcaw,
        pcbw,
        pcai,
        pcbi,
        pcar,
        pcbr,
        pcas,
        pcbs,
        pcag,
        pcbg,
        pcah,
        pcbh,
        edaw,
        edbw,
        edai,
        edbi,
        edar,
        edbr,
        edas,
        edbs,
        edag,
        edbg,
        edah,
        edbh,
        oeaw,
        oebw,
        oeai,
        oebi,
        oear,
        oebr,
        oeas,
        oebs,
        oeag,
        oebg,
        oeah,
        oebh,
        rraw,
        rrbw,
        rrai,
        rrbi,
        rrar,
        rrbr,
        rras,
        rrbs,
        rrag,
        rrbg,
        rrah,
        rrbh,
        tvai,
        tvbi,
        tvar,
        tvbr,
        tvas,
        tvbs,
        tvag,
        tvbg,
        tvah,
        tvbh,
        crevp_1,
        crevp_2,
        crevp_3,
        crevp_4,
        crevp_5,
        cssub_1,
        cssub_2,
        cssub_3,
        cssub_4,
        cssub_5,
        cgsub_1,
        cgsub_2,
        cgsub_3,
        cgsub_4,
        cgsub_5,
        csmlt_1,
        csmlt_2,
        csmlt_3,
        csmlt_4,
        cgmlt_1,
        cgmlt_2,
        cgmlt_3,
        cgmlt_4,
        cgfr_1,
        cgfr_2,
        normw,
        normr,
        normi,
        norms,
        normg,
        expow,
        expor,
        expoi,
        expos,
        expog,
        cracw,
        craci,
        csacw,
        csaci,
        cgacw,
        cgaci,
        cracs,
        csacr,
        cgacr,
        cgacs,
        acc,
        acco1,
        acco2,
        acco3,
    ):
        c_air[:] = self.config.c_air
        c_vap[:] = self.config.c_vap
        d0_vap[:] = self.config.d0_vap
        lv00[:] = self.config.lv00
        li00[:] = self.config.li00
        li20[:] = self.config.li20
        d1_vap[:] = self.config.d1_vap
        d1_ice[:] = self.config.d1_ice
        c1_vap[:] = self.config.c1_vap
        c1_liq[:] = self.config.c1_liq
        c1_ice[:] = self.config.c1_ice
        t_wfr[:] = self.config.t_wfr
        pcaw[:] = self.config.pcaw
        pcbw[:] = self.config.pcbw
        pcai[:] = self.config.pcai
        pcbi[:] = self.config.pcbi
        pcar[:] = self.config.pcar
        pcbr[:] = self.config.pcbr
        pcas[:] = self.config.pcas
        pcbs[:] = self.config.pcbs
        pcag[:] = self.config.pcag
        pcbg[:] = self.config.pcbg
        pcah[:] = self.config.pcah
        pcbh[:] = self.config.pcbh
        edaw[:] = self.config.edaw
        edbw[:] = self.config.edbw
        edai[:] = self.config.edai
        edbi[:] = self.config.edbi
        edar[:] = self.config.edar
        edbr[:] = self.config.edbr
        edas[:] = self.config.edas
        edbs[:] = self.config.edbs
        edag[:] = self.config.edag
        edbg[:] = self.config.edbg
        edah[:] = self.config.edah
        edbh[:] = self.config.edbh
        oeaw[:] = self.config.oeaw
        oebw[:] = self.config.oebw
        oeai[:] = self.config.oeai
        oebi[:] = self.config.oebi
        oear[:] = self.config.oear
        oebr[:] = self.config.oebr
        oeas[:] = self.config.oeas
        oebs[:] = self.config.oebs
        oeag[:] = self.config.oeag
        oebg[:] = self.config.oebg
        oeah[:] = self.config.oeah
        oebh[:] = self.config.oebh
        rraw[:] = self.config.rraw
        rrbw[:] = self.config.rrbw
        rrai[:] = self.config.rrai
        rrbi[:] = self.config.rrbi
        rrar[:] = self.config.rrar
        rrbr[:] = self.config.rrbr
        rras[:] = self.config.rras
        rrbs[:] = self.config.rrbs
        rrag[:] = self.config.rrag
        rrbg[:] = self.config.rrbg
        rrah[:] = self.config.rrah
        rrbh[:] = self.config.rrbh
        tvai[:] = self.config.tvai
        tvbi[:] = self.config.tvbi
        tvar[:] = self.config.tvar
        tvbr[:] = self.config.tvbr
        tvas[:] = self.config.tvas
        tvbs[:] = self.config.tvbs
        tvag[:] = self.config.tvag
        tvbg[:] = self.config.tvbg
        tvah[:] = self.config.tvah
        tvbh[:] = self.config.tvbh
        crevp_1[:] = self.config.crevp_1
        crevp_2[:] = self.config.crevp_2
        crevp_3[:] = self.config.crevp_3
        crevp_4[:] = self.config.crevp_4
        crevp_5[:] = self.config.crevp_5
        cssub_1[:] = self.config.cssub_1
        cssub_2[:] = self.config.cssub_2
        cssub_3[:] = self.config.cssub_3
        cssub_4[:] = self.config.cssub_4
        cssub_5[:] = self.config.cssub_5
        cgsub_1[:] = self.config.cgsub_1
        cgsub_2[:] = self.config.cgsub_2
        cgsub_3[:] = self.config.cgsub_3
        cgsub_4[:] = self.config.cgsub_4
        cgsub_5[:] = self.config.cgsub_5
        csmlt_1[:] = self.config.csmlt_1
        csmlt_2[:] = self.config.csmlt_2
        csmlt_3[:] = self.config.csmlt_3
        csmlt_4[:] = self.config.csmlt_4
        cgmlt_1[:] = self.config.cgmlt_1
        cgmlt_2[:] = self.config.cgmlt_2
        cgmlt_3[:] = self.config.cgmlt_3
        cgmlt_4[:] = self.config.cgmlt_4
        cgfr_1[:] = self.config.cgfr_1
        cgfr_2[:] = self.config.cgfr_2
        normw[:] = self.config.normw
        normr[:] = self.config.normr
        normi[:] = self.config.normi
        norms[:] = self.config.norms
        normg[:] = self.config.normg
        expow[:] = self.config.expow
        expor[:] = self.config.expor
        expoi[:] = self.config.expoi
        expos[:] = self.config.expos
        expog[:] = self.config.expog
        cracw[:] = self.config.cracw
        craci[:] = self.config.craci
        csacw[:] = self.config.csacw
        csaci[:] = self.config.csaci
        cgacw[:] = self.config.cgacw
        cgaci[:] = self.config.cgaci
        cracs[:] = self.config.cracs
        csacr[:] = self.config.csacr
        cgacr[:] = self.config.cgacr
        cgacs[:] = self.config.cgacs
        acc[:, :, : self.l1] = self.config.acc
        acco1[:, :, : self.l2] = self.config.acco[0]
        acco2[:, :, : self.l2] = self.config.acco[1]
        acco3[:, :, : self.l2] = self.config.acco[2]


class TranslateConfigInit(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "c_air": {"serialname": "ini_c_air", "mp3": True},
            "c_vap": {"serialname": "ini_c_vap", "mp3": True},
            "d0_vap": {"serialname": "ini_d0_vap", "mp3": True},
            "lv00": {"serialname": "ini_lv00", "mp3": True},
            "li00": {"serialname": "ini_li00", "mp3": True},
            "li20": {"serialname": "ini_li20", "mp3": True},
            "d1_vap": {"serialname": "ini_d1_vap", "mp3": True},
            "d1_ice": {"serialname": "ini_d1_ice", "mp3": True},
            "c1_vap": {"serialname": "ini_c1_vap", "mp3": True},
            "c1_liq": {"serialname": "ini_c1_liq", "mp3": True},
            "c1_ice": {"serialname": "ini_c1_ice", "mp3": True},
            "t_wfr": {"serialname": "ini_t_wfr", "mp3": True},
            "pcaw": {"serialname": "ini_pcaw", "mp3": True},
            "pcbw": {"serialname": "ini_pcbw", "mp3": True},
            "pcai": {"serialname": "ini_pcai", "mp3": True},
            "pcbi": {"serialname": "ini_pcbi", "mp3": True},
            "pcar": {"serialname": "ini_pcar", "mp3": True},
            "pcbr": {"serialname": "ini_pcbr", "mp3": True},
            "pcas": {"serialname": "ini_pcas", "mp3": True},
            "pcbs": {"serialname": "ini_pcbs", "mp3": True},
            "pcag": {"serialname": "ini_pcag", "mp3": True},
            "pcbg": {"serialname": "ini_pcbg", "mp3": True},
            "pcah": {"serialname": "ini_pcah", "mp3": True},
            "pcbh": {"serialname": "ini_pcbh", "mp3": True},
            "edaw": {"serialname": "ini_edaw", "mp3": True},
            "edbw": {"serialname": "ini_edbw", "mp3": True},
            "edai": {"serialname": "ini_edai", "mp3": True},
            "edbi": {"serialname": "ini_edbi", "mp3": True},
            "edar": {"serialname": "ini_edar", "mp3": True},
            "edbr": {"serialname": "ini_edbr", "mp3": True},
            "edas": {"serialname": "ini_edas", "mp3": True},
            "edbs": {"serialname": "ini_edbs", "mp3": True},
            "edag": {"serialname": "ini_edag", "mp3": True},
            "edbg": {"serialname": "ini_edbg", "mp3": True},
            "edah": {"serialname": "ini_edah", "mp3": True},
            "edbh": {"serialname": "ini_edbh", "mp3": True},
            "oeaw": {"serialname": "ini_oeaw", "mp3": True},
            "oebw": {"serialname": "ini_oebw", "mp3": True},
            "oeai": {"serialname": "ini_oeai", "mp3": True},
            "oebi": {"serialname": "ini_oebi", "mp3": True},
            "oear": {"serialname": "ini_oear", "mp3": True},
            "oebr": {"serialname": "ini_oebr", "mp3": True},
            "oeas": {"serialname": "ini_oeas", "mp3": True},
            "oebs": {"serialname": "ini_oebs", "mp3": True},
            "oeag": {"serialname": "ini_oeag", "mp3": True},
            "oebg": {"serialname": "ini_oebg", "mp3": True},
            "oeah": {"serialname": "ini_oeah", "mp3": True},
            "oebh": {"serialname": "ini_oebh", "mp3": True},
            "rraw": {"serialname": "ini_rraw", "mp3": True},
            "rrbw": {"serialname": "ini_rrbw", "mp3": True},
            "rrai": {"serialname": "ini_rrai", "mp3": True},
            "rrbi": {"serialname": "ini_rrbi", "mp3": True},
            "rrar": {"serialname": "ini_rrar", "mp3": True},
            "rrbr": {"serialname": "ini_rrbr", "mp3": True},
            "rras": {"serialname": "ini_rras", "mp3": True},
            "rrbs": {"serialname": "ini_rrbs", "mp3": True},
            "rrag": {"serialname": "ini_rrag", "mp3": True},
            "rrbg": {"serialname": "ini_rrbg", "mp3": True},
            "rrah": {"serialname": "ini_rrah", "mp3": True},
            "rrbh": {"serialname": "ini_rrbh", "mp3": True},
            "tvai": {"serialname": "ini_tvai", "mp3": True},
            "tvbi": {"serialname": "ini_tvbi", "mp3": True},
            "tvar": {"serialname": "ini_tvar", "mp3": True},
            "tvbr": {"serialname": "ini_tvbr", "mp3": True},
            "tvas": {"serialname": "ini_tvas", "mp3": True},
            "tvbs": {"serialname": "ini_tvbs", "mp3": True},
            "tvag": {"serialname": "ini_tvag", "mp3": True},
            "tvbg": {"serialname": "ini_tvbg", "mp3": True},
            "tvah": {"serialname": "ini_tvah", "mp3": True},
            "tvbh": {"serialname": "ini_tvbh", "mp3": True},
            "crevp_1": {"serialname": "ini_crevp_1", "mp3": True},
            "crevp_2": {"serialname": "ini_crevp_2", "mp3": True},
            "crevp_3": {"serialname": "ini_crevp_3", "mp3": True},
            "crevp_4": {"serialname": "ini_crevp_4", "mp3": True},
            "crevp_5": {"serialname": "ini_crevp_5", "mp3": True},
            "cssub_1": {"serialname": "ini_cssub_1", "mp3": True},
            "cssub_2": {"serialname": "ini_cssub_2", "mp3": True},
            "cssub_3": {"serialname": "ini_cssub_3", "mp3": True},
            "cssub_4": {"serialname": "ini_cssub_4", "mp3": True},
            "cssub_5": {"serialname": "ini_cssub_5", "mp3": True},
            "cgsub_1": {"serialname": "ini_cgsub_1", "mp3": True},
            "cgsub_2": {"serialname": "ini_cgsub_2", "mp3": True},
            "cgsub_3": {"serialname": "ini_cgsub_3", "mp3": True},
            "cgsub_4": {"serialname": "ini_cgsub_4", "mp3": True},
            "cgsub_5": {"serialname": "ini_cgsub_5", "mp3": True},
            "csmlt_1": {"serialname": "ini_csmlt_1", "mp3": True},
            "csmlt_2": {"serialname": "ini_csmlt_2", "mp3": True},
            "csmlt_3": {"serialname": "ini_csmlt_3", "mp3": True},
            "csmlt_4": {"serialname": "ini_csmlt_4", "mp3": True},
            "cgmlt_1": {"serialname": "ini_cgmlt_1", "mp3": True},
            "cgmlt_2": {"serialname": "ini_cgmlt_2", "mp3": True},
            "cgmlt_3": {"serialname": "ini_cgmlt_3", "mp3": True},
            "cgmlt_4": {"serialname": "ini_cgmlt_4", "mp3": True},
            "cgfr_1": {"serialname": "ini_cgfr_1", "mp3": True},
            "cgfr_2": {"serialname": "ini_cgfr_2", "mp3": True},
            "normw": {"serialname": "ini_normw", "mp3": True},
            "normr": {"serialname": "ini_normr", "mp3": True},
            "normi": {"serialname": "ini_normi", "mp3": True},
            "norms": {"serialname": "ini_norms", "mp3": True},
            "normg": {"serialname": "ini_normg", "mp3": True},
            "expow": {"serialname": "ini_expow", "mp3": True},
            "expor": {"serialname": "ini_expor", "mp3": True},
            "expoi": {"serialname": "ini_expoi", "mp3": True},
            "expos": {"serialname": "ini_expos", "mp3": True},
            "expog": {"serialname": "ini_expog", "mp3": True},
            "cracw": {"serialname": "ini_cracw", "mp3": True},
            "craci": {"serialname": "ini_craci", "mp3": True},
            "csacw": {"serialname": "ini_csacw", "mp3": True},
            "csaci": {"serialname": "ini_csaci", "mp3": True},
            "cgacw": {"serialname": "ini_cgacw", "mp3": True},
            "cgaci": {"serialname": "ini_cgaci", "mp3": True},
            "cracs": {"serialname": "ini_cracs", "mp3": True},
            "csacr": {"serialname": "ini_csacr", "mp3": True},
            "cgacr": {"serialname": "ini_cgacr", "mp3": True},
            "cgacs": {"serialname": "ini_cgacs", "mp3": True},
            "acc": {"serialname": "ini_acc", "mp3": True},
            "acco1": {"serialname": "ini_acco1", "mp3": True},
            "acco2": {"serialname": "ini_acco2", "mp3": True},
            "acco3": {"serialname": "ini_acco3", "mp3": True},
        }

        self.out_vars = {
            "c_air": {"serialname": "ini_c_air", "mp3": True},
            "c_vap": {"serialname": "ini_c_vap", "mp3": True},
            "d0_vap": {"serialname": "ini_d0_vap", "mp3": True},
            "lv00": {"serialname": "ini_lv00", "mp3": True},
            "li00": {"serialname": "ini_li00", "mp3": True},
            "li20": {"serialname": "ini_li20", "mp3": True},
            "d1_vap": {"serialname": "ini_d1_vap", "mp3": True},
            "d1_ice": {"serialname": "ini_d1_ice", "mp3": True},
            "c1_vap": {"serialname": "ini_c1_vap", "mp3": True},
            "c1_liq": {"serialname": "ini_c1_liq", "mp3": True},
            "c1_ice": {"serialname": "ini_c1_ice", "mp3": True},
            "t_wfr": {"serialname": "ini_t_wfr", "mp3": True},
            "pcaw": {"serialname": "ini_pcaw", "mp3": True},
            "pcbw": {"serialname": "ini_pcbw", "mp3": True},
            "pcai": {"serialname": "ini_pcai", "mp3": True},
            "pcbi": {"serialname": "ini_pcbi", "mp3": True},
            "pcar": {"serialname": "ini_pcar", "mp3": True},
            "pcbr": {"serialname": "ini_pcbr", "mp3": True},
            "pcas": {"serialname": "ini_pcas", "mp3": True},
            "pcbs": {"serialname": "ini_pcbs", "mp3": True},
            "pcag": {"serialname": "ini_pcag", "mp3": True},
            "pcbg": {"serialname": "ini_pcbg", "mp3": True},
            "pcah": {"serialname": "ini_pcah", "mp3": True},
            "pcbh": {"serialname": "ini_pcbh", "mp3": True},
            "edaw": {"serialname": "ini_edaw", "mp3": True},
            "edbw": {"serialname": "ini_edbw", "mp3": True},
            "edai": {"serialname": "ini_edai", "mp3": True},
            "edbi": {"serialname": "ini_edbi", "mp3": True},
            "edar": {"serialname": "ini_edar", "mp3": True},
            "edbr": {"serialname": "ini_edbr", "mp3": True},
            "edas": {"serialname": "ini_edas", "mp3": True},
            "edbs": {"serialname": "ini_edbs", "mp3": True},
            "edag": {"serialname": "ini_edag", "mp3": True},
            "edbg": {"serialname": "ini_edbg", "mp3": True},
            "edah": {"serialname": "ini_edah", "mp3": True},
            "edbh": {"serialname": "ini_edbh", "mp3": True},
            "oeaw": {"serialname": "ini_oeaw", "mp3": True},
            "oebw": {"serialname": "ini_oebw", "mp3": True},
            "oeai": {"serialname": "ini_oeai", "mp3": True},
            "oebi": {"serialname": "ini_oebi", "mp3": True},
            "oear": {"serialname": "ini_oear", "mp3": True},
            "oebr": {"serialname": "ini_oebr", "mp3": True},
            "oeas": {"serialname": "ini_oeas", "mp3": True},
            "oebs": {"serialname": "ini_oebs", "mp3": True},
            "oeag": {"serialname": "ini_oeag", "mp3": True},
            "oebg": {"serialname": "ini_oebg", "mp3": True},
            "oeah": {"serialname": "ini_oeah", "mp3": True},
            "oebh": {"serialname": "ini_oebh", "mp3": True},
            "rraw": {"serialname": "ini_rraw", "mp3": True},
            "rrbw": {"serialname": "ini_rrbw", "mp3": True},
            "rrai": {"serialname": "ini_rrai", "mp3": True},
            "rrbi": {"serialname": "ini_rrbi", "mp3": True},
            "rrar": {"serialname": "ini_rrar", "mp3": True},
            "rrbr": {"serialname": "ini_rrbr", "mp3": True},
            "rras": {"serialname": "ini_rras", "mp3": True},
            "rrbs": {"serialname": "ini_rrbs", "mp3": True},
            "rrag": {"serialname": "ini_rrag", "mp3": True},
            "rrbg": {"serialname": "ini_rrbg", "mp3": True},
            "rrah": {"serialname": "ini_rrah", "mp3": True},
            "rrbh": {"serialname": "ini_rrbh", "mp3": True},
            "tvai": {"serialname": "ini_tvai", "mp3": True},
            "tvbi": {"serialname": "ini_tvbi", "mp3": True},
            "tvar": {"serialname": "ini_tvar", "mp3": True},
            "tvbr": {"serialname": "ini_tvbr", "mp3": True},
            "tvas": {"serialname": "ini_tvas", "mp3": True},
            "tvbs": {"serialname": "ini_tvbs", "mp3": True},
            "tvag": {"serialname": "ini_tvag", "mp3": True},
            "tvbg": {"serialname": "ini_tvbg", "mp3": True},
            "tvah": {"serialname": "ini_tvah", "mp3": True},
            "tvbh": {"serialname": "ini_tvbh", "mp3": True},
            "crevp_1": {"serialname": "ini_crevp_1", "mp3": True},
            "crevp_2": {"serialname": "ini_crevp_2", "mp3": True},
            "crevp_3": {"serialname": "ini_crevp_3", "mp3": True},
            "crevp_4": {"serialname": "ini_crevp_4", "mp3": True},
            "crevp_5": {"serialname": "ini_crevp_5", "mp3": True},
            "cssub_1": {"serialname": "ini_cssub_1", "mp3": True},
            "cssub_2": {"serialname": "ini_cssub_2", "mp3": True},
            "cssub_3": {"serialname": "ini_cssub_3", "mp3": True},
            "cssub_4": {"serialname": "ini_cssub_4", "mp3": True},
            "cssub_5": {"serialname": "ini_cssub_5", "mp3": True},
            "cgsub_1": {"serialname": "ini_cgsub_1", "mp3": True},
            "cgsub_2": {"serialname": "ini_cgsub_2", "mp3": True},
            "cgsub_3": {"serialname": "ini_cgsub_3", "mp3": True},
            "cgsub_4": {"serialname": "ini_cgsub_4", "mp3": True},
            "cgsub_5": {"serialname": "ini_cgsub_5", "mp3": True},
            "csmlt_1": {"serialname": "ini_csmlt_1", "mp3": True},
            "csmlt_2": {"serialname": "ini_csmlt_2", "mp3": True},
            "csmlt_3": {"serialname": "ini_csmlt_3", "mp3": True},
            "csmlt_4": {"serialname": "ini_csmlt_4", "mp3": True},
            "cgmlt_1": {"serialname": "ini_cgmlt_1", "mp3": True},
            "cgmlt_2": {"serialname": "ini_cgmlt_2", "mp3": True},
            "cgmlt_3": {"serialname": "ini_cgmlt_3", "mp3": True},
            "cgmlt_4": {"serialname": "ini_cgmlt_4", "mp3": True},
            "cgfr_1": {"serialname": "ini_cgfr_1", "mp3": True},
            "cgfr_2": {"serialname": "ini_cgfr_2", "mp3": True},
            "normw": {"serialname": "ini_normw", "mp3": True},
            "normr": {"serialname": "ini_normr", "mp3": True},
            "normi": {"serialname": "ini_normi", "mp3": True},
            "norms": {"serialname": "ini_norms", "mp3": True},
            "normg": {"serialname": "ini_normg", "mp3": True},
            "expow": {"serialname": "ini_expow", "mp3": True},
            "expor": {"serialname": "ini_expor", "mp3": True},
            "expoi": {"serialname": "ini_expoi", "mp3": True},
            "expos": {"serialname": "ini_expos", "mp3": True},
            "expog": {"serialname": "ini_expog", "mp3": True},
            "cracw": {"serialname": "ini_cracw", "mp3": True},
            "craci": {"serialname": "ini_craci", "mp3": True},
            "csacw": {"serialname": "ini_csacw", "mp3": True},
            "csaci": {"serialname": "ini_csaci", "mp3": True},
            "cgacw": {"serialname": "ini_cgacw", "mp3": True},
            "cgaci": {"serialname": "ini_cgaci", "mp3": True},
            "cracs": {"serialname": "ini_cracs", "mp3": True},
            "csacr": {"serialname": "ini_csacr", "mp3": True},
            "cgacr": {"serialname": "ini_cgacr", "mp3": True},
            "cgacs": {"serialname": "ini_cgacs", "mp3": True},
            "acc": {"serialname": "ini_acc", "kend": 20, "mp3": True},
            "acco1": {"serialname": "ini_acco1", "kend": 10, "mp3": True},
            "acco2": {"serialname": "ini_acco2", "kend": 10, "mp3": True},
            "acco3": {"serialname": "ini_acco3", "kend": 10, "mp3": True},
        }

        pconf = PhysicsConfig.from_namelist(namelist)
        self.config = pconf.microphysics

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)

        compute_func = PassVars(
            self.config,
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
