from gt4py.cartesian.gtscript import (
    FORWARD,
    BACKWARD,
    PARALLEL,
    computation,
    exp,
    interval,
    sqrt,
    log10,
)

import ndsl.constants as constants
import pySHiELD.constants as physcons
from pySHiELD.functions.physics_functions import fpvs
from pySHiELD._config import PhysicsConfig

# from pace.dsl.dace.orchestration import orchestrate
from ndsl import StencilFactory, QuantityFactory
from ndsl.dsl.typing import (
    Bool,
    BoolFieldIJ,
    BoolField,
    Float,
    FloatFieldIJ,
    FloatField,
    Int,
    IntFieldIJ,
    IntField,
)
from ndsl.stencils.basic_operations import sign


def pa_to_cb(
    psp: FloatField,
    prslp: FloatField,
    delp: FloatField,
    ps: FloatField,
    prsl: FloatField,
    del0: FloatField,
):
    with computation(PARALLEL), interval(...):
        # Convert input Pa terms to Cb terms
        ps = psp * 0.001
        prsl = prslp * 0.001
        del0 = delp * 0.001


def init_col_arr(
    kcnv: IntField,
    cnvflg: BoolField,
    kbot: IntField,
    ktop: IntField,
    kbcon: IntField,
    kb: IntField,
    rn: FloatField,
    gdx: FloatField,
    garea: FloatField,
    km: Int
):
    with computation(PARALLEL), interval(...):
        # Initialize column-integrated and other single-value-per-column
        # variable arrays
        if kcnv == 1:
            cnvflg = False

        if cnvflg:
            kbot = km + 1
            ktop = 0

        rn = 0.0
        kbcon = km
        kb = km
        gdx = sqrt(garea)


def init_par_and_arr(
    islimsk: IntField,
    c0: FloatField,
    t1: FloatField,
    c0t: FloatField,
    cnvw: FloatField,
    cnvc: FloatField,
    ud_mf: FloatField,
    dt_mf: FloatField,
    c0s: Float,
    asolfac: Float,
    d0: Float
):
    with computation(PARALLEL), interval(...):
        # Determine aerosol-aware rain conversion parameter over land
        if islimsk == 1:
            c0 = c0s * asolfac
        else:
            c0 = c0s

        # Determine rain conversion parameter above the freezing level
        # which exponentially decreases with decreasing temperature
        # from Han et al.'s (2017) \cite han_et_al_2017 equation 8
        tem = exp(d0 * (t1 - 273.16))
        if t1 > 273.16:
            c0t = c0
        else:
            c0t = c0 * tem

        # Initialize convective cloud water and cloud cover to zero
        cnvw = 0.0
        cnvc = 0.0

        # Initialize updraft mass fluxes to zero
        ud_mf = 0.0
        dt_mf = 0.0


def init_kbm_kmax(
    kbm: IntField,
    k_idx: IntField,
    kmax: IntField,
    state_buf1: BoolField,
    state_buf2: BoolField,
    tx1: FloatField,
    ps: FloatField,
    prsl: FloatField,
    km: Int
):
    with computation(FORWARD):
        # Determine maximum indices for the parcel starting point (kbm)
        # and cloud top (kmax)
        with interval(0, 1):
            tx1 = 1.0 / ps

            if prsl * tx1 > 0.7:
                kbm = k_idx + 1
                state_buf1 = 1
            else:
                kbm = km
                state_buf1 = 0  # means kbm is set to default `km`

        with interval(1, None):
            tx1 = 1.0 / ps

            if prsl * tx1 > 0.7:
                kbm = k_idx + 1
                state_buf1 = 1
            elif state_buf1[0, 0, -1]:
                kbm = kbm[0, 0, -1]
                state_buf1 = 1
            else:
                kbm = km
                state_buf1 = 0

    with computation(FORWARD):
        with interval(0, 1):
            if prsl * tx1 > 0.6:
                kmax = k_idx + 1
                state_buf2 = 1  # reuse flg
            else:
                kmax = km
                state_buf2 = 0  # means kmax is set to default `km`

        with interval(1, None):
            if prsl * tx1 > 0.6:
                kmax = k_idx + 1
                state_buf2 = 1
            elif state_buf2[0, 0, -1]:
                kmax = kmax[0, 0, -1]
                state_buf2 = 1
            else:
                kmax = km
                state_buf2 = 0

    with computation(BACKWARD):
        with interval(-1, None):
            kbm = min(kbm, kmax)

        with interval(0, -1):
            kbm = kbm[0, 0, 1]
            kmax = kmax[0, 0, 1]
            kbm = min(kbm, kmax)


def init_final(
    kbm: IntField,
    k_idx: IntField,
    kmax: IntField,
    flg: BoolField,
    cnvflg: BoolField,
    kpbl: IntField,
    tx1: FloatField,
    ps: FloatField,
    prsl: FloatField,
    zo: FloatField,
    phil: FloatField,
    zi: FloatField,
    pfld: FloatField,
    eta: FloatField,
    hcko: FloatField,
    qcko: FloatField,
    qrcko: FloatField,
    ucko: FloatField,
    vcko: FloatField,
    dbyo: FloatField,
    pwo: FloatField,
    dellal: FloatField,
    to: FloatField,
    qo: FloatField,
    uo: FloatField,
    vo: FloatField,
    wu2: FloatField,
    buo: FloatField,
    drag: FloatField,
    cnvwt: FloatField,
    qeso: FloatField,
    heo: FloatField,
    heso: FloatField,
    hpbl: FloatField,
    t1: FloatField,
    q1: FloatField,
    u1: FloatField,
    v1: FloatField,
    km: Int
):
    with computation(FORWARD), interval(...):
        # Calculate hydrostatic height at layer centers assuming a flat
        # surface (no terrain) from the geopotential
        zo = phil / constants.GRAV

        # Initialize flg in parallel computation block
        flg = cnvflg

        kpbl = 1

    with computation(PARALLEL), interval(0, -1):
        # Calculate interface height
        zi = 0.5 * (zo[0, 0, 0] + zo[0, 0, +1])

    with computation(FORWARD), interval(1, -1):
        # Find the index for the PBL top using the PBL height; enforce
        # that it is lower than the maximum parcel starting level
        flg = flg[0, 0, -1]
        kpbl = kpbl[0, 0, -1]
        if flg and (zo <= hpbl):
            kpbl = k_idx
        else:
            flg = False

    with computation(FORWARD), interval(-1, None):
        flg = flg[0, 0, -1]
        kpbl = kpbl[0, 0, -1]

    with computation(BACKWARD), interval(0, -1):
        # Propagate results back to update whole field
        kpbl = kpbl[0, 0, 1]
        flg = flg[0, 0, 1]

    with computation(PARALLEL), interval(...):
        kpbl = min(kpbl, kbm)

        # Temporary var have to be defined outside of if-clause
        val1 = 0.0
        val2 = 0.0
        tem = 0.0
        fpvsto = fpvs(t1)  # fpvs(to) and to = t1

        if cnvflg and k_idx <= kmax:

            # Convert prsl from centibar to millibar, set normalized mass
            # flux to 1, cloud properties to 0, and save model state
            # variables (after advection/turbulence)
            pfld = prsl * 10.0
            eta = 1.0
            hcko = 0.0
            qcko = 0.0
            qrcko = 0.0
            ucko = 0.0
            vcko = 0.0
            dbyo = 0.0
            pwo = 0.0
            dellal = 0.0
            to = t1
            qo = q1
            uo = u1
            vo = v1
            wu2 = 0.0
            buo = 0.0
            drag = 0.0
            cnvwt = 0.0

            # Calculate saturation specific humidity and enforce minimum
            # moisture values
            qeso = 0.01 * fpvsto
            qeso = (constants.EPS * qeso) / (
                pfld + (constants.EPS - 1) * qeso
            )  # fpsv is a function (can't be called inside conditional)
            # also how to access lookup table?
            val1 = 1.0e-8
            val2 = 1.0e-10
            qeso = max(qeso, val1)
            qo = max(qo, val2)

            # Calculate moist static energy (heo) and saturation moist
            # static energy (heso)
            tem = phil + constants.CP_AIR * to
            heo = tem + constants.HLV * qo
            heso = tem + constants.HLV * qeso


def init_tracers(
    cnvflg: BoolField,
    k_idx: IntField,
    kmax: IntField,
    ctr: FloatField,
    ctro: FloatField,
    ecko: FloatField,
    qtr: FloatField,
):
    with computation(PARALLEL), interval(...):
        # Initialize tracer variables
        if cnvflg and k_idx <= kmax:
            ctr = qtr
            ctro = qtr
            ecko = 0.0


def stencil_static0(
    cnvflg: BoolField,
    hmax: FloatField,
    heo: FloatField,
    kb: IntField,
    k_idx: IntField,
    kpbl: IntField,
    kmax: IntField,
    zo: FloatField,
    to: FloatField,
    qeso: FloatField,
    qo: FloatField,
    po: FloatField,
    uo: FloatField,
    vo: FloatField,
    heso: FloatField,
    pfld: FloatField,
):
    """
    Scale-Aware Mass-Flux Shallow Convection
    :to use the k_idx[1,0:im,0:km] as storage of 1 to k_idx index.
    """
    with computation(FORWARD), interval(0, 1):
        if cnvflg:
            hmax = heo
            kb = 1

    with computation(FORWARD), interval(1, None):
        hmax = hmax[0, 0, -1]
        kb = kb[0, 0, -1]
        if (cnvflg) and (k_idx <= kpbl):
            if heo > hmax:
                kb = k_idx
                hmax = heo

    # To make all slice like the final slice
    with computation(BACKWARD), interval(0, -1):
        kb = kb[0, 0, 1]
        hmax = hmax[0, 0, 1]

    with computation(FORWARD), interval(0, -1):
        tmp = fpvs(to[0, 0, 1])
        dz = 1.0
        dp = 1.0
        es = 1.0
        pprime = 1.0
        qs = 1.0
        dqsdp = 1.0
        desdt = 1.0
        dqsdt = 1.0
        gamma = 1.0
        dt = 1.0
        dq = 1.0

        if cnvflg[0, 0, 0] and k_idx[0, 0, 0] <= kmax[0, 0, 0] - 1:
            dz = 0.5 * (zo[0, 0, 1] - zo[0, 0, 0])
            dp = 0.5 * (pfld[0, 0, 1] - pfld[0, 0, 0])
            es = 0.01 * tmp  # fpvs is in pa
            pprime = pfld[0, 0, 1] + (constants.EPS - 1) * es
            qs = constants.EPS * es / pprime
            dqsdp = -qs / pprime
            desdt = es * (
                physcons.FACT1 / to[0, 0, 1] + physcons.FACT2 / (to[0, 0, 1] ** 2)
            )
            dqsdt = qs * pfld[0, 0, 1] * desdt / (es * pprime)
            gamma = physcons.EL2ORC * qeso[0, 0, 1] / (to[0, 0, 1] ** 2)
            dt = (constants.GRAV * dz + constants.HLV * dqsdp * dp) / (
                constants.CP_AIR * (1.0 + gamma)
            )
            dq = dqsdt * dt + dqsdp * dp
            to = to[0, 0, 1] + dt
            qo = qo[0, 0, 1] + dq
            po = 0.5 * (pfld[0, 0, 0] + pfld[0, 0, 1])

    with computation(FORWARD), interval(0, -1):
        tmp = fpvs(to)

        if cnvflg[0, 0, 0] and k_idx[0, 0, 0] <= kmax[0, 0, 0] - 1:
            qeso = 0.01 * tmp  # fpvs is in pa
            qeso = constants.EPS * qeso[0, 0, 0] / (
                po[0, 0, 0] + (constants.EPS - 1) * qeso[0, 0, 0]
            )
            # val1      =    1.e-8
            qeso = qeso[0, 0, 0] if (qeso[0, 0, 0] > 1.0e-8) else 1.0e-8
            # val2      =    1.e-10
            qo = qo[0, 0, 0] if (qo[0, 0, 0] > 1.0e-10) else 1.0e-10
            # qo   = min(qo[0,0,0],qeso[0,0,0])
            heo = (
                0.5 * constants.GRAV * (zo[0, 0, 0] + zo[0, 0, 1])
                + constants.CP_AIR * to[0, 0, 0]
                + constants.HLV * qo[0, 0, 0]
            )
            heso = (
                0.5 * constants.GRAV * (zo[0, 0, 0] + zo[0, 0, 1])
                + constants.CP_AIR * to[0, 0, 0]
                + constants.HLV * qeso[0, 0, 0]
            )
            uo = 0.5 * (uo[0, 0, 0] + uo[0, 0, 1])
            vo = 0.5 * (vo[0, 0, 0] + vo[0, 0, 1])


# ntr stencil put at last
def stencil_ntrstatic0(
    cnvflg: BoolField, k_idx: IntField, kmax: IntField, ctro: FloatField
):
    with computation(PARALLEL), interval(0, -1):
        if (cnvflg) and (k_idx <= (kmax - 1)):
            ctro = 0.5 * (ctro + ctro[0, 0, 1])


def stencil_static1(
    cnvflg: BoolField,
    flg: BoolField,
    kbcon: IntField,
    kmax: IntField,
    k_idx: IntField,
    kbm: IntField,
    kb: IntField,
    heo_kb: FloatField,
    heso: FloatField,
):
    with computation(PARALLEL), interval(...):
        flg = cnvflg
        if flg:
            kbcon = kmax

    with computation(FORWARD), interval(1, -1):
        kbcon = kbcon[0, 0, -1]
        flg = flg[0, 0, -1]
        if flg and k_idx < kbm:
            # To use heo_kb to represent heo(i,kb(i))
            if k_idx[0, 0, 0] > kb[0, 0, 0] and heo_kb > heso[0, 0, 0]:
                kbcon = k_idx
                flg = False

    # To make all slices like the final slice
    with computation(FORWARD), interval(-1, None):
        kbcon = kbcon[0, 0, -1]
        flg = flg[0, 0, -1]

    with computation(BACKWARD), interval(0, -1):
        kbcon = kbcon[0, 0, 1]
        flg = flg[0, 0, 1]

    with computation(PARALLEL), interval(...):
        if cnvflg:
            if kbcon == kmax:
                cnvflg = False


# Judge LFC and return 553-558
def stencil_static2(
    cnvflg: BoolField,
    pdot: FloatField,
    dot_kbcon: FloatField,
    islimsk: IntField,
    k_idx: IntField,
    kbcon: IntField,
    kb: IntField,
    pfld_kb: FloatField,
    pfld_kbcon: FloatField,
):
    with computation(PARALLEL), interval(...):
        if cnvflg:
            # To use dotkbcon to represent dot(i,kbcon(i))
            # pdot(i)  = 10.* dotkbcon
            pdot[0, 0, 0] = 0.01 * dot_kbcon  # Now dot is in Pa/s

    with computation(PARALLEL), interval(...):
        w1 = physcons.W1S
        w2 = physcons.W2S
        w3 = physcons.W3S
        w4 = physcons.W4S
        tem = 0.0
        tem1 = 0.0
        ptem = 0.0
        ptem1 = 0.0
        cinpcr = 0.0

        if cnvflg:
            if islimsk == 1:
                physcons.W1L
                w2 = physcons.W2L
                w3 = physcons.W3L
                w4 = physcons.W4L
            if pdot <= w4:
                tem = (pdot - w4) / (w3 - w4)
            elif pdot >= -w4:
                tem = -(pdot + w4) / (w4 - w3)
            else:
                tem = 0.0

            tem = tem if (tem > -1) else -1
            tem = tem if (tem < 1) else 1
            ptem = 1.0 - tem
            ptem1 = 0.5 * (physcons.CINPCRMX - physcons.CINPCRMN)
            cinpcr = physcons.CINPCRMX - ptem * ptem1

            # To use pfld_kb and pfld_kbcon to represent pfld(i,kb(i))
            tem1 = pfld_kb - pfld_kbcon
            if tem1 > cinpcr:
                cnvflg = False


# Do totflg judgement and return
# if ntk > 0 : also need to define ntk dimension to 1
def stencil_static3(
    sumx: FloatField,
    tkemean: FloatField,
    cnvflg: BoolField,
    k_idx: IntField,
    kb: IntField,
    kbcon: IntField,
    zo: FloatField,
    qtr: FloatField,
    clamt: FloatField,
    clam: Float,
):
    with computation(BACKWARD), interval(-1, None):
        if cnvflg:
            sumx = 0.0
            tkemean = 0.0

    with computation(BACKWARD), interval(0, -1):
        dz = 0.0
        tem = 0.0
        tkemean = tkemean[0, 0, 1]
        sumx = sumx[0, 0, 1]

        if cnvflg:
            if (k_idx >= kb) and (k_idx < kbcon):
                dz = zo[0, 0, 1] - zo[0, 0, 0]
                tem = 0.5 * (qtr[0, 0, 0] + qtr[0, 0, 1])
                tkemean = tkemean[0, 0, 1] + tem * dz  # dz, tem to be 3d
                sumx = sumx[0, 0, 1] + dz

    with computation(FORWARD), interval(1, None):
        tkemean = tkemean[0, 0, -1]
        sumx = sumx[0, 0, -1]

    with computation(PARALLEL), interval(...):
        tkemean = tkemean / sumx
        tem1 = 1.0 - 2.0 * (physcons.TKEMX - tkemean) / physcons.DTKE

        if cnvflg:
            if tkemean > physcons.TKEMX:  # tkemx, clam, clamd, tkemnm, dtke to be 3d
                clamt = clam + physcons.CLAMD
            elif tkemean < physcons.TKEMN:
                clamt = clam - physcons.CLAMD
            else:
                clamt = clam + physcons.CLAMD * tem1


# else :
def stencil_static4(cnvflg: BoolField, clamt: FloatField, *, clam: Float):
    with computation(PARALLEL), interval(...):
        if cnvflg:
            clamt = clam


# Start updraft entrainment rate.
# pass
def stencil_static5(
    cnvflg: BoolField,
    xlamue: FloatField,
    clamt: FloatField,
    zi: FloatField,
    xlamud: FloatField,
    k_idx: IntField,
    kbcon: IntField,
    kb: IntField,
    # dz   : FloatField,
    # ptem : FloatField,
    eta: FloatField,
    ktconn: IntField,
    kmax: IntField,
    kbm: IntField,
    hcko: FloatField,
    ucko: FloatField,
    vcko: FloatField,
    heo: FloatField,
    uo: FloatField,
    vo: FloatField,
):
    with computation(FORWARD), interval(0, -1):
        if cnvflg:
            xlamue = clamt / zi

    with computation(BACKWARD), interval(-1, None):
        if cnvflg:
            xlamue[0, 0, 0] = xlamue[0, 0, -1]

    with computation(PARALLEL), interval(...):
        if cnvflg:
            # xlamud(i) = xlamue(i,kbcon(i))
            # xlamud(i) = crtlamd
            xlamud = 0.001 * clamt

    with computation(BACKWARD), interval(0, -1):
        dz = 0.0
        ptem = 0.0
        if cnvflg:
            if k_idx < kbcon and k_idx >= kb:
                dz = zi[0, 0, 1] - zi[0, 0, 0]
                ptem = 0.5 * (xlamue[0, 0, 0] + xlamue[0, 0, 1]) - xlamud[0, 0, 0]
                eta = eta[0, 0, 1] / (1.0 + ptem * dz)

    with computation(PARALLEL), interval(...):
        flg = cnvflg

    with computation(FORWARD), interval(1, -1):
        flg = flg[0, 0, -1]
        kmax = kmax[0, 0, -1]
        ktconn = ktconn[0, 0, -1]
        kbm = kbm[0, 0, -1]
        if flg:
            if k_idx > kbcon and k_idx < kmax:
                dz = zi[0, 0, 0] - zi[0, 0, -1]
                ptem = 0.5 * (xlamue[0, 0, 0] + xlamue[0, 0, -1]) - xlamud[0, 0, 0]
                eta = eta[0, 0, -1] * (1 + ptem * dz)

                if eta <= 0.0:
                    kmax = k_idx
                    ktconn = k_idx
                    kbm = kbm if (kbm < kmax) else kmax
                    flg = False

    # To make all slice same as final slice
    with computation(FORWARD), interval(-1, None):
        flg = flg[0, 0, -1]
        kmax = kmax[0, 0, -1]
        ktconn = ktconn[0, 0, -1]
        kbm = kbm[0, 0, -1]

    with computation(BACKWARD), interval(0, -1):
        flg = flg[0, 0, 1]
        kmax = kmax[0, 0, 1]
        ktconn = ktconn[0, 0, 1]
        kbm = kbm[0, 0, 1]

    with computation(PARALLEL), interval(...):
        if cnvflg:
            # indx = kb
            if k_idx == kb:
                hcko = heo
                ucko = uo
                vcko = vo


# for tracers do n = 1, ntr: use ecko, ctro [n] => [1,i,k_idx]
# pass
def stencil_ntrstatic1(
    cnvflg: BoolField,
    k_idx: IntField,
    kb: IntField,
    ecko: FloatField,
    ctro: FloatField,
):
    with computation(PARALLEL), interval(...):
        if (cnvflg) and (k_idx == kb):
            ecko = ctro


# Line 769
# Calculate the cloud properties as a parcel ascends, modified by entrainment and
# detrainment. Discretization follows Appendix B of Grell (1993) \cite grell_1993.
# Following Han and Pan (2006) \cite han_and_pan_2006, the convective momentum
# transport is reduced by the convection-induced pressure gradient force by the
# constant "pgcon", currently set to 0.55 after Zhang and Wu (2003) 
# \cite zhang_and_wu_2003.
# pass
def stencil_static7(
    cnvflg: BoolField,
    k_idx: IntField,
    kb: IntField,
    kmax: IntField,
    zi: FloatField,
    xlamue: FloatField,
    xlamud: FloatField,
    hcko: FloatField,
    heo: FloatField,
    dbyo: FloatField,
    heso: FloatField,
    ucko: FloatField,
    uo: FloatField,
    vcko: FloatField,
    vo: FloatField,
    pgcon: Float
):
    with computation(FORWARD), interval(1, -1):
        dz = 0.0
        tem = 0.0
        tem1 = 0.0
        ptem = 0.0
        ptem1 = 0.0
        factor = 0.0

        if cnvflg:
            if k_idx > kb and k_idx < kmax:
                dz = zi[0, 0, 0] - zi[0, 0, -1]
                tem = 0.5 * (xlamue[0, 0, 0] + xlamue[0, 0, -1]) * dz
                tem1 = 0.5 * xlamud * dz
                factor = 1.0 + tem - tem1
                hcko = (
                    (1.0 - tem1) * hcko[0, 0, -1] + tem * 0.5 * (heo + heo[0, 0, -1])
                ) / factor
                dbyo = hcko - heso

                tem = 0.5 * physcons.CM * tem
                factor = 1.0 + tem
                ptem = tem + pgcon
                ptem1 = tem - pgcon
                ucko = (
                    (1.0 - tem) * ucko[0, 0, -1] + ptem * uo + ptem1 * uo[0, 0, -1]
                ) / factor
                vcko = (
                    (1.0 - tem) * vcko[0, 0, -1] + ptem * vo + ptem1 * vo[0, 0, -1]
                ) / factor


# for n = 1, ntr:
# pass
def stencil_ntrstatic2(
    cnvflg: BoolField,
    k_idx: IntField,
    kb: IntField,
    kmax: IntField,
    zi: FloatField,
    xlamue: FloatField,
    ecko: FloatField,
    ctro: FloatField,
):
    with computation(FORWARD), interval(1, -1):
        tem = 0.0
        dz = 0.0
        factor = 0.0

        if cnvflg:
            if k_idx > kb and k_idx < kmax:
                dz = zi - zi[0, 0, -1]
                tem = 0.25 * (xlamue + xlamue[0, 0, -1]) * dz
                factor = 1.0 + tem
                ecko = (
                    (1.0 - tem) * ecko[0, 0, -1] + tem * (ctro + ctro[0, 0, -1])
                ) / factor


# enddo
def stencil_update_kbcon1_cnvflg(
    dbyo: FloatField,
    cnvflg: BoolField,
    kmax: IntField,
    kbm: IntField,
    kbcon: IntField,
    kbcon1: IntField,
    flg: BoolField,
    k_idx: IntField,
):
    with computation(FORWARD), interval(0, 1):
        flg = cnvflg
        kbcon1 = kmax

    with computation(FORWARD), interval(1, None):
        flg = flg[0, 0, -1]
        kbcon1 = kbcon1[0, 0, -1]

        if flg and (k_idx < kbm):
            if (k_idx >= kbcon) and (dbyo > 0.0):
                kbcon1 = k_idx
                flg = False

    with computation(BACKWARD), interval(0, -1):
        flg = flg[0, 0, 1]
        kbcon1 = kbcon1[0, 0, 1]

    with computation(PARALLEL), interval(...):
        if cnvflg:
            if kbcon1 == kmax:
                cnvflg = False


# pass
def stencil_static9(
    cnvflg: BoolField, pfld_kbcon: FloatField, pfld_kbcon1: FloatField
):
    with computation(PARALLEL), interval(...):
        tem = 0.0

        if cnvflg:

            # Use pfld_kbcon and pfld_kbcon1 to represent
            # tem = pfld(i,kbcon(i)) - pfld(i,kbcon1(i))
            tem = pfld_kbcon - pfld_kbcon1
            if tem > physcons.DTHK:
                cnvflg = False


# Judge totflg return

# Calculate convective inhibition
# pass
def stencil_static10(
    cina: FloatField,
    cnvflg: BoolField,
    k_idx: IntField,
    kb: IntField,
    kbcon1: IntField,
    zo: FloatField,
    qeso: FloatField,
    to: FloatField,
    dbyo: FloatField,
    qo: FloatField,
    pdot: FloatField,
    islimsk: IntField,
):
    with computation(FORWARD), interval(1, -1):
        dz1 = 0.0
        gamma = 0.0
        rfact = 0.0
        cina = cina[0, 0, -1]

        if cnvflg:
            if k_idx > kb and k_idx < kbcon1:
                dz1 = zo[0, 0, 1] - zo
                gamma = physcons.EL2ORC * qeso / (to * to)
                rfact = 1.0 + physcons.DELTA * constants.CP_AIR * (
                    gamma * to / constants.HLV
                )
                cina = cina + dz1 * (
                    constants.GRAV / (constants.CP_AIR * to)
                ) * dbyo / (1.0 + gamma) * rfact
                # val   = 0.
                cina = (
                    (
                        cina + dz1 * constants.GRAV * physcons.DELTA * (qeso - qo)
                        # dz1 * eta(i,k_idx) * g * delta *
                    )
                    if ((qeso - qo) > 0.0)
                    else cina
                )

    # To make all slices like the final slice
    with computation(FORWARD), interval(-1, None):
        cina = cina[0, 0, -1]

    with computation(BACKWARD), interval(0, -1):
        cina = cina[0, 0, 1]

    with computation(PARALLEL), interval(...):
        w1 = physcons.W1S
        w2 = physcons.W2S
        w3 = physcons.W3S
        w4 = physcons.W4S
        tem = 0.0
        tem1 = 0.0
        cinacr = 0.0

        if cnvflg:
            if islimsk == 1:
                w1 = physcons.W1L
                w2 = physcons.W2L
                w3 = physcons.W3L
                w4 = physcons.W4L

            if pdot <= w4:
                tem = (pdot - w4) / (w3 - w4)
            elif pdot >= -w4:
                tem = -(pdot + w4) / (w4 - w3)
            else:
                tem = 0.0

            # val1   =            -1.
            tem = tem if (tem > -1.0) else -1.0
            # val2   =             1.
            tem = tem if (tem < 1.0) else 1.0
            tem = 1.0 - tem
            tem1 = 0.5 * (physcons.CINACRMX - physcons.CINACRMN)
            cinacr = physcons.CINACRMX - tem * tem1
            # cinacr = cinacrmx
            if cina < cinacr:
                cnvflg = False


# totflag and return

#  Determine first guess cloud top as the level of zero buoyancy
#    limited to the level of P/Ps=0.7
# pass
def stencil_static11(
    flg: BoolField,
    cnvflg: BoolField,
    ktcon: IntField,
    kbm: IntField,
    kbcon1: IntField,
    dbyo: FloatField,
    kbcon: IntField,
    del0: FloatField,
    xmbmax: FloatField,
    aa1: FloatField,
    kb: IntField,
    qcko: FloatField,
    qo: FloatField,
    qrcko: FloatField,
    zi: FloatField,
    qeso: FloatField,
    to: FloatField,
    xlamue: FloatField,
    xlamud: FloatField,
    eta: FloatField,
    c0t: FloatField,
    dellal: FloatField,
    buo: FloatField,
    drag: FloatField,
    zo: FloatField,
    k_idx: IntField,
    pwo: FloatField,
    cnvwt: FloatField,
    c1: Float,
    dt2: Float,
    ncloud: Int
):
    with computation(PARALLEL), interval(...):
        flg = cnvflg
        if flg:
            ktcon = kbm

    with computation(FORWARD), interval(1, -1):
        flg = flg[0, 0, -1]
        ktcon = ktcon[0, 0, -1]
        if flg and k_idx < kbm:
            if k_idx > kbcon1 and dbyo < 0.0:
                ktcon = k_idx
                flg = False

    # To make all slices like final slice
    with computation(FORWARD), interval(-1, None):
        flg = flg[0, 0, -1]
        ktcon = ktcon[0, 0, -1]

    with computation(BACKWARD), interval(0, -1):
        flg = flg[0, 0, 1]
        ktcon = ktcon[0, 0, 1]

    # Specify upper limit of mass flux at cloud base

    with computation(FORWARD), interval(...):
        dp = 0.0

        if k_idx != 1:
            xmbmax = xmbmax[0, 0, -1]

        if cnvflg:
            if k_idx == kbcon:
                dp = 1000.0 * del0

                xmbmax = dp / (2.0 * constants.GRAV * dt2)

    with computation(BACKWARD), interval(0, -1):
        xmbmax = xmbmax[0, 0, 1]

    # Compute cloud moisture property and precipitation
    with computation(PARALLEL), interval(...):
        if cnvflg:
            aa1 = 0.0
            if k_idx == kb:
                qcko = qo
                qrcko = qo

    # Calculate the moisture content of the entraining/detraining parcel (qcko)
    # and the value it would have if just saturated (qrch), according to equation A.14
    # in Grell (1993) \cite grell_1993 . Their difference is the amount of convective
    # cloud water (qlk = rain + condensate). Determine the portion of convective cloud
    # water that remains suspended and the portion that is converted into convective
    # precipitation (pwo). Calculate and save the negative cloud work function (aa1)
    # due to water loading. Above the level of minimum moist static energy, some of the
    # cloud water is detrained into the grid-scale cloud water from every cloud layer
    # with a rate of 0.0005 \f$m^{-1}\f$ (dellal).
    with computation(FORWARD), interval(1, -1):
        dz = 0.0
        gamma = 0.0
        qrch = 0.0
        tem = 0.0
        tem1 = 0.0
        factor = 0.0
        dq = 0.0
        etah = 0.0
        dp = 0.0
        ptem = 0.0
        qlk = 0.0
        rfact = 0.0

        if cnvflg:
            if k_idx > kb and k_idx < ktcon:
                dz = zi - zi[0, 0, -1]
                gamma = physcons.EL2ORC * qeso / (to ** 2)
                qrch = qeso + gamma * dbyo / (constants.HLV * (1.0 + gamma))
                # j
                tem = 0.5 * (xlamue + xlamue[0, 0, -1]) * dz
                tem1 = 0.5 * xlamud * dz
                factor = 1.0 + tem - tem1
                qcko = (
                    (1.0 - tem1) * qcko[0, 0, -1] + tem * 0.5 * (qo + qo[0, 0, -1])
                ) / factor
                qrcko = qcko
                # j
                dq = eta * (qcko - qrch)

                # rhbar(i) = rhbar(i) + qo(i,k_idx) / qeso(i,k_idx)

                # Below lfc check if there is excess moisture to release
                # latent heat
                if k_idx >= kbcon and dq > 0.0:
                    etah = 0.5 * (eta + eta[0, 0, -1])
                    dp = 1000.0 * del0

                    if ncloud > 0:
                        ptem = c0t + c1
                        qlk = dq / (eta + etah * ptem * dz)
                        dellal = etah * c1 * dz * qlk * constants.GRAV / dp
                    else:
                        qlk = dq / (eta + etah * c0t * dz)

                    buo = buo - constants.GRAV * qlk
                    qcko = qlk + qrch
                    pwo = etah * c0t * dz * qlk
                    cnvwt = etah * qlk * constants.GRAV / dp

                if k_idx >= kbcon:
                    rfact = 1.0 + physcons.DELTA * constants.CP_AIR * gamma * (
                        to / constants.HLV
                    )
                    buo = buo + (
                        constants.GRAV / (constants.CP_AIR * to)
                    ) * dbyo / (1.0 + gamma) * rfact

                    # val = 0.
                    buo = (
                        (
                            buo + constants.GRAV * physcons.constants.GRAV * (qeso - qo)
                        ) if ((qeso - qo) > 0.0) else buo
                    )
                    drag = xlamue if (xlamue > xlamud) else xlamud

    # L1064: Calculate the cloud work function according to Pan and Wu (1995) 
    # \cite pan_and_wu_1995 equation 4
    with computation(PARALLEL), interval(...):
        if cnvflg:
            aa1 = 0.0

    with computation(FORWARD), interval(1, -1):
        aa1 = aa1[0, 0, -1]
        dz1 = 0.0
        if cnvflg:
            if k_idx >= kbcon and k_idx < ktcon:
                dz1 = zo[0, 0, 1] - zo
                aa1 = aa1 + buo * dz1

    # To make all slices like final slice
    with computation(FORWARD), interval(-1, None):
        aa1 = aa1[0, 0, -1]
    with computation(BACKWARD), interval(0, -1):
        aa1 = aa1[0, 0, 1]

    with computation(PARALLEL), interval(...):
        if cnvflg and aa1 <= 0.0:
            cnvflg = False


# totflg and return

# Estimate the onvective overshooting as the level
#   where the [aafac * cloud work function] becomes zero,
#   which is the final cloud top
#   limited to the level of P/Ps=0.7

# Continue calculating the cloud work function past the point of neutral buoyancy to
# represent overshooting according to Han and Pan (2011) \cite han_and_pan_2011.
# Convective overshooting stops when \f$ cA_u < 0\f$ where \f$c\f$ is currently 10%,
# or when 10% of the updraft cloud work function has been consumed by the stable
# buoyancy force. Overshooting is also limited to the level where \f$p=0.7p_{sfc}\f$.
# pass
def stencil_static12(
    cnvflg: BoolField,
    aa1: FloatField,
    flg: BoolField,
    ktcon1: IntField,
    kbm: IntField,
    k_idx: IntField,
    ktcon: IntField,
    zo: FloatField,
    qeso: FloatField,
    to: FloatField,
    dbyo: FloatField,
    zi: FloatField,
    xlamue: FloatField,
    xlamud: FloatField,
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
    wc: FloatField,
    sumx: FloatField,
    kbcon1: IntField,
    drag: FloatField,
    dellal: FloatField,
    c1: Float,
    ncloud: Int
):
    with computation(PARALLEL), interval(...):
        if cnvflg:
            aa1 = physcons.AAFAC * aa1

        flg = cnvflg
        ktcon1 = kbm

    with computation(FORWARD), interval(1, -1):
        dz1 = 0.0
        gamma = 0.0
        rfact = 0.0
        aa1 = aa1[0, 0, -1]
        ktcon1 = ktcon1[0, 0, -1]
        flg = flg[0, 0, -1]

        if flg:
            if k_idx >= ktcon and k_idx < kbm:
                dz1 = zo[0, 0, 1] - zo
                gamma = physcons.EL2ORC * qeso / (to ** 2)
                rfact = 1.0 + physcons.DELTA * constants.CP_AIR * gamma * (
                    to / constants.HLV
                )
                aa1 = aa1 + dz1 * (
                    constants.GRAV / (constants.CP_AIR * to)
                ) * dbyo / (1.0 + gamma) * rfact

                # val = 0.
                # aa1(i) = aa1(i) +
                #         dz1 * eta(i,k_idx) * g * delta *
                #         dz1 * g * delta *
                #         max(val,(qeso(i,k_idx) - qo(i,k_idx)))

                if aa1 < 0.0:
                    ktcon1 = k_idx
                    flg = False

    # To make all slice like final slice
    with computation(FORWARD), interval(-1, None):
        aa1 = aa1[0, 0, -1]
        ktcon1 = ktcon1[0, 0, -1]
        flg = flg[0, 0, -1]

    with computation(BACKWARD), interval(0, -1):
        aa1 = aa1[0, 0, 1]
        ktcon1 = ktcon1[0, 0, 1]
        flg = flg[0, 0, 1]

    # Compute cloud moisture property, detraining cloud water
    # and precipitation in overshooting layers

    # For the overshooting convection, calculate the moisture content of the
    # entraining/detraining parcel as before. Partition convective cloud water and
    # precipitation and detrain convective cloud water in the overshooting layers.
    with computation(FORWARD), interval(1, -1):
        dz = 0.0
        gamma = 0.0
        qrch = 0.0
        tem = 0.0
        tem1 = 0.0
        factor = 0.0
        dq = 0.0
        etah = 0.0
        ptem = 0.0
        qlk = 0.0
        dp = 0.0

        if cnvflg:
            if k_idx >= ktcon and k_idx < ktcon1:
                dz = zi - zi[0, 0, -1]
                gamma = physcons.EL2ORC * qeso / (to ** 2)
                qrch = qeso + gamma * dbyo / (constants.HLV * (1.0 + gamma))
                # j
                tem = 0.5 * (xlamue + xlamue[0, 0, -1]) * dz
                tem1 = 0.5 * xlamud * dz
                factor = 1.0 + tem - tem1
                qcko = (
                    (1.0 - tem1) * qcko[0, 0, -1] + tem * 0.5 * (qo + qo[0, 0, -1])
                ) / factor
                qrcko = qcko
                # j
                dq = eta * (qcko - qrch)

                # Check if there is excess moisture to release latent heat
                if dq > 0.0:
                    etah = 0.5 * (eta + eta[0, 0, -1])
                    dp = 1000.0 * del0
                    if ncloud > 0:
                        ptem = c0t + c1
                        qlk = dq / (eta + etah * ptem * dz)
                        dellal = etah * c1 * dz * qlk * constants.GRAV / dp
                    else:
                        qlk = dq / (eta + etah * c0t * dz)

                    qcko = qlk + qrch
                    pwo = etah * c0t * dz * qlk
                    cnvwt = etah * qlk * constants.GRAV / dp

    # Compute updraft velocity square(wu2)
    # Calculate updraft velocity square(wu2) according to Han et al.'s
    # (2017) \cite han_et_al_2017 equation 7.
    with computation(FORWARD), interval(1, -1):
        dz = 0.0
        tem = 0.0
        tem1 = 0.0
        ptem = 0.0
        ptem1 = 0.0
        # bb1   = 4.0
        # bb2   = 0.8
        if cnvflg:
            if k_idx > kbcon1 and k_idx < ktcon:
                dz = zi - zi[0, 0, -1]
                tem = 0.25 * 4.0 * (drag + drag[0, 0, -1]) * dz
                tem1 = 0.5 * 0.8 * (buo + buo[0, 0, -1]) * dz
                ptem = (1.0 - tem) * wu2[0, 0, -1]
                ptem1 = 1.0 + tem
                wu2 = (ptem + tem1) / ptem1
                wu2 = wu2 if (wu2 > 0.0) else 0.0

    # Compute updraft velocity averaged over the whole cumulus
    with computation(PARALLEL), interval(...):
        wc = 0.0
        sumx = 0.0

    with computation(FORWARD), interval(1, -1):
        dz = 0.0
        tem = 0.0
        wc = wc[0, 0, -1]
        sumx = sumx[0, 0, -1]

        if cnvflg:
            if k_idx > kbcon1 and k_idx < ktcon:
                dz = zi - zi[0, 0, -1]
                tem = 0.5 * ((wu2) ** 0.5 + (wu2[0, 0, -1]) ** 0.5)
                wc = wc + tem * dz
                sumx = sumx + dz

    # To make all slices like final slice
    with computation(FORWARD), interval(-1, None):
        wc = wc[0, 0, -1]
        sumx = sumx[0, 0, -1]

    with computation(BACKWARD), interval(0, -1):
        wc = wc[0, 0, 1]
        sumx = sumx[0, 0, 1]

    with computation(PARALLEL), interval(...):

        if cnvflg:
            if sumx == 0.0:
                cnvflg = False
            else:
                wc = wc / sumx

            # val = 1.e-4
            if wc < 1.0e-4:
                cnvflg = False

    # Exchange ktcon with ktcon1
    with computation(PARALLEL), interval(...):
        kk = 1
        if cnvflg:
            kk = ktcon
            ktcon = ktcon1
            ktcon1 = kk


# This section is ready for cloud water
#  if(ncloud > 0):
# pass
def stencil_static13(
    cnvflg: BoolField,
    k_idx: IntField,
    ktcon: IntField,
    qeso: FloatField,
    to: FloatField,
    dbyo: FloatField,
    qcko: FloatField,
    qlko_ktcon: FloatField,
):
    with computation(FORWARD), interval(1, None):
        gamma = 0.0
        qrch = 0.0
        dq = 0.0

        if cnvflg:

            qlko_ktcon = qlko_ktcon[0, 0, -1]
            if k_idx == ktcon - 1:
                gamma = physcons.EL2ORC * qeso / (to * to)
                qrch = qeso + gamma * dbyo / (constants.HLV * (1.0 + gamma))
                dq = qcko - qrch
                # Check if there is excess moisture to release latent heat
                if dq > 0.0:
                    qlko_ktcon = dq
                    qcko = qrch

    with computation(BACKWARD), interval(0, -1):
        qlko_ktcon = qlko_ktcon[0, 0, 1]


# endif

# Compute precipitation efficiency in terms of windshear
# pass
def stencil_static14(
    cnvflg: BoolField,
    vshear: FloatField,
    k_idx: IntField,
    kb: IntField,
    ktcon: IntField,
    uo: FloatField,
    vo: FloatField,
    zi: FloatField,
    edt: FloatField,
):
    with computation(PARALLEL), interval(...):
        if cnvflg:
            vshear = 0.0

    with computation(FORWARD), interval(1, None):
        vshear = vshear[0, 0, -1]
        if cnvflg:
            if k_idx > kb and k_idx <= ktcon:
                # shear = ((uo-uo[0,0,-1]) ** 2 \
                #      + (vo-vo[0,0,-1]) ** 2)**0.5
                vshear = (
                    vshear
                    + ((uo - uo[0, 0, -1]) ** 2 + (vo - vo[0, 0, -1]) ** 2) ** 0.5
                )

    # To make all slice like final slice
    with computation(BACKWARD), interval(0, -1):
        vshear = vshear[0, 0, 1]

    with computation(FORWARD), interval(...):
        zi_kb = zi
        zi_ktcon = zi

        if k_idx != 1:
            zi_kb = zi_kb[0, 0, -1]
            zi_ktcon = zi_ktcon[0, 0, -1]

        if k_idx == kb:
            zi_kb = zi

        if k_idx == ktcon:
            zi_ktcon = zi

    with computation(BACKWARD), interval(0, -1):
        zi_kb = zi_kb[0, 0, 1]
        zi_ktcon = zi_ktcon[0, 0, 1]

    with computation(PARALLEL), interval(...):
        if cnvflg:
            # Use ziktcon and zikb to represent zi(ktcon) and zi(kb)
            vshear = 1.0e3 * vshear / (zi_ktcon - zi_kb)

            # e1 = 1.591-.639*vshear \
            #   + .0953*(vshear**2)-.00496*(vshear**3)

            edt = 1.0 - (
                1.591
                - 0.639 * vshear
                + 0.0953 * (vshear ** 2)
                - 0.00496 * (vshear ** 3)
            )
            # val = .9
            edt = edt if (edt < 0.9) else 0.9
            # val = .0
            edt = edt if (edt > 0.0) else 0.0


def comp_tendencies(
    cnvflg: BoolField,
    k_idx: IntField,
    kmax: IntField,
    kb: IntField,
    ktcon: IntField,
    ktcon1: IntField,
    kbcon1: IntField,
    kbcon: IntField,
    dellah: FloatField,
    dellaq: FloatField,
    dellau: FloatField,
    dellav: FloatField,
    del0: FloatField,
    zi: FloatField,
    zi_ktcon1: FloatField,
    zi_kbcon1: FloatField,
    heo: FloatField,
    qo: FloatField,
    xlamue: FloatField,
    xlamud: FloatField,
    eta: FloatField,
    hcko: FloatField,
    qrcko: FloatField,
    uo: FloatField,
    ucko: FloatField,
    vo: FloatField,
    vcko: FloatField,
    qcko: FloatField,
    dellal: FloatField,
    qlko_ktcon: FloatField,
    wc: FloatField,
    gdx: FloatField,
    dtconv: FloatField,
    u1: FloatField,
    v1: FloatField,
    po: FloatField,
    to: FloatField,
    tauadv: FloatField,
    xmb: FloatField,
    sigmagfm: FloatField,
    garea: FloatField,
    scaldfunc: FloatField,
    xmbmax: FloatField,
    sumx: FloatField,
    umean: FloatField,
    dt2: Float,
):
    # Calculate the change in moist static energy, moisture
    # mixing ratio, and horizontal winds per unit cloud base mass
    # flux for all layers below cloud top from equations B.14
    # and B.15 from Grell (1993) \cite grell_1993, and for the
    # cloud top from B.16 and B.17

    # Initialize zi_ktcon1 and zi_kbcon1 fields (propagate forward)
    with computation(FORWARD), interval(...):

        if k_idx == ktcon1:
            zi_ktcon1 = zi
        elif k_idx > 0:
            zi_ktcon1 = zi_ktcon1[0, 0, -1]

        if k_idx == kbcon1:
            zi_kbcon1 = zi
        elif k_idx > 0:
            zi_kbcon1 = zi_kbcon1[0, 0, -1]

    # Initialize zi_ktcon1 and zi_kbcon1 fields (propagate backward)
    with computation(BACKWARD), interval(0, -1):

        zi_ktcon1 = zi_ktcon1[0, 0, 1]
        zi_kbcon1 = zi_kbcon1[0, 0, 1]

    with computation(PARALLEL), interval(...):

        if cnvflg and k_idx <= kmax:
            dellah = 0.0
            dellaq = 0.0
            dellau = 0.0
            dellav = 0.0

    with computation(PARALLEL), interval(1, -1):

        dp = 0.0
        dz = 0.0
        gdp = 0.0

        dv1h = 0.0
        dv3h = 0.0
        dv2h = 0.0

        dv1q = 0.0
        dv3q = 0.0
        dv2q = 0.0

        tem = 0.0
        tem1 = 0.0

        eta_curr = 0.0
        eta_prev = 0.0

        tem2 = 0.0

        # Changes due to subsidence and entrainment
        if cnvflg and k_idx > kb and k_idx < ktcon:

            dp = 1000.0 * del0
            dz = zi[0, 0, 0] - zi[0, 0, -1]
            gdp = constants.GRAV / dp

            dv1h = heo[0, 0, 0]
            dv3h = heo[0, 0, -1]
            dv2h = 0.5 * (dv1h + dv3h)

            dv1q = qo[0, 0, 0]
            dv3q = qo[0, 0, -1]
            dv2q = 0.5 * (dv1q + dv3q)

            tem = 0.5 * (xlamue[0, 0, 0] + xlamue[0, 0, -1])
            tem1 = xlamud

            eta_curr = eta[0, 0, 0]
            eta_prev = eta[0, 0, -1]

            dellah = (
                dellah
                + (
                    eta_curr * dv1h
                    - eta_prev * dv3h
                    - eta_prev * dv2h * tem * dz
                    + eta_prev * tem1 * 0.5 * dz * (hcko[0, 0, 0] + hcko[0, 0, -1])
                )
                * gdp
            )

            dellaq = (
                dellaq
                + (
                    eta_curr * dv1q
                    - eta_prev * dv3q
                    - eta_prev * dv2q * tem * dz
                    + eta_prev * tem1 * 0.5 * dz * (qrcko[0, 0, 0] + qcko[0, 0, -1])
                )
                * gdp
            )

            tem1 = eta_curr * (uo[0, 0, 0] - ucko[0, 0, 0])
            tem2 = eta_prev * (uo[0, 0, -1] - ucko[0, 0, -1])
            dellau = dellau + (tem1 - tem2) * gdp

            tem1 = eta_curr * (vo[0, 0, 0] - vcko[0, 0, 0])
            tem2 = eta_prev * (vo[0, 0, -1] - vcko[0, 0, -1])
            dellav = dellav + (tem1 - tem2) * gdp

    with computation(PARALLEL), interval(1, None):

        tfac = 0.0

        # Cloud top
        if cnvflg:

            if ktcon == k_idx:

                dp = 1000.0 * del0
                gdp = constants.GRAV / dp

                dv1h = heo[0, 0, -1]
                dellah = eta[0, 0, -1] * (hcko[0, 0, -1] - dv1h) * gdp

                dv1q = qo[0, 0, -1]
                dellaq = eta[0, 0, -1] * (qcko[0, 0, -1] - dv1q) * gdp

                dellau = eta[0, 0, -1] * (ucko[0, 0, -1] - uo[0, 0, -1]) * gdp
                dellav = eta[0, 0, -1] * (vcko[0, 0, -1] - vo[0, 0, -1]) * gdp

                # Cloud water
                dellal = eta[0, 0, -1] * qlko_ktcon * gdp

    with computation(PARALLEL), interval(...):

        # Following Bechtold et al. (2008) \cite
        # bechtold_et_al_2008, calculate the convective turnover
        # time using the mean updraft velocity (wc) and the cloud
        # depth. It is also proportional to the grid size (gdx).
        if cnvflg:

            tem = zi_ktcon1 - zi_kbcon1
            tfac = 1.0 + gdx / 75000.0
            dtconv = tfac * tem / wc
            dtconv = max(dtconv, physcons.DTMIN)
            dtconv = max(dtconv, dt2)
            dtconv = min(dtconv, physcons.DTMAX)

            # Initialize field for advective time scale computation
            sumx = 0.0
            umean = 0.0

    # Calculate advective time scale (tauadv) using a mean cloud layer
    # wind speed (propagate forward)
    with computation(FORWARD), interval(1, -1):

        if cnvflg:
            if k_idx >= kbcon1 and k_idx < ktcon1:
                dz = zi[0, 0, 0] - zi[0, 0, -1]
                tem = (u1 * u1 + v1 * v1) ** 0.5  # sqrt(u1*u1 + v1*v1)
                umean = umean[0, 0, -1] + tem * dz
                sumx = sumx[0, 0, -1] + dz
            else:
                umean = umean[0, 0, -1]
                sumx = sumx[0, 0, -1]

    # Calculate advective time scale (tauadv) using a mean cloud layer
    # wind speed (propagate backward)
    with computation(BACKWARD), interval(1, -2):
        if cnvflg:
            umean = umean[0, 0, 1]
            sumx = sumx[0, 0, 1]

    with computation(PARALLEL), interval(...):

        rho = 0.0
        val = 1.0
        val1 = 2.0e-4
        val2 = 6.0e-4
        val3 = 0.001
        val4 = 0.999
        val5 = 0.0

        if cnvflg:
            umean = umean / sumx
            umean = max(
                umean, val
            )  # Passing literals (e.g. 1.0) to functions might cause errors
            # in conditional statements
            tauadv = gdx / umean

    with computation(FORWARD):

        with interval(0, 1):

            if cnvflg and k_idx == kbcon:

                # From Han et al.'s (2017) \cite han_et_al_2017 equation
                # 6, calculate cloud base mass flux as a function of the
                # mean updraft velocity
                rho = po * 100.0 / (constants.RDGAS * to)
                tfac = tauadv / dtconv
                tfac = min(tfac, val)  # Same as above: literals
                xmb = tfac * physcons.BETAW * rho * wc

                # For scale-aware parameterization, the updraft fraction
                # (sigmagfm) is first computed as a function of the
                # lateral entrainment rate at cloud base (see Han et
                # al.'s (2017) \cite han_et_al_2017 equation 4 and 5),
                # following the study by Grell and Freitas (2014) \cite
                # grell_and_freitus_2014
                tem = max(xlamue, val1)
                tem = min(tem, val2)
                tem = 0.2 / tem
                tem1 = 3.14 * tem * tem

                sigmagfm = tem1 / garea
                sigmagfm = max(sigmagfm, val3)
                sigmagfm = min(sigmagfm, val4)

        with interval(1, None):

            if cnvflg and k_idx == kbcon:

                # From Han et al.'s (2017) \cite han_et_al_2017 equation
                # 6, calculate cloud base mass flux as a function of the
                # mean updraft velocity
                rho = po * 100.0 / (constants.RDGAS * to)
                tfac = tauadv / dtconv
                tfac = min(tfac, val)  # Same as above: literals
                xmb = tfac * physcons.BETAW * rho * wc

                # For scale-aware parameterization, the updraft fraction
                # (sigmagfm) is first computed as a function of the
                # lateral entrainment rate at cloud base (see Han et
                # al.'s (2017) \cite han_et_al_2017 equation 4 and 5),
                # following the study by Grell and Freitas (2014) \cite
                # grell_and_freitus_2014
                tem = max(xlamue, val1)
                tem = min(tem, val2)
                tem = 0.2 / tem
                tem1 = 3.14 * tem * tem

                sigmagfm = tem1 / garea
                sigmagfm = max(sigmagfm, val3)
                sigmagfm = min(sigmagfm, val4)

            else:

                xmb = xmb[0, 0, -1]
                sigmagfm = sigmagfm[0, 0, -1]

    with computation(BACKWARD), interval(0, -1):

        if cnvflg:
            xmb = xmb[0, 0, 1]
            sigmagfm = sigmagfm[0, 0, 1]

    with computation(PARALLEL), interval(...):

        # Vertical convective eddy transport of mass flux as a
        # function of updraft fraction from the studies by Arakawa
        # and Wu (2013) \cite arakawa_and_wu_2013 (also see Han et
        # al.'s (2017) \cite han_et_al_2017 equation 1 and 2). The
        # final cloud base mass flux with scale-aware
        # parameterization is obtained from the mass flux when
        # sigmagfm << 1, multiplied by the reduction factor (Han et
        # al.'s (2017) \cite han_et_al_2017 equation 2).
        if cnvflg:
            if gdx < physcons.DXCRT:
                scaldfunc = (1.0 - sigmagfm) * (1.0 - sigmagfm)
                scaldfunc = min(scaldfunc, val)
                scaldfunc = max(scaldfunc, val5)
            else:
                scaldfunc = 1.0

            xmb = xmb * scaldfunc
            xmb = min(xmb, xmbmax)


def comp_tendencies_tr(
    cnvflg: BoolField,
    k_idx: IntField,
    kmax: IntField,
    kb: IntField,
    ktcon: IntField,
    dellae: FloatField,
    del0: FloatField,
    eta: FloatField,
    ctro: FloatField,
    ecko: FloatField,
):
    with computation(PARALLEL), interval(...):

        if cnvflg and k_idx <= kmax:

            dellae = 0.0

    with computation(PARALLEL), interval(1, -1):

        tem1 = 0.0
        tem2 = 0.0
        dp = 0.0

        if cnvflg and k_idx > kb and k_idx < ktcon:

            # Changes due to subsidence and entrainment
            dp = 1000.0 * del0

            tem1 = eta[0, 0, 0] * (ctro[0, 0, 0] - ecko[0, 0, 0])
            tem2 = eta[0, 0, -1] * (ctro[0, 0, -1] - ecko[0, 0, -1])

            dellae = dellae + (tem1 - tem2) * constants.GRAV / dp

    with computation(PARALLEL), interval(1, None):

        # Cloud top
        if cnvflg and ktcon == k_idx:

            dp = 1000.0 * del0

            dellae = eta[0, 0, -1] * (
                ecko[0, 0, -1] - ctro[0, 0, -1]
            ) * constants.GRAV / dp


def feedback_control_update(
    cnvflg: BoolField,
    k_idx: IntField,
    kmax: IntField,
    kb: IntField,
    ktcon: IntField,
    flg: BoolField,
    islimsk: IntField,
    ktop: IntField,
    kbot: IntField,
    kbcon: IntField,
    kcnv: IntField,
    qeso: FloatField,
    pfld: FloatField,
    delhbar: FloatField,
    delqbar: FloatField,
    deltbar: FloatField,
    delubar: FloatField,
    delvbar: FloatField,
    qcond: FloatField,
    dellah: FloatField,
    dellaq: FloatField,
    t1: FloatField,
    xmb: FloatField,
    q1: FloatField,
    u1: FloatField,
    dellau: FloatField,
    v1: FloatField,
    dellav: FloatField,
    del0: FloatField,
    rntot: FloatField,
    delqev: FloatField,
    delq2: FloatField,
    pwo: FloatField,
    deltv: FloatField,
    delq: FloatField,
    qevap: FloatField,
    rn: FloatField,
    edt: FloatField,
    cnvw: FloatField,
    cnvwt: FloatField,
    cnvc: FloatField,
    ud_mf: FloatField,
    dt_mf: FloatField,
    eta: FloatField,
    dt2: Float,
    evfact: Float,
    evfactl: Float,
):
    with computation(PARALLEL), interval(...):

        # Initialize flg
        flg = cnvflg

        # Recalculate saturation specific humidity
        qeso = 0.01 * fpvs(t1)  # fpvs is in Pa
        qeso = constants.EPS * qeso / (pfld + (constants.EPS - 1) * qeso)
        val = 1.0e-8
        qeso = max(qeso, val)

        dellat = 0.0
        fpvst1 = 0.0

        # - Calculate the temperature tendency from the moist
        #   static energy and specific humidity tendencies
        # - Update the temperature, specific humidity, and
        #   horizontal wind state variables by multiplying the
        #   cloud base mass flux-normalized tendencies by the
        #   cloud base mass flux
        if cnvflg:
            if k_idx > kb and k_idx <= ktcon:
                dellat = (dellah - constants.HLV * dellaq) / constants.CP_AIR
                t1 = t1 + dellat * xmb * dt2

        fpvst1 = 0.01 * fpvs(t1)  # fpvs is in Pa

        if cnvflg:
            if k_idx > kb and k_idx <= ktcon:

                q1 = q1 + dellaq * xmb * dt2
                u1 = u1 + dellau * xmb * dt2
                v1 = v1 + dellav * xmb * dt2

                # Recalculate saturation specific humidity using the
                # updated temperature
                qeso = fpvst1
                qeso = constants.EPS * qeso / (pfld + (constants.EPS - 1) * qeso)
                qeso = max(qeso, val)

    # Accumulate column-integrated tendencies (propagate forward)
    with computation(FORWARD):

        # To avoid conditionals in the full interval
        with interval(0, 1):

            dp = 0.0
            dpg = 0.0

            if cnvflg and k_idx > kb and k_idx <= ktcon:

                dp = 1000.0 * del0
                dpg = dp / constants.GRAV

                delhbar = delhbar + dellah * xmb * dpg
                delqbar = delqbar + dellaq * xmb * dpg
                deltbar = deltbar + dellat * xmb * dpg
                delubar = delubar + dellau * xmb * dpg
                delvbar = delvbar + dellav * xmb * dpg

        with interval(1, None):

            if cnvflg:
                if k_idx > kb and k_idx <= ktcon:

                    dp = 1000.0 * del0
                    dpg = dp / constants.GRAV

                    delhbar = delhbar[0, 0, -1] + dellah * xmb * dpg
                    delqbar = delqbar[0, 0, -1] + dellaq * xmb * dpg
                    deltbar = deltbar[0, 0, -1] + dellat * xmb * dpg
                    delubar = delubar[0, 0, -1] + dellau * xmb * dpg
                    delvbar = delvbar[0, 0, -1] + dellav * xmb * dpg

                else:

                    delhbar = delhbar[0, 0, -1]
                    delqbar = delqbar[0, 0, -1]
                    deltbar = deltbar[0, 0, -1]
                    delubar = delubar[0, 0, -1]
                    delvbar = delvbar[0, 0, -1]

    with computation(BACKWARD):

        # To avoid conditionals in the full interval
        with interval(-1, None):

            if cnvflg:
                if k_idx > kb and k_idx < ktcon:
                    rntot = rntot + pwo * xmb * 0.001 * dt2

        with interval(0, -1):
            if cnvflg:

                # Accumulate column-integrated tendencies (propagate backward)
                delhbar = delhbar[0, 0, 1]
                delqbar = delqbar[0, 0, 1]
                deltbar = deltbar[0, 0, 1]
                delubar = delubar[0, 0, 1]
                delvbar = delvbar[0, 0, 1]

                # Add up column-integrated convective precipitation by
                # multiplying the normalized value by the cloud base
                # mass flux (propagate backward)
                if k_idx > kb and k_idx < ktcon:

                    rntot = rntot[0, 0, 1] + pwo * xmb * 0.001 * dt2

                else:

                    rntot = rntot[0, 0, 1]

    # Add up column-integrated convective precipitation by
    # multiplying the normalized value by the cloud base
    # mass flux (propagate forward)
    with computation(FORWARD), interval(1, None):

        if cnvflg:
            rntot = rntot[0, 0, -1]

    # - Determine the evaporation of the convective precipitation
    #   and update the integrated convective precipitation
    # - Update state temperature and moisture to account for
    #   evaporation of convective precipitation
    # - Update column-integrated tendencies to account for
    #   evaporation of convective precipitation
    with computation(BACKWARD):

        with interval(-1, None):

            evef = 0.0
            dp = 0.0
            tem = 0.0
            tem1 = 0.0

            if k_idx <= kmax:

                deltv = 0.0
                delq = 0.0
                qevap = 0.0

                if cnvflg:
                    if k_idx > kb and k_idx < ktcon:
                        rn = rn + pwo * xmb * 0.001 * dt2

                if flg and k_idx < ktcon:

                    if islimsk == 1:
                        evef = edt * evfactl
                    else:
                        evef = edt * evfact

                    qcond = evef * (q1 - qeso) / (
                        1.0 + physcons.EL2ORC * qeso / (t1 ** 2)
                    )

                    dp = 1000.0 * del0

                    if rn > 0.0 and qcond < 0.0:

                        tem = dt2 * rn
                        tem = sqrt(tem)
                        tem = -0.32 * tem
                        tem = exp(tem)
                        qevap = -qcond * (1.0 - tem)
                        tem = 1000.0 * constants.GRAV / dp
                        qevap = min(qevap, tem)
                        delq2 = delqev + 0.001 * qevap * dp / constants.GRAV

                    if rn > 0.0 and qcond < 0.0 and delq2 > rntot:

                        qevap = 1000.0 * constants.GRAV * (rntot - delqev) / dp
                        flg = False

                    else:
                        flg = flg

                    if rn > 0.0 and qevap > 0.0:

                        tem = 0.001 * dp / constants.GRAV
                        tem1 = qevap * tem

                        if tem1 > rn:
                            qevap = rn / tem
                            rn = 0.0
                        else:
                            rn = rn - tem1

                        q1 = q1 + qevap
                        t1 = t1 - physcons.ELOCP * qevap
                        deltv = -physcons.ELOCP * qevap / dt2
                        delq = qevap / dt2

                        delqev = delqev + 0.001 * dp * qevap / constants.GRAV

                    else:
                        delqev = delqev

                    delqbar = delqbar + delq * dp / constants.GRAV
                    deltbar = deltbar + deltv * dp / constants.GRAV

        with interval(0, -1):

            rn = rn[0, 0, 1]
            flg = flg[0, 0, 1]
            delqev = delqev[0, 0, 1]
            delqbar = delqbar[0, 0, 1]
            deltbar = deltbar[0, 0, 1]

            if k_idx <= kmax:

                deltv = 0.0
                delq = 0.0
                qevap = 0.0

                if cnvflg:
                    if k_idx > kb and k_idx < ktcon:
                        rn = rn + pwo * xmb * 0.001 * dt2

                if flg and k_idx < ktcon:

                    if islimsk == 1:
                        evef = edt * evfactl
                    else:
                        evef = edt * evfact

                    qcond = evef * (q1 - qeso) / (
                        1.0 + physcons.EL2ORC * qeso / (t1 ** 2)
                    )

                    dp = 1000.0 * del0

                    if rn > 0.0 and qcond < 0.0:

                        tem = dt2 * rn
                        tem = sqrt(tem)
                        tem = -0.32 * tem
                        tem = exp(tem)
                        qevap = -qcond * (1.0 - tem)
                        tem = 1000.0 * constants.GRAV / dp
                        qevap = min(qevap, tem)
                        delq2 = delqev + 0.001 * qevap * dp / constants.GRAV

                    if rn > 0.0 and qcond < 0.0 and delq2 > rntot:

                        qevap = 1000.0 * constants.GRAV * (rntot - delqev) / dp
                        flg = False

                    else:
                        flg = flg

                    if rn > 0.0 and qevap > 0.0:

                        tem = 0.001 * dp / constants.GRAV
                        tem1 = qevap * tem

                        if tem1 > rn:
                            qevap = rn / tem
                            rn = 0.0
                        else:
                            rn = rn - tem1

                        q1 = q1 + qevap
                        t1 = t1 - physcons.ELOCP * qevap
                        deltv = -physcons.ELOCP * qevap / dt2
                        delq = qevap / dt2

                        delqev = delqev + 0.001 * dp * qevap / constants.GRAV

                    else:
                        delqev = delqev

                    delqbar = delqbar + delq * dp / constants.GRAV
                    deltbar = deltbar + deltv * dp / constants.GRAV

    with computation(FORWARD), interval(1, None):

        rn = rn[0, 0, -1]
        flg = flg[0, 0, -1]

    with computation(PARALLEL), interval(...):

        val1 = 0.0
        if cnvflg and k_idx >= kbcon and k_idx < ktcon:
            val1 = 1.0 + 675.0 * eta * xmb

        val2 = 0.2
        val3 = 0.0
        val4 = 1.0e6
        cnvc_log = 0.0

        cnvc_log = 0.04 * log10(
            val1, val4
        )  # 1.0e6 seems to get reasonable results, since val1 is on average ~50

        if cnvflg:

            if rn < 0.0 or flg == 0:
                rn = 0.0

            ktop = ktcon
            kbot = kbcon
            kcnv = 2

            if k_idx >= kbcon and k_idx < ktcon:

                # Calculate shallow convective cloud water
                cnvw = cnvwt * xmb * dt2

                # Calculate convective cloud cover, which is used when
                # pdf-based cloud fraction is used
                cnvc = min(cnvc_log, val2)
                cnvc = max(cnvc, val3)

            # Calculate the updraft convective mass flux
            if k_idx >= kb and k_idx < ktop:
                ud_mf = eta * xmb * dt2

            # Save the updraft convective mass flux at cloud top
            if k_idx == ktop - 1:
                dt_mf = ud_mf


def feedback_control_upd_trr(
    cnvflg: BoolField,
    k_idx: IntField,
    kmax: IntField,
    ktcon: IntField,
    del0: FloatField,
    delebar: FloatField,
    ctr: FloatField,
    dellae: FloatField,
    xmb: FloatField,
    qtr: FloatField,
    dt2: Float,
):
    with computation(PARALLEL), interval(...):
        delebar = 0.0

        if cnvflg and k_idx <= kmax and k_idx <= ktcon:

            ctr = ctr + dellae * xmb * dt2
            qtr = ctr

    # Propagate forward delebar values
    with computation(FORWARD):

        with interval(0, 1):

            dp = 0.0

            if cnvflg and k_idx <= kmax and k_idx <= ktcon:
                dp = 1000.0 * del0

                delebar = (
                    delebar + dellae * xmb * dp / constants.GRAV
                )  # Where does dp come from? Is it correct to use the last value at
                # line 1559 of samfshalcnv.F?

        with interval(1, None):

            if cnvflg:

                dp = 1000.0 * del0

                if k_idx <= kmax and k_idx <= ktcon:
                    delebar = delebar[0, 0, -1] + dellae * xmb * dp / constants.GRAV
                else:
                    delebar = delebar[0, 0, -1] + dellae * xmb * dp / constants.GRAV

    # Propagate backward delebar values
    with computation(BACKWARD), interval(0, -1):

        if cnvflg:
            delebar = delebar[0, 0, 1]


def store_aero_conc(
    cnvflg: BoolField,
    k_idx: IntField,
    kmax: IntField,
    rn: FloatField,
    qtr: FloatField,
    qaero: FloatField,
):
    with computation(PARALLEL), interval(...):

        # Store aerosol concentrations if present
        if cnvflg and rn > 0.0 and k_idx <= kmax:
            qtr = qaero


def separate_detrained_cw(
    cnvflg: BoolField,
    k_idx: IntField,
    kbcon: IntField,
    ktcon: IntField,
    dellal: FloatField,
    xmb: FloatField,
    t1: FloatField,
    qtr_1: FloatField,
    qtr_0: FloatField,
    dt2: Float,
    tcr: Float,
    tcrf: Float
):
    with computation(PARALLEL), interval(...):

        # Separate detrained cloud water into liquid and ice species as
        # a function of temperature only

        tem = 0.0
        val1 = 1.0
        val2 = 0.0
        tem1 = 0.0

        if cnvflg and k_idx >= kbcon and k_idx <= ktcon:

            tem = dellal * xmb * dt2
            tem1 = (tcr - t1) * tcrf
            tem1 = min(val1, tem1)
            tem1 = max(val2, tem1)

            if qtr_1 > -999.0:
                qtr_0 = qtr_0 + tem * tem1
                qtr_1 = qtr_1 + tem * (1.0 - tem1)
            else:
                qtr_0 = qtr_0 + tem


def tke_contribution(
    cnvflg: BoolField,
    k_idx: IntField,
    kb: IntField,
    ktop: IntField,
    eta: FloatField,
    xmb: FloatField,
    pfld: FloatField,
    t1: FloatField,
    sigmagfm: FloatField,
    qtr_ntk: FloatField,
):
    with computation(PARALLEL), interval(1, -1):

        tem = 0.0
        tem1 = 0.0
        ptem = 0.0

        # Include TKE contribution from shallow convection
        if cnvflg and k_idx > kb and k_idx < ktop:

            tem = 0.5 * (eta[0, 0, -1] + eta[0, 0, 0]) * xmb
            tem1 = pfld * 100.0 / (constants.RDGAS * t1)
            sigmagfm = max(sigmagfm, physcons.BETAW)
            ptem = tem / (sigmagfm * tem1)
            qtr_ntk = qtr_ntk + 0.5 * sigmagfm * ptem * ptem


class ScaleAwareMassFluxShallowConvection:
    """
    Fortran name is samfshalconv
    """
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        namelist: PhysicsConfig,
    ):
        pass

    def __call__(self):
        pass
