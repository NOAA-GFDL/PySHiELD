from dataclasses import InitVar, dataclass, field, fields
from typing import Any, Dict, Mapping

import xarray as xr

import ndsl.dsl.gt4py_utils as gt_utils
from ndsl import GridSizer, Quantity, QuantityFactory
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from ndsl.dsl.typing import Float
from ndsl.types import NumpyModule


@dataclass()
class RadiationState:
    prsi: Quantity = field(
        metadata={
            "name": "interface_pressure",
            "dims": [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            "units": "Pa",
            "intent": "in",
        }
    )
    prsl: Quantity = field(
        metadata={
            "name": "layer_pressure",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    tlyr: Quantity = field(
        metadata={
            "name": "layer_air_temperature",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "degK",
            "intent": "in",
        }
    )
    tlvl: Quantity = field(
        metadata={
            "name": "level_air_temperature",
            "dims": [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            "units": "degK",
            "intent": "inout",
        }
    )
    tsfc: Quantity = field(
        metadata={
            "name": "surface_temperature",
            "dims": [X_DIM, Y_DIM],
            "units": "degK",
            "intent": "in",
        }
    )
    mu0: Quantity = field(
        metadata={
            "name": "cosine_zenith_angle",
            "dims": [X_DIM, Y_DIM],
            "units": "none",
            "intent": "inout",
        }
    )
    albedo: Quantity = field(
        metadata={
            "name": "surface_albedo",
            "dims": [X_DIM, Y_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    sfc_emis: Quantity = field(
        metadata={
            "name": "surface_emissivity",
            "dims": [X_DIM, Y_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    qvapor: Quantity = field(
        metadata={
            "name": "specific_humidity",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/kg",
            "intent": "in",
        }
    )
    qliquid: Quantity = field(
        metadata={
            "name": "cloud_water_mixing_ratio",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/kg",
            "intent": "in",
        }
    )
    qice: Quantity = field(
        metadata={
            "name": "cloud_ice_mixing_ratio",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/kg",
            "intent": "in",
        }
    )
    qo3mr: Quantity = field(
        metadata={
            "name": "ozone_mixing_ratio",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/kg",
            "intent": "in",
        }
    )
    qcld: Quantity = field(
        metadata={
            "name": "cloud_fraction",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "",
            "intent": "in",
        }
    )
    co2: Quantity = field(
        metadata={
            "name": "co2_concentration",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    clwp: Quantity = field(
        metadata={
            "name": "cloud_liquid_water_path",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    cip: Quantity = field(
        metadata={
            "name": "cloud_ice_path",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    clwr: Quantity = field(
        metadata={
            "name": "cloud_liquid_water_radius",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    cir: Quantity = field(
        metadata={
            "name": "cloud_ice_radius",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    flwu: Quantity = field(
        metadata={
            "name": "longwave_flux_up",
            "dims": [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            "units": "",
            "intent": "out",
        }
    )
    flwd: Quantity = field(
        metadata={
            "name": "longwave_flux_down",
            "dims": [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            "units": "",
            "intent": "out",
        }
    )
    fswu: Quantity = field(
        metadata={
            "name": "shortwave_flux_up",
            "dims": [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            "units": "",
            "intent": "out",
        }
    )
    fswd: Quantity = field(
        metadata={
            "name": "shortwave_flux_down",
            "dims": [X_DIM, Y_DIM, Z_INTERFACE_DIM],
            "units": "",
            "intent": "out",
        }
    )
    hrtlw: Quantity = field(
        metadata={
            "name": "longwave_heating_rate",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "",
            "intent": "out",
        }
    )
    hrtsw: Quantity = field(
        metadata={
            "name": "shortwave_heating_rate",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "",
            "intent": "out",
        }
    )
    quantity_factory: InitVar[QuantityFactory]
    np_like: InitVar[NumpyModule]

    def __post_init__(
        self,
        quantity_factory: QuantityFactory,
        np_like: NumpyModule,
    ):
        self._np = np_like
        self._nz = quantity_factory.sizer.nz
        self._nx = quantity_factory.sizer.nx
        self._ny = quantity_factory.sizer.ny

    @classmethod
    def init_zeros(
        cls,
        quantity_factory,
        np_like: NumpyModule,
    ) -> "RadiationState":
        initial_arrays = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                initial_arrays[_field.name] = quantity_factory.zeros(
                    _field.metadata["dims"],
                    _field.metadata["units"],
                    dtype=Float,
                )
        return cls(
            **initial_arrays,
            quantity_factory=quantity_factory,
            np_like=np_like,
        )

    @classmethod
    def init_from_storages(
        cls,
        storages: Mapping[str, Any],
        sizer: GridSizer,
        quantity_factory: QuantityFactory,
        np_like: NumpyModule,
    ) -> "RadiationState":
        inputs: Dict[str, Quantity] = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                dims = _field.metadata["dims"]
                quantity = Quantity(
                    storages[_field.name],
                    dims,
                    _field.metadata["units"],
                    origin=sizer.get_origin(dims),
                    extent=sizer.get_extent(dims),
                )
                inputs[_field.name] = quantity
        return cls(
            **inputs,
            quantity_factory=quantity_factory,
            np_like=np_like,
        )

    @property
    def xr_dataset(self):
        data_vars = {}
        for name, field_info in self.__dataclass_fields__.items():
            if name not in ["quantity_factory", "np_like"]:
                if issubclass(field_info.type, Quantity):
                    dims = [
                        f"{dim_name}_{name}" for dim_name in field_info.metadata["dims"]
                    ]
                    data_vars[name] = xr.DataArray(
                        gt_utils.asarray(getattr(self, name).data),
                        dims=dims,
                        attrs={
                            "long_name": field_info.metadata["name"],
                            "units": field_info.metadata.get("units", "unknown"),
                        },
                    )
        return xr.Dataset(data_vars=data_vars)

    # @property
    def to_rterrtmgp_xr(self) -> xr.Dataset:
        """
        creates an xarray dataset of the state for RTE-RRTMGP
        Halos and padding points are removed, and horizontal axes are combined
        and dimension attributes are also changed to columns and levels or layers
        to fit pyRTE-RRTMGP's api
        For example, prsl's shape is changed from (npx, npy, npz) to (nx * ny, nz)
        with dimensions (column, layer)
        """
        data_vars = {}
        for name, field_info in self.__dataclass_fields__.items():
            if name not in ["quantity_factory", "np_like"]:
                if field_info.metadata["intent"] != "out":
                    if issubclass(field_info.type, Quantity):
                        dims = []
                        slice_list = []
                        ndims = len(field_info.metadata["dims"])
                        nz = self._nz
                        for dim_name in field_info.metadata["dims"]:
                            # dims.append(f"{dim_name}_{name}")
                            if dim_name == "z_interface":
                                slice_list.append(self._np.s_[:])
                                nz = self._nz + 1
                                dims.append("level")
                            elif dim_name == "z":
                                slice_list.append(self._np.s_[:-1])
                                dims.append("layer")
                            elif "INTERFACE" in dim_name:
                                slice_list.append(self._np.s_[3:-3])
                            else:
                                slice_list.append(self._np.s_[3:-4])
                        # We have to reshape to get the max 2D shape rterrtmgp expects:
                        if ndims == 3:  # x-y-z array:
                            newshape = (-1, nz)
                            dims.insert(0, "column")
                        elif ndims == 2:  # x-y array:
                            newshape = (-1,)  # noqa
                            dims.insert(0, "column")
                        elif ndims == 1:  # z-array
                            newshape == (nz,)
                        else:
                            raise NotImplementedError(
                                (
                                    "RadiationShape doesn't support more than 3D "
                                    f"arrays, {dim_name} has {ndims} axes"
                                )
                            )
                        data_vars[name] = xr.DataArray(
                            gt_utils.asarray(getattr(self, name).data)[
                                tuple(slice_list)
                            ].reshape(newshape),
                            dims=dims,
                            attrs={
                                "long_name": field_info.metadata["name"],
                            },
                        )
        return xr.Dataset(data_vars=data_vars)
