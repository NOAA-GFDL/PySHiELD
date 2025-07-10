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
            "intent": "inout",
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
            "intent": "inout",
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
            "name": "surface_air_temperature",
            "dims": [X_DIM, Y_DIM],
            "units": "degK",
            "intent": "inout",
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
        }
    )
    qliquid: Quantity = field(
        metadata={
            "name": "cloud_water_mixing_ratio",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qice: Quantity = field(
        metadata={
            "name": "cloud_ice_mixing_ratio",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    # qrain: Quantity = field(
    #     metadata={
    #         "name": "rain_mixing_ratio",
    #         "dims": [X_DIM, Y_DIM, Z_DIM],
    #         "units": "kg/kg",
    #         "intent": "inout",
    #     }
    # )
    # qsnow: Quantity = field(
    #     metadata={
    #         "name": "snow_mixing_ratio",
    #         "dims": [X_DIM, Y_DIM, Z_DIM],
    #         "units": "kg/kg",
    #         "intent": "inout",
    #     }
    # )
    # qgraupel: Quantity = field(
    #     metadata={
    #         "name": "graupel_mixing_ratio",
    #         "dims": [X_DIM, Y_DIM, Z_DIM],
    #         "units": "kg/kg",
    #         "intent": "inout",
    #     }
    # )
    qo3mr: Quantity = field(
        metadata={
            "name": "ozone_mixing_ratio",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qcld: Quantity = field(
        metadata={
            "name": "cloud_fraction",
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
    # delp: Quantity = field(
    #     metadata={
    #         "name": "pressure_thickness_of_atmospheric_layer",
    #         "dims": [X_DIM, Y_DIM, Z_DIM],
    #         "units": "Pa",
    #         "intent": "inout",
    #     }
    # )
    # delz: Quantity = field(
    #     metadata={
    #         "name": "vertical_thickness_of_atmospheric_layer",
    #         "dims": [X_DIM, Y_DIM, Z_DIM],
    #         "units": "m",
    #         "intent": "inout",
    #     }
    # )
    # delprsi: Quantity = field(
    #     metadata={
    #         "name": "model_level_pressure_thickness_in_physics",
    #         "dims": [X_DIM, Y_DIM, Z_DIM],
    #         "units": "Pa",
    #         "intent": "inout",
    #     }
    # )
    # dz: Quantity = field(
    #     metadata={
    #         "name": "geopotential_height_thickness",
    #         "dims": [X_DIM, Y_DIM, Z_DIM],
    #         "units": "m",
    #         "intent": "inout",
    #     }
    # )
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
                ).data
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
                        newshape = (-1) # noqa
                        dims.insert(0, "column")
                    elif ndims == 1:  # z-array
                        newshape == (nz)
                    else:
                        raise NotImplementedError(
                            (
                                "RadiationShape doesn't support more than 3D arrays, "
                                f"{dim_name} has {ndims} axes"
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
