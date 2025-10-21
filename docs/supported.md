# Supported Functions

Unless noted below, functions are expected to be overloaded and working correctly in numba.

## Not Supported

### `xraylib`

<div class="center-table" markdown>

|                       Function                       |         Reason          |
| :--------------------------------------------------: | :---------------------: |
|            `xraylib.AtomicNumberToSymbol`            |        `-> str`         |
|               `xraylib.Atomic_Factors`               |          None           | <!-- TODO(nin17): no reason not to do this -->
|                `xraylib.Bragg_angle`                 |    `Crystal_Struct*`    |
|               `xraylib.CompoundParser`               |     `compoundData*`     |
|             `xraylib.Crystal_AddCrystal`             |    `Crystal_Struct*`    |
|             `xraylib.Crystal_ArrayFree`              |     `Crystal_Array`     |
|             `xraylib.Crystal_ArrayInit`              |    `Crystal_Array*`     |
|                `xraylib.Crystal_Atom`                |     `Crystal_Atom`      |
|        `xraylib.Crystal_F_H_StructureFactor`         |                         |
|    `xraylib.Crystal_F_H_StructureFactor_Partial`     |                         |
|             `xraylib.Crystal_GetCrystal`             |                         |
|          `xraylib.Crystal_GetCrystalsList`           |                         |
|              `xraylib.Crystal_MakeCopy`              |                         |
|              `xraylib.Crystal_ReadFile`              |                         |
|               `xraylib.Crystal_Struct`               |                         |
|           `xraylib.Crystal_UnitCellVolume`           |                         |
|              `xraylib.Crystal_dSpacing`              |                         |
|         `xraylib.GetCompoundDataNISTByIndex`         |                         |
|         `xraylib.GetCompoundDataNISTByName`          |                         |
|          `xraylib.GetCompoundDataNISTList`           |                         |
|              `xraylib.GetErrorMessages`              |       Deprecated        |
|               `xraylib.GetExitStatus`                |       Deprecated        |
|         `xraylib.GetRadioNuclideDataByIndex`         |                         |
|         `xraylib.GetRadioNuclideDataByName`          |                         |
|          `xraylib.GetRadioNuclideDataList`           |                         |
|           `xraylib.Q_scattering_amplitude`           |                         |
|              `xraylib.Refractive_Index`              |     `-> xrlComplex`     |
|              `xraylib.SetErrorMessages`              |       Deprecated        |
|               `xraylib.SetExitStatus`                |       Deprecated        |
|                `xraylib.SetHardExit`                 |       Deprecated        |
|            `xraylib.SymbolToAtomicNumber`            |                         |
|                  `xraylib.XRayInit`                  |           ???           |  <!-- ???: what does it do? -->
|            `xraylib._SwigNonDynamicMeta`             | Private: Build Artefact |
|            `xraylib._swig_add_metaclass`             | Private: Build Artefact |
|                 `xraylib._swig_repr`                 | Private: Build Artefact |
|  `xraylib._swig_setattr_nondynamic_class_variable`   | Private: Build Artefact |
| `xraylib._swig_setattr_nondynamic_instance_variable` | Private: Build Artefact |
|             `xraylib.add_compound_data`              |                         |

</div>

### `xraylib_np`

All functions in the `xraylib_np` namespace are supported in nopython mode!!

Though the following deprecated and private functions are not implemented.

<div class="center-table" markdown>

|            Function            |   Reason   |
| :----------------------------: | :--------: |
| `xraylib_np.GetErrorMessages`  | Deprecated |
|   `xraylib_np.GetExitStatus`   | Deprecated |
| `xraylib_np.SetErrorMessages`  | Deprecated |
|   `xraylib_np.SetExitStatus`   | Deprecated |
|    `xraylib_np.SetHardExit`    | Deprecated |
|      `xraylib_np.XRL_1I`       |  Private   |
|      `xraylib_np.XRL_2II`      |  Private   |
|     `xraylib_np.XRayInit`      |    ???     |  <!-- ???: what does it do? -->
| `xraylib_np._AtomicLevelWidth` |  Private   |
|    `xraylib_np._AugerRate`     |  Private   |
|    `xraylib_np._AugerYield`    |  Private   |
| `xraylib_np._CosKronTransProb` |  Private   |
|    `xraylib_np._EdgeEnergy`    |  Private   |
|  `xraylib_np._ElectronConfig`  |  Private   |
|  `xraylib_np._ElementDensity`  |  Private   |
|    `xraylib_np._FluorYield`    |  Private   |
|    `xraylib_np._JumpFactor`    |  Private   |
|    `xraylib_np._LineEnergy`    |  Private   |
|     `xraylib_np._RadRate`      |  Private   |

</div>
