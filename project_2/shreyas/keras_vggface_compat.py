import sys
import types


def patch_keras_for_vggface() -> None:
    import keras.utils as keras_utils

    if not hasattr(keras_utils, "layer_utils"):
        try:
            from keras.src.utils import layer_utils as modern_layer_utils

            keras_utils.layer_utils = modern_layer_utils
            sys.modules.setdefault(
                "keras.utils.layer_utils",
                modern_layer_utils,
            )
        except Exception:
            pass

    if "keras.utils.data_utils" not in sys.modules:
        data_utils_mod = types.ModuleType("keras.utils.data_utils")
        if hasattr(keras_utils, "get_file"):
            data_utils_mod.get_file = keras_utils.get_file
        sys.modules["keras.utils.data_utils"] = data_utils_mod

    if "keras.engine" not in sys.modules:
        sys.modules["keras.engine"] = types.ModuleType("keras.engine")

    if "keras.engine.topology" not in sys.modules:
        topology_mod = types.ModuleType("keras.engine.topology")
        if hasattr(keras_utils, "get_source_inputs"):
            topology_mod.get_source_inputs = keras_utils.get_source_inputs
        sys.modules["keras.engine.topology"] = topology_mod
