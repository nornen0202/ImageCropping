from mobilecropnet_v4.data import DECISION_VOCAB, TARGET_AR_VOCAB, MobileCropNetV4BatchDataset, mobilecropnet_v4_collate, target_ar_id

__all__ = [
    "DECISION_VOCAB",
    "TARGET_AR_VOCAB",
    "MobileCropNetV4",
    "MobileCropNetV4BatchDataset",
    "compute_mobilecropnet_v4_loss",
    "mobilecropnet_v4_collate",
    "model_config_to_dict",
    "target_ar_id",
]


def __getattr__(name: str):
    if name in {"MobileCropNetV4", "compute_mobilecropnet_v4_loss", "model_config_to_dict"}:
        from mobilecropnet_v4.model import MobileCropNetV4, compute_mobilecropnet_v4_loss, model_config_to_dict

        exports = {
            "MobileCropNetV4": MobileCropNetV4,
            "compute_mobilecropnet_v4_loss": compute_mobilecropnet_v4_loss,
            "model_config_to_dict": model_config_to_dict,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
