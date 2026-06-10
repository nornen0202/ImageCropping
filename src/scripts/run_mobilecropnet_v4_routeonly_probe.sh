#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?variant required}"
RUN_NAME="${2:?run name required}"

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3}"
OUT_BASE="${OUT_BASE:-artifacts/mobilecropnet_v4/quality_first_20260427/route_smokes/sstk_uctr_supporttarget}"
RUN_DIR="${OUT_BASE}/${RUN_NAME}"
STATUS_JSON="${RUN_DIR}/route_smoke_status.json"

TRAIN_JSONL="${TRAIN_JSONL:-data/SSTK/Full_10000/artifacts/training_labels_uctr_stage3_product_ar_260422_hashsplit_80_10_10_260422/train/train_conditional_detr_batch.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/SSTK/Full_10000/artifacts/training_labels_uctr_stage3_product_ar_260422_hashsplit_80_10_10_260422/val/train_conditional_detr_batch.jsonl}"

cd "${ROOT}"
mkdir -p "${RUN_DIR}"

write_status() {
  local phase="$1"
  local state="$2"
  "${PYTHON_BIN}" - <<PY
import json
import time
from pathlib import Path

run_dir = Path(${RUN_DIR@Q})
payload = {
    "phase": ${phase@Q},
    "state": ${state@Q},
    "variant": ${VARIANT@Q},
    "run_name": ${RUN_NAME@Q},
    "run_dir": str(run_dir),
    "last_update_time_unix": time.time(),
    "train_log": str(run_dir / "train.log"),
    "config_path": str(run_dir / "config.json"),
    "metrics_path": str(run_dir / "metrics.json"),
    "summary_path": str(run_dir / "summary.json"),
    "best_checkpoint": str(run_dir / "best.pt"),
    "latest_checkpoint": str(run_dir / "last.pt"),
}
(run_dir / "route_smoke_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

trap 'write_status failed failed' ERR

declare -a PROFILE_ARGS
declare -a ROUTE_INPUT_ARGS=(--route_use_candidate_context)
declare -a ROUTE_STRUCTURAL_ARGS
declare -a EXTRA_TRAIN_ARGS=()
STOK_INIT_CHECKPOINT="${STOK_INIT_CHECKPOINT:-artifacts/mobilecropnet_v4/quality_first_20260427/route_smokes/sstk_uctr_supporttarget/mcn-q-b448-intprior-stok-noref-r2/best.pt}"
case "${VARIANT}" in
  cnv2b448_routeonly_fullval_margin)
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-14.0}"
    ROUTE_FOCAL_GAMMA="${ROUTE_FOCAL_GAMMA:-1.45}"
    ROUTE_HIERARCHY_WEIGHT="${ROUTE_HIERARCHY_WEIGHT:-1.2}"
    ROUTE_CARDINALITY_WEIGHT="${ROUTE_CARDINALITY_WEIGHT:-0.6}"
    ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT="${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT:-0.28}"
    ROUTE_OBJECT_SINGLE_MARGIN="${ROUTE_OBJECT_SINGLE_MARGIN:-0.18}"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-12.0}"
    PROFILE_ARGS=(
      --model_profile quality_cnv2b_448
      --epochs "${EPOCHS:-5}"
      --batch_size "${BATCH_SIZE:-8}"
      --lr "${LR:-0.000015}"
      --max_train_rows "${MAX_TRAIN_ROWS:-22000}"
      --max_val_rows "${MAX_VAL_ROWS:-3200}"
      --limit_train_steps "${LIMIT_TRAIN_STEPS:-1000}"
      --limit_val_steps "${LIMIT_VAL_STEPS:-400}"
      --image_tensor_cache_size "${IMAGE_TENSOR_CACHE_SIZE:-96}"
      --seed "${SEED:-20260430}"
    )
    ;;
  cnv2b448_routeimage_fullval_margin)
    ROUTE_INPUT_ARGS=(--route_image_only)
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-14.0}"
    ROUTE_FOCAL_GAMMA="${ROUTE_FOCAL_GAMMA:-1.45}"
    ROUTE_HIERARCHY_WEIGHT="${ROUTE_HIERARCHY_WEIGHT:-1.2}"
    ROUTE_CARDINALITY_WEIGHT="${ROUTE_CARDINALITY_WEIGHT:-0.6}"
    ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT="${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT:-0.28}"
    ROUTE_OBJECT_SINGLE_MARGIN="${ROUTE_OBJECT_SINGLE_MARGIN:-0.18}"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-12.0}"
    PROFILE_ARGS=(
      --model_profile quality_cnv2b_448
      --epochs "${EPOCHS:-5}"
      --batch_size "${BATCH_SIZE:-8}"
      --lr "${LR:-0.000015}"
      --max_train_rows "${MAX_TRAIN_ROWS:-22000}"
      --max_val_rows "${MAX_VAL_ROWS:-3200}"
      --limit_train_steps "${LIMIT_TRAIN_STEPS:-1000}"
      --limit_val_steps "${LIMIT_VAL_STEPS:-400}"
      --image_tensor_cache_size "${IMAGE_TENSOR_CACHE_SIZE:-96}"
      --seed "${SEED:-20260431}"
    )
    ;;
  cnv2b448_routeimage_hierdecode_margin)
    ROUTE_INPUT_ARGS=(--route_image_only)
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-14.0}"
    ROUTE_FOCAL_GAMMA="${ROUTE_FOCAL_GAMMA:-1.45}"
    ROUTE_HIERARCHY_WEIGHT="${ROUTE_HIERARCHY_WEIGHT:-1.5}"
    ROUTE_CARDINALITY_WEIGHT="${ROUTE_CARDINALITY_WEIGHT:-0.8}"
    ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT="${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT:-0.28}"
    ROUTE_OBJECT_SINGLE_MARGIN="${ROUTE_OBJECT_SINGLE_MARGIN:-0.18}"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-14.0}"
    ROUTE_STRUCTURAL_ARGS=(
      --route_aux_heads
      --route_decode_mode hierarchical
      --route_decode_kind_weight "${ROUTE_DECODE_KIND_WEIGHT:-1.0}"
      --route_decode_cardinality_weight "${ROUTE_DECODE_CARDINALITY_WEIGHT:-0.8}"
    )
    PROFILE_ARGS=(
      --model_profile quality_cnv2b_448
      --epochs "${EPOCHS:-5}"
      --batch_size "${BATCH_SIZE:-8}"
      --lr "${LR:-0.000015}"
      --max_train_rows "${MAX_TRAIN_ROWS:-22000}"
      --max_val_rows "${MAX_VAL_ROWS:-3200}"
      --limit_train_steps "${LIMIT_TRAIN_STEPS:-1000}"
      --limit_val_steps "${LIMIT_VAL_STEPS:-400}"
      --image_tensor_cache_size "${IMAGE_TENSOR_CACHE_SIZE:-96}"
      --seed "${SEED:-20260432}"
    )
    ;;
  cnv2b448_routeonly_hierdecode_margin)
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-14.0}"
    ROUTE_FOCAL_GAMMA="${ROUTE_FOCAL_GAMMA:-1.45}"
    ROUTE_HIERARCHY_WEIGHT="${ROUTE_HIERARCHY_WEIGHT:-1.5}"
    ROUTE_CARDINALITY_WEIGHT="${ROUTE_CARDINALITY_WEIGHT:-0.8}"
    ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT="${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT:-0.28}"
    ROUTE_OBJECT_SINGLE_MARGIN="${ROUTE_OBJECT_SINGLE_MARGIN:-0.18}"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-14.0}"
    ROUTE_STRUCTURAL_ARGS=(
      --route_aux_heads
      --route_decode_mode hierarchical
      --route_decode_kind_weight "${ROUTE_DECODE_KIND_WEIGHT:-1.0}"
      --route_decode_cardinality_weight "${ROUTE_DECODE_CARDINALITY_WEIGHT:-0.8}"
    )
    PROFILE_ARGS=(
      --model_profile quality_cnv2b_448
      --epochs "${EPOCHS:-5}"
      --batch_size "${BATCH_SIZE:-8}"
      --lr "${LR:-0.000015}"
      --max_train_rows "${MAX_TRAIN_ROWS:-22000}"
      --max_val_rows "${MAX_VAL_ROWS:-3200}"
      --limit_train_steps "${LIMIT_TRAIN_STEPS:-1000}"
      --limit_val_steps "${LIMIT_VAL_STEPS:-400}"
      --image_tensor_cache_size "${IMAGE_TENSOR_CACHE_SIZE:-96}"
      --seed "${SEED:-20260433}"
    )
    ;;
  cnv2b448_routeonly_auxfine_nomargin)
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-12.0}"
    ROUTE_FOCAL_GAMMA="${ROUTE_FOCAL_GAMMA:-1.25}"
    ROUTE_HIERARCHY_WEIGHT="${ROUTE_HIERARCHY_WEIGHT:-1.2}"
    ROUTE_CARDINALITY_WEIGHT="${ROUTE_CARDINALITY_WEIGHT:-0.6}"
    ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT="${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT:-0.0}"
    ROUTE_OBJECT_SINGLE_MARGIN="${ROUTE_OBJECT_SINGLE_MARGIN:-0.18}"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-10.0}"
    ROUTE_STRUCTURAL_ARGS=(
      --route_aux_heads
      --route_decode_mode fine
    )
    PROFILE_ARGS=(
      --model_profile quality_cnv2b_448
      --epochs "${EPOCHS:-5}"
      --batch_size "${BATCH_SIZE:-8}"
      --lr "${LR:-0.000015}"
      --max_train_rows "${MAX_TRAIN_ROWS:-22000}"
      --max_val_rows "${MAX_VAL_ROWS:-3200}"
      --limit_train_steps "${LIMIT_TRAIN_STEPS:-1000}"
      --limit_val_steps "${LIMIT_VAL_STEPS:-400}"
      --image_tensor_cache_size "${IMAGE_TENSOR_CACHE_SIZE:-96}"
      --seed "${SEED:-20260434}"
    )
    ;;
  cnv2b448_routeimage_auxfine_nomargin)
    ROUTE_INPUT_ARGS=(--route_image_only)
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-12.0}"
    ROUTE_FOCAL_GAMMA="${ROUTE_FOCAL_GAMMA:-1.25}"
    ROUTE_HIERARCHY_WEIGHT="${ROUTE_HIERARCHY_WEIGHT:-1.2}"
    ROUTE_CARDINALITY_WEIGHT="${ROUTE_CARDINALITY_WEIGHT:-0.6}"
    ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT="${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT:-0.0}"
    ROUTE_OBJECT_SINGLE_MARGIN="${ROUTE_OBJECT_SINGLE_MARGIN:-0.18}"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-10.0}"
    ROUTE_STRUCTURAL_ARGS=(
      --route_aux_heads
      --route_decode_mode fine
    )
    PROFILE_ARGS=(
      --model_profile quality_cnv2b_448
      --epochs "${EPOCHS:-5}"
      --batch_size "${BATCH_SIZE:-8}"
      --lr "${LR:-0.000015}"
      --max_train_rows "${MAX_TRAIN_ROWS:-22000}"
      --max_val_rows "${MAX_VAL_ROWS:-3200}"
      --limit_train_steps "${LIMIT_TRAIN_STEPS:-1000}"
      --limit_val_steps "${LIMIT_VAL_STEPS:-400}"
      --image_tensor_cache_size "${IMAGE_TENSOR_CACHE_SIZE:-96}"
      --seed "${SEED:-20260435}"
    )
    ;;
  cnv2b448_intprior_routehead_auxfine_fromstok)
    ROUTE_INPUT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --route_use_subject_box_features
      --route_use_subject_spatial_token
      --policy_use_subject_prior
      --proposal_use_subject_prior
    )
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-12.0}"
    ROUTE_FOCAL_GAMMA="${ROUTE_FOCAL_GAMMA:-1.20}"
    ROUTE_HIERARCHY_WEIGHT="${ROUTE_HIERARCHY_WEIGHT:-1.2}"
    ROUTE_CARDINALITY_WEIGHT="${ROUTE_CARDINALITY_WEIGHT:-0.6}"
    ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT="${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT:-0.0}"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-12.0}"
    ROUTE_HEAD_DEPTH="${ROUTE_HEAD_DEPTH:-1}"
    ROUTE_HEAD_HIDDEN_MULT="${ROUTE_HEAD_HIDDEN_MULT:-0.5}"
    ROUTE_HEAD_DROPOUT="${ROUTE_HEAD_DROPOUT:-0.0}"
    ROUTE_STRUCTURAL_ARGS=(
      --route_aux_heads
      --route_decode_mode fine
    )
    EXTRA_TRAIN_ARGS=(
      --init_checkpoint "${STOK_INIT_CHECKPOINT}"
      --subject_box_target_source support_latent
      --subject_box_head
      --subject_box_spatial_head
      --subject_box_spatial_output spatial
      --subject_box_spatial_box_mode mask_moment_regress
      --subject_box_spatial_mix_bias -1.0
      --subject_box_coord_space content
      --use_pred_subject_box_as_prior
      --detach_pred_subject_prior
      --policy_score_head
      --trainable_module_prefixes route_head,route_kind_head,route_cardinality_head
    )
    PROFILE_ARGS=(
      --model_profile quality_cnv2b_448
      --epochs "${EPOCHS:-5}"
      --batch_size "${BATCH_SIZE:-8}"
      --lr "${LR:-0.00012}"
      --max_train_rows "${MAX_TRAIN_ROWS:-22000}"
      --max_val_rows "${MAX_VAL_ROWS:-3200}"
      --limit_train_steps "${LIMIT_TRAIN_STEPS:-1000}"
      --limit_val_steps "${LIMIT_VAL_STEPS:-400}"
      --image_tensor_cache_size "${IMAGE_TENSOR_CACHE_SIZE:-128}"
      --seed "${SEED:-20260440}"
    )
    ;;
  cnv2b448_intprior_routehead_hierdecode_fromstok)
    ROUTE_INPUT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --route_use_subject_box_features
      --route_use_subject_spatial_token
      --policy_use_subject_prior
      --proposal_use_subject_prior
    )
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-12.0}"
    ROUTE_FOCAL_GAMMA="${ROUTE_FOCAL_GAMMA:-1.20}"
    ROUTE_HIERARCHY_WEIGHT="${ROUTE_HIERARCHY_WEIGHT:-1.2}"
    ROUTE_CARDINALITY_WEIGHT="${ROUTE_CARDINALITY_WEIGHT:-0.6}"
    ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT="${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT:-0.0}"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-12.0}"
    ROUTE_HEAD_DEPTH="${ROUTE_HEAD_DEPTH:-1}"
    ROUTE_HEAD_HIDDEN_MULT="${ROUTE_HEAD_HIDDEN_MULT:-0.5}"
    ROUTE_HEAD_DROPOUT="${ROUTE_HEAD_DROPOUT:-0.0}"
    ROUTE_STRUCTURAL_ARGS=(
      --route_aux_heads
      --route_decode_mode hierarchical
      --route_decode_kind_weight "${ROUTE_DECODE_KIND_WEIGHT:-1.0}"
      --route_decode_cardinality_weight "${ROUTE_DECODE_CARDINALITY_WEIGHT:-0.8}"
    )
    EXTRA_TRAIN_ARGS=(
      --init_checkpoint "${STOK_INIT_CHECKPOINT}"
      --subject_box_target_source support_latent
      --subject_box_head
      --subject_box_spatial_head
      --subject_box_spatial_output spatial
      --subject_box_spatial_box_mode mask_moment_regress
      --subject_box_spatial_mix_bias -1.0
      --subject_box_coord_space content
      --use_pred_subject_box_as_prior
      --detach_pred_subject_prior
      --policy_score_head
      --trainable_module_prefixes route_head,route_kind_head,route_cardinality_head
    )
    PROFILE_ARGS=(
      --model_profile quality_cnv2b_448
      --epochs "${EPOCHS:-5}"
      --batch_size "${BATCH_SIZE:-8}"
      --lr "${LR:-0.00012}"
      --max_train_rows "${MAX_TRAIN_ROWS:-22000}"
      --max_val_rows "${MAX_VAL_ROWS:-3200}"
      --limit_train_steps "${LIMIT_TRAIN_STEPS:-1000}"
      --limit_val_steps "${LIMIT_VAL_STEPS:-400}"
      --image_tensor_cache_size "${IMAGE_TENSOR_CACHE_SIZE:-128}"
      --seed "${SEED:-20260441}"
    )
    ;;
  clipcnvb384_routeimage_auxfine_nomargin)
    ROUTE_INPUT_ARGS=(--route_image_only)
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-12.0}"
    ROUTE_FOCAL_GAMMA="${ROUTE_FOCAL_GAMMA:-1.20}"
    ROUTE_HIERARCHY_WEIGHT="${ROUTE_HIERARCHY_WEIGHT:-1.2}"
    ROUTE_CARDINALITY_WEIGHT="${ROUTE_CARDINALITY_WEIGHT:-0.6}"
    ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT="${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT:-0.0}"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-10.0}"
    ROUTE_STRUCTURAL_ARGS=(
      --route_aux_heads
      --route_decode_mode fine
    )
    PROFILE_ARGS=(
      --model_profile quality_clip_cnvb_384
      --epochs "${EPOCHS:-5}"
      --batch_size "${BATCH_SIZE:-8}"
      --lr "${LR:-0.000012}"
      --max_train_rows "${MAX_TRAIN_ROWS:-22000}"
      --max_val_rows "${MAX_VAL_ROWS:-3200}"
      --limit_train_steps "${LIMIT_TRAIN_STEPS:-1000}"
      --limit_val_steps "${LIMIT_VAL_STEPS:-400}"
      --image_tensor_cache_size "${IMAGE_TENSOR_CACHE_SIZE:-96}"
      --seed "${SEED:-20260436}"
    )
    ;;
  siglipb384_routeimage_auxfine_nomargin)
    ROUTE_INPUT_ARGS=(--route_image_only)
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-12.0}"
    ROUTE_FOCAL_GAMMA="${ROUTE_FOCAL_GAMMA:-1.20}"
    ROUTE_HIERARCHY_WEIGHT="${ROUTE_HIERARCHY_WEIGHT:-1.2}"
    ROUTE_CARDINALITY_WEIGHT="${ROUTE_CARDINALITY_WEIGHT:-0.6}"
    ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT="${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT:-0.0}"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-10.0}"
    ROUTE_STRUCTURAL_ARGS=(
      --route_aux_heads
      --route_decode_mode fine
    )
    PROFILE_ARGS=(
      --model_profile quality_siglipb_384
      --epochs "${EPOCHS:-5}"
      --batch_size "${BATCH_SIZE:-5}"
      --lr "${LR:-0.000008}"
      --max_train_rows "${MAX_TRAIN_ROWS:-22000}"
      --max_val_rows "${MAX_VAL_ROWS:-3200}"
      --limit_train_steps "${LIMIT_TRAIN_STEPS:-900}"
      --limit_val_steps "${LIMIT_VAL_STEPS:-400}"
      --image_tensor_cache_size "${IMAGE_TENSOR_CACHE_SIZE:-48}"
      --seed "${SEED:-20260437}"
    )
    ;;
  clipcnvb384_routeimage_headonly_auxfine_nomargin)
    ROUTE_INPUT_ARGS=(--route_image_only)
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-10.0}"
    ROUTE_FOCAL_GAMMA="${ROUTE_FOCAL_GAMMA:-1.15}"
    ROUTE_HIERARCHY_WEIGHT="${ROUTE_HIERARCHY_WEIGHT:-1.0}"
    ROUTE_CARDINALITY_WEIGHT="${ROUTE_CARDINALITY_WEIGHT:-0.5}"
    ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT="${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT:-0.0}"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-10.0}"
    ROUTE_STRUCTURAL_ARGS=(
      --route_aux_heads
      --route_decode_mode fine
    )
    EXTRA_TRAIN_ARGS=(
      --freeze_backbone
      --trainable_module_prefixes route_head,route_kind_head,route_cardinality_head
    )
    PROFILE_ARGS=(
      --model_profile quality_clip_cnvb_384
      --epochs "${EPOCHS:-5}"
      --batch_size "${BATCH_SIZE:-16}"
      --lr "${LR:-0.00035}"
      --max_train_rows "${MAX_TRAIN_ROWS:-22000}"
      --max_val_rows "${MAX_VAL_ROWS:-3200}"
      --limit_train_steps "${LIMIT_TRAIN_STEPS:-800}"
      --limit_val_steps "${LIMIT_VAL_STEPS:-220}"
      --image_tensor_cache_size "${IMAGE_TENSOR_CACHE_SIZE:-128}"
      --seed "${SEED:-20260438}"
    )
    ;;
  siglipb384_routeimage_headonly_auxfine_nomargin)
    ROUTE_INPUT_ARGS=(--route_image_only)
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-10.0}"
    ROUTE_FOCAL_GAMMA="${ROUTE_FOCAL_GAMMA:-1.15}"
    ROUTE_HIERARCHY_WEIGHT="${ROUTE_HIERARCHY_WEIGHT:-1.0}"
    ROUTE_CARDINALITY_WEIGHT="${ROUTE_CARDINALITY_WEIGHT:-0.5}"
    ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT="${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT:-0.0}"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-10.0}"
    ROUTE_STRUCTURAL_ARGS=(
      --route_aux_heads
      --route_decode_mode fine
    )
    EXTRA_TRAIN_ARGS=(
      --freeze_backbone
      --trainable_module_prefixes route_head,route_kind_head,route_cardinality_head
    )
    PROFILE_ARGS=(
      --model_profile quality_siglipb_384
      --epochs "${EPOCHS:-5}"
      --batch_size "${BATCH_SIZE:-8}"
      --lr "${LR:-0.00025}"
      --max_train_rows "${MAX_TRAIN_ROWS:-22000}"
      --max_val_rows "${MAX_VAL_ROWS:-3200}"
      --limit_train_steps "${LIMIT_TRAIN_STEPS:-800}"
      --limit_val_steps "${LIMIT_VAL_STEPS:-220}"
      --image_tensor_cache_size "${IMAGE_TENSOR_CACHE_SIZE:-64}"
      --seed "${SEED:-20260439}"
    )
    ;;
  eva02b448_routeonly_hier)
    PROFILE_ARGS=(
      --model_profile quality_eva02b_448
      --epochs "${EPOCHS:-4}"
      --batch_size "${BATCH_SIZE:-4}"
      --lr "${LR:-0.000008}"
      --max_train_rows "${MAX_TRAIN_ROWS:-18000}"
      --max_val_rows "${MAX_VAL_ROWS:-3600}"
      --limit_train_steps "${LIMIT_TRAIN_STEPS:-650}"
      --limit_val_steps "${LIMIT_VAL_STEPS:-160}"
      --image_tensor_cache_size "${IMAGE_TENSOR_CACHE_SIZE:-64}"
      --seed "${SEED:-20260427}"
    )
    ;;
  dinov2b518_routeonly_hier)
    PROFILE_ARGS=(
      --model_profile quality_dinov2b_subject_518
      --epochs "${EPOCHS:-4}"
      --batch_size "${BATCH_SIZE:-3}"
      --lr "${LR:-0.000005}"
      --max_train_rows "${MAX_TRAIN_ROWS:-16000}"
      --max_val_rows "${MAX_VAL_ROWS:-3200}"
      --limit_train_steps "${LIMIT_TRAIN_STEPS:-560}"
      --limit_val_steps "${LIMIT_VAL_STEPS:-140}"
      --image_tensor_cache_size "${IMAGE_TENSOR_CACHE_SIZE:-48}"
      --seed "${SEED:-20260428}"
    )
    ;;
  swinv2b384_routeonly_hier)
    PROFILE_ARGS=(
      --model_profile quality_swinv2b_384
      --epochs "${EPOCHS:-4}"
      --batch_size "${BATCH_SIZE:-5}"
      --lr "${LR:-0.000008}"
      --max_train_rows "${MAX_TRAIN_ROWS:-18000}"
      --max_val_rows "${MAX_VAL_ROWS:-3600}"
      --limit_train_steps "${LIMIT_TRAIN_STEPS:-650}"
      --limit_val_steps "${LIMIT_VAL_STEPS:-160}"
      --image_tensor_cache_size "${IMAGE_TENSOR_CACHE_SIZE:-64}"
      --seed "${SEED:-20260429}"
    )
    ;;
  *)
    echo "unsupported route-only probe variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

declare -a ZERO_HEAD_ARGS=(
  --score_weight 0.0
  --listwise_weight 0.0
  --pairwise_weight 0.0
  --explicit_pairwise_weight 0.0
  --positive_weight 0.0
  --risk_weight 0.0
  --top1_risk_weight 0.0
  --top_return_weight 0.0
  --macro_weight 0.0
  --checklist_class_weight 0.0
  --checklist_applicability_weight 0.0
  --detail_score_weight 0.0
  --why_tag_weight 0.0
  --decision_weight 0.0
  --policy_score_weight 0.0
  --delta_weight 0.0
  --proposal_weight 0.0
  --proposal_subject_weight 0.0
  --subject_proposal_align_weight 0.0
  --subject_box_weight 0.0
  --subject_box_valid_weight 0.0
  --generated_proposal_align_weight 0.0
)

write_status environment running
{
  date -u +"start_utc=%Y-%m-%dT%H:%M:%SZ"
  nvidia-smi --query-gpu=name,memory.total,driver_version,utilization.gpu,memory.used --format=csv,noheader
  "${PYTHON_BIN}" - <<'PY'
import torch
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available()
print("device_name", torch.cuda.get_device_name(0))
PY
  "${PYTHON_BIN}" -m py_compile src/mobilecropnet_v4/model.py src/mobilecropnet_v4/data.py src/scripts/train_mobilecropnet_v4.py
  bash -n src/scripts/run_mobilecropnet_v4_routeonly_probe.sh
} | tee "${RUN_DIR}/environment.log"

write_status train running
"${PYTHON_BIN}" src/scripts/train_mobilecropnet_v4.py \
  --train_jsonl "${TRAIN_JSONL}" \
  --val_jsonl "${VAL_JSONL}" \
  --project_root . \
  --output_dir "${RUN_DIR}" \
  "${PROFILE_ARGS[@]}" \
  --weight_decay "${WEIGHT_DECAY:-0.0}" \
  --num_workers "${NUM_WORKERS:-2}" \
  --selection_metric route_balanced \
  "${ZERO_HEAD_ARGS[@]}" \
  --route_weight "${ROUTE_WEIGHT:-12.0}" \
  --route_balanced_ce \
  --route_focal_gamma "${ROUTE_FOCAL_GAMMA:-1.15}" \
  --route_hierarchy_weight "${ROUTE_HIERARCHY_WEIGHT:-0.8}" \
  --route_cardinality_weight "${ROUTE_CARDINALITY_WEIGHT:-0.4}" \
  --route_object_single_margin_weight "${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT:-0.0}" \
  --route_object_single_margin "${ROUTE_OBJECT_SINGLE_MARGIN:-0.18}" \
  "${ROUTE_INPUT_ARGS[@]}" \
  "${ROUTE_STRUCTURAL_ARGS[@]}" \
  "${EXTRA_TRAIN_ARGS[@]}" \
  --route_head_depth "${ROUTE_HEAD_DEPTH:-3}" \
  --route_head_hidden_mult "${ROUTE_HEAD_HIDDEN_MULT:-1.5}" \
  --route_head_dropout "${ROUTE_HEAD_DROPOUT:-0.10}" \
  --train_route_balanced_sampler \
  --train_route_balanced_sampler_max_weight "${ROUTE_SAMPLER_MAX_WEIGHT:-8.0}" \
  --amp \
  --device cuda \
  --gpu_usage_sample_interval 20 \
  --progress_log_interval 50 \
  2>&1 | tee "${RUN_DIR}/train.log"

write_status summarize running
"${PYTHON_BIN}" - "${RUN_DIR}" <<'PY'
import json
import sys
import time
from pathlib import Path

run_dir = Path(sys.argv[1])
metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")) if (run_dir / "metrics.json").exists() else []
best = max(metrics, key=lambda row: row.get("selection_score", -999.0)) if metrics else {}
payload = {
    "state": "completed",
    "phase": "completed",
    "run_name": run_dir.name,
    "run_dir": str(run_dir),
    "best_epoch": best.get("epoch"),
    "best_selection_score": best.get("selection_score"),
    "best_val": best.get("val"),
    "history_length": len(metrics),
    "last_update_time_unix": time.time(),
}
(run_dir / "route_smoke_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(run_dir / "route_smoke_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
