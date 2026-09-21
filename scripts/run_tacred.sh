echo "=========================================="
echo "Starting TACRED Experiments..."
echo "=========================================="

DATA_TYPE="tacred"
TOTAL_EPOCHS=20
MAX_LEN=512
BATCH_SIZE=3
ACC_STEP=3
LR_VALUE="3e-5"
RESULT_FILE="./temp_results.json"

KSHOT_PREFIXES=("1" "5" "16")

for prefix in "${KSHOT_PREFIXES[@]}"; do
  # Dataset-specific parameter matching for TACRED
  if [[ $prefix == "1" ]]; then
    reject_value="0.5"
    soft_value="0.3"
  elif [[ $prefix == "5" ]]; then
    reject_value="0.5"
    soft_value="0.3"
  elif [[ $prefix == "16" ]]; then
    reject_value="0.5"
    soft_value="0.4"
  fi

  RUN_COUNT=0
  for suffix in {5..5}; do
    kshot="${prefix}-${suffix}"

    current_num=${kshot#*-}
    next_num=$(( (current_num % 5) + 1 ))
    val_kshot="16-$next_num"

    echo "-----------------------------------------------------"
    echo "Run: $RUN_COUNT | Dataset: $DATA_TYPE | Train: $kshot | Val: $val_kshot"
    echo "LR: $LR_VALUE | Soft_Value: $soft_value | Reject_Value: $reject_value"
    echo "-----------------------------------------------------"

    python main.py --max_epochs=$TOTAL_EPOCHS --num_workers=8 \
      --model_name_or_path ./models/roberta-large \
      --config ./models/roberta-large \
      --accumulate_grad_batches $ACC_STEP \
      --batch_size $BATCH_SIZE \
      --dev_batch_size 16 \
      --data_type $DATA_TYPE \
      --data_dir dataset/$DATA_TYPE/k-shot/$kshot \
      --check_val_every_n_epoch 1 \
      --data_class WIKI80 \
      --max_seq_length $MAX_LEN \
      --model_class RobertaForPrompt \
      --t_lambda 0.001 \
      --litmodel_class BertLitModel \
      --task_name wiki80 \
      --lr $LR_VALUE \
      --use_template_words 0 \
      --init_type_words 0 \
      --init_answer_words 1 \
      --contrastive_ratio 1.2 \
      --contrastive_beta 0.7 \
      --use_contrastive \
      --pipeline_init \
      --output_dir output/$DATA_TYPE/k-shot/$kshot \
      --MVRE \
      --multi_viewer_num 3 \
      --mode train \
      --result_file $RESULT_FILE \
      --run_ids $RUN_COUNT \
      --class_learning_threshold $soft_value \
      --reject_lambda $reject_value

    let RUN_COUNT++
  done

  echo "All runs completed for K-shot: $prefix. Reporting to NNI..."
  python main.py --max_epochs=1 --num_workers=8 \
    --model_name_or_path ./models/roberta-large \
    --config ./models/roberta-large \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --dev_batch_size 8 \
    --data_dir dataset/$DATA_TYPE/k-shot/$kshot \
    --check_val_every_n_epoch 1 \
    --data_class WIKI80 \
    --max_seq_length $MAX_LEN \
    --model_class RobertaForPrompt \
    --t_lambda 0.001 \
    --litmodel_class BertLitModel \
    --task_name wiki80 \
    --lr $LR_VALUE \
    --use_template_words 0 \
    --init_type_words 0 \
    --init_answer_words 1 \
    --contrastive_ratio 1.2 \
    --contrastive_beta 0.7 \
    --use_contrastive \
    --pipeline_init \
    --output_dir output/$DATA_TYPE/k-shot/$kshot \
    --MVRE \
    --multi_viewer_num 3 \
    --mode average \
    --result_file $RESULT_FILE \
    --run_ids $RUN_COUNT \
    --data_type $DATA_TYPE
done