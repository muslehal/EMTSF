#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# =====================
# Configuration Section
# =====================

# Default values for parameters
DEFAULT_N_EPOCHS=100
DEFAULT_DSET="electricity"  # Replace with your actual dataset name
PYTHON_SCRIPT="main.py"  # Replace with your Python script name if different


# Fixed parameters
CONTEXT_POINTS=512
TARGET_POINTS_LIST=(192 336 720)

# =====================
# Argument Parsing
# =====================

# Function to display usage
usage() {
    echo "Usage: $0 [-e n_epochs] [-d dset] [--test] [-- additional arguments]"
    echo "  -e    Number of training epochs (default: $DEFAULT_N_EPOCHS)"
    echo "  -d    Dataset name (default: $DEFAULT_DSET)"
    echo "  --test    Run testing after training"
    exit 1
}

# Initialize with default values
N_EPOCHS=$DEFAULT_N_EPOCHS
DSET=$DEFAULT_DSET
RUN_TEST=False

ADDITIONAL_ARGS=""

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -e|--n_epochs)
            N_EPOCHS="$2"
            shift 2
            ;;
        -d|--dset)
            DSET="$2"
            shift 2
            ;;
        --test)
            RUN_TEST=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        --)
            shift
            break
            ;;
        *)
            ADDITIONAL_ARGS+=" $1"
            shift
            ;;
    esac
done

# =====================
# Execution Section
# =====================

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found!"
    exit 1
fi

# Create a base directory for saving models if not exists
SAVE_BASE_DIR="saved_models/${DSET}"
mkdir -p "${SAVE_BASE_DIR}"

# Loop over each target_points value and execute the Python script
for TARGET_POINTS in "${TARGET_POINTS_LIST[@]}"
do
    echo "-------------------------------------------"
    echo "Starting training with parameters:"
    echo "  Dataset         : ${DSET}"
    echo "  Epochs          : ${N_EPOCHS}"
    echo "  Context Points  : ${CONTEXT_POINTS}"
    echo "  Target Points   : ${TARGET_POINTS}"
    echo "  Additional Args : ${ADDITIONAL_ARGS}"
    echo "-------------------------------------------"

    # Define a unique save path for each target_points
    SAVE_PATH="${SAVE_BASE_DIR}/target_${TARGET_POINTS}"
    mkdir -p "${SAVE_PATH}"

    # Construct the model name based on parameters
    SAVE_MODEL_NAME="model_${DSET}_epochs${N_EPOCHS}_context${CONTEXT_POINTS}_target${TARGET_POINTS}"

    # Execute the Python training script with the specified parameters
    python "${PYTHON_SCRIPT}" \
        --n_epochs "${N_EPOCHS}" \
        --dset "${DSET}" \
        --context_points "${CONTEXT_POINTS}" \
        --target_points "${TARGET_POINTS}" \
        --save_path "${SAVE_PATH}" \
        --save_model_name "${SAVE_MODEL_NAME}" \
        $ADDITIONAL_ARGS

    echo "Training completed for target_points=${TARGET_POINTS}. Model saved to ${SAVE_PATH}/${SAVE_MODEL_NAME}.pth"

    # =====================
    # Testing Section
    # =====================
    if [ "$RUN_TEST" = true ]; then
        echo "-------------------------------------------"
        echo "Starting testing phase for target_points=${TARGET_POINTS}"
        echo "-------------------------------------------"

        python "${PYTHON_SCRIPT}" \
            --is_train 0 \
            --dset "${DSET}" \
            --context_points "${CONTEXT_POINTS}" \
            --target_points "${TARGET_POINTS}" \
            --save_path "${SAVE_PATH}" \
            --save_model_name "${SAVE_MODEL_NAME}" \
            $ADDITIONAL_ARGS

        echo "Testing completed for target_points=${TARGET_POINTS}. Results saved."
    fi

done

echo "All processes completed successfully!"
