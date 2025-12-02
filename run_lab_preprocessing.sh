#!/bin/bash

#SBATCH -A MST109178              # Account name/project number
#SBATCH -J CRBSI_lab              # Job name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00		  # Kill the job after 12 hours
#SBATCH -p ngs372G                 # Partition Name (CPU queue)
#SBATCH -c 8                      # Number of cores (8 cores recommended for data processing)
#SBATCH --mem=360G                 # Memory per node
#SBATCH -o crbsi_lab_%j.out       # Path to the standard output file
#SBATCH -e crbsi_lab_%j.err       # Path to the standard error output file
#SBATCH --mail-user=fortunemavuso4@gmail.com    # Email for notifications
#SBATCH --mail-type=BEGIN,END     # Send email when job begins and ends

# Print job information
echo "========================================="
echo "CRBSI Data Preprocessing Job"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""


python crbsi_preprocessing_lab_based.py \
    --mimic_path /staging/biology/${USER}/AI_Intelligent_Medicine/final_project/Public_Mimi_iv_dataset/MIMIC-IV-3_1 \
    --output_path /staging/biology/${USER}/AI_Intelligent_Medicine/final_project/Public_Mimi_iv_dataset/CRBSI_processed_lab \
    --feature_window 48 \
    --prediction_window 72 \
    --survival_window 168 \
    --chunk_size 1000000
