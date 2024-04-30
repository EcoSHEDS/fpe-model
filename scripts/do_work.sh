#!/bin/bash

annotations_root="../data/processed"

# Loop through sites
# sites=("West Whately_01171005")
# sites=("West Brook Lower_01171090" "West Brook Reservoir_01171020" "West Brook Upper_01171030")
# sites=("Avery Brook_River Left_01171000")
sites=("Avery Brook_Bridge_01171000")

for i in "${!sites[@]}"; do
    annotations_dir="${annotations_root}/${sites[$i]}/FLOW_CFS/kernighan-lin_test.vary_n_train_annot"
    val_pairs="${annotations_dir}/pairs-val.csv"
    test_pairs="${annotations_dir}/pairs-test.csv"
    # Get all files with pairs-train in the name in this directory
    train_pairs_files=($(ls "${annotations_dir}" | grep "pairs-train"))
    # Sort the files by the number of annotations (i.e. by NUM in pairs-train-NUM.csv)
    train_pairs_files=($(for file in "${train_pairs_files[@]}"; do echo "$file"; done | sort -t'-' -k3 -n))
    # Loop through the sorted files
    for j in "${!train_pairs_files[@]}"; do
        # Skip the first 2 iterations
        if [ $j -lt 2 ]; then
            continue
        fi
        # If iteration is greater than 3, skip
        if [ $j -gt 3 ]; then
            continue
        fi
        output_dir="../results/vary_annot_frac.kernighan-lin_test/${sites[$i]}-$(basename "${train_pairs_files[$j]}" .csv)"
        train_pairs="${annotations_dir}/${train_pairs_files[$j]}"
        # Define the command-line arguments for train_ranking_model.py
        train_args=("--site" "${sites[$i]}"
                    "-o" "${output_dir}"
                    "--image-root-dir" "/home/amritagupta/ssdprivate/data/Streamflow/fpe_stations/${sites[$i]}/FLOW_CFS/"
                    "--train-data-file" "${train_pairs}"
                    "--val-data-file" "${val_pairs}"
                    "--test-data-file" "${test_pairs}"
                    "--gpu" 0)
        # Run train_ranking_model.py with the specified arguments
        python train_ranking_model.py "${train_args[@]}"

        # # Get the best model from the training results
        best_model_path="${output_dir}/train_ranking_model_${sites[$i]}_1/checkpoints/best_model.ckpt"
        inference_data_file="/home/amritagupta/ssdprivate/data/Streamflow/fpe_stations/${sites[$i]}/FLOW_CFS/images.csv"
        inference_image_root_dir="/home/amritagupta/ssdprivate/data/Streamflow/fpe_stations/${sites[$i]}/FLOW_CFS/"
        inference_output_root_dir="${output_dir}/inference"
        inference_args=("--inference-data-file" "${inference_data_file}"
                        "--inference-image-root-dir" "${inference_image_root_dir}"
                        "--inference-output-root-dir" "${inference_output_root_dir}"
                        "--ckpt-path" "${best_model_path}"
                        "--gpu" 0)
        # Run inference_ranking_model.py with the specified arguments
        python inference_ranking_model.py "${inference_args[@]}"
        
    done
done