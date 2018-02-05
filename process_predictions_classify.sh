#!/bin/bash
echo "Run Binary Segmentation + Classification Scripts"

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol1_rev1/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol1_rev1/

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol2_rev1/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol2_rev1/

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol3_rev1/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol3_rev1/

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol4_rev1/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol4_rev1/

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol5_rev1/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol5_rev1/

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol6_rev2/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol6_rev2/

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol7_rev2/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol7_rev2/

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/base_10um_rev1/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/base_10um_rev1/

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/base_rbc_half_rev1/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/base_rbc_half_rev1/

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/base_wbc_full_rev1/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/base_wbc_full_rev1/