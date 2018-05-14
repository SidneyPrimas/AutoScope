#!/bin/bash
echo "Run Binary Segmentation + Classification Scripts"

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol1/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol1/

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol2/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol2/

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol3/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol3/

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol4/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol4/

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol5/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol5/

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol6/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol6/

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol7/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol7/

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol8/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol8/

python urine_particles/process_urine_segment.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol9/
python urine_particles/process_urine_classify.py -r ./urine_particles/data/clinical_experiment/prediction_folder/sol9/