MODEL='dualencoder_calgary'   #  dautomap_calgary ,  unet_calgary ,  dualencoder_calgary
DATASET_TYPE='calgary'
BASE_PATH='/media/student1/NewVolume/MR_Reconstruction/experiments/calgary'
#USMASK_PATH=${BASE_PATH}'/Reconstruction-for-MRI/us_masks/'${DATASET_TYPE}
TARGET_PATH='/media/student1/NewVolume/MR_Reconstruction/datasets/calgary_singlecoil/Val'
NORMALIZED='Unnormalized'      # Normalized , Unnormalized
LR='lre4'



#<<ACC_FACTOR_4x
ACC_FACTOR='2x'

PREDICTIONS_PATH=${BASE_PATH}'/acc_'${ACC_FACTOR}'/'${LR}'/'${MODEL}'/results'
REPORT_PATH=${BASE_PATH}'/acc_'${ACC_FACTOR}'/'${LR}'/'${MODEL}

python measures_csv.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
#ACC_FACTOR_2x


<<ACC_FACTOR_4x
ACC_FACTOR='4x'

PREDICTIONS_PATH=${BASE_PATH}'/acc_'${ACC_FACTOR}'/'${LR}'/'${MODEL}'/results'
REPORT_PATH=${BASE_PATH}'/acc_'${ACC_FACTOR}'/'${LR}'/'${MODEL}

python measures_csv.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} 
ACC_FACTOR_4x
