MODEL='dualencoder_cardiac'   #  dautomap_cardiac ,  unet_cardiac ,  dualencoder_cardiac
DATASET_TYPE='cardiac'
BASE_PATH='/media/student1/NewVolume/MR_Reconstruction'
#USMASK_PATH=${BASE_PATH}'/Reconstruction-for-MRI/us_masks/'${DATASET_TYPE}

<<ACC_FACTOR_2x
ACC_FACTOR='2x'
TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/validation'
PREDICTIONS_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}


python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH}
ACC_FACTOR_2x


#<<ACC_FACTOR_4x
ACC_FACTOR='4x'
TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/validation'
PREDICTIONS_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}


python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH}
#ACC_FACTOR_4x

<<ACC_FACTOR_6x
ACC_FACTOR='6x'
TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/validation'
PREDICTIONS_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}


python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH}
ACC_FACTOR_6x

<<ACC_FACTOR_8x
ACC_FACTOR='8x'
TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/validation'
PREDICTIONS_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}


python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH}
ACC_FACTOR_8x
