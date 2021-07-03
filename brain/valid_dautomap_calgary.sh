MODEL='dautomap_calgary'
DATASET_TYPE='calgary'
BASE_PATH='/media/student1/NewVolume/MR_Reconstruction'
DATA_PATH='/media/student1/NewVolume/MR_Reconstruction/datasets/calgary_singlecoil/Val'
LR='lre4'

#<<ACC_FACTOR_2x
ACC_FACTOR='2x'
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'acc_${ACC_FACTOR}'/'${LR}'/'${MODEL}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'acc_${ACC_FACTOR}'/'${LR}'/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:0'

# echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} 
python valid_dautomap_calgary.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} 
#ACC_FACTOR_2x


<<ACC_FACTOR_4x
ACC_FACTOR='4x'
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'acc_${ACC_FACTOR}'/'${LR}'/'${MODEL}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'acc_${ACC_FACTOR}'/'${LR}'/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:0'

# echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} 
python valid_dautomap_calgary.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} 
ACC_FACTOR_4x

<<ACC_FACTOR_6x
ACC_FACTOR='6x'
CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'acc_${ACC_FACTOR}'/'${LR}'/'${MODEL}'/best_model.pt'
OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'acc_${ACC_FACTOR}'/'${LR}'/'${MODEL}'/results'
BATCH_SIZE=1
DEVICE='cuda:0'

# echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} 
python valid_dautomap_calgary.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} 
ACC_FACTOR_6x