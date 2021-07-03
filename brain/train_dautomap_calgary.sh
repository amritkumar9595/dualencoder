MODEL='dautomap_calgary'
DATASET_TYPE='calgary'
BASE_PATH='/media/student1/NewVolume/MR_Reconstruction'
SAMPLE_RATE=0.6
TRAIN_PATH='/media/student1/NewVolume/MR_Reconstruction/datasets/calgary_singlecoil/Train'
VALIDATION_PATH='/media/student1/NewVolume/MR_Reconstruction/datasets/calgary_singlecoil/Val'
LEARNING_RATE=0.0001
LR='lre4'



<<ACC_FACTOR_2x
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='2x'
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'acc_${ACC_FACTOR}'/'${LR}'/'${MODEL}
python train_dautomap_calgary.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --sample ${SAMPLE_RATE} --lr ${LEARNING_RATE}  
ACC_FACTOR_2x



<<ACC_FACTOR_4x
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='4x'
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'acc_${ACC_FACTOR}'/'${LR}'/'${MODEL}
python train_dautomap_calgary.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --sample ${SAMPLE_RATE} --lr ${LEARNING_RATE}  
ACC_FACTOR_4x


<<ACC_FACTOR_6x
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='6x'
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'acc_${ACC_FACTOR}'/'${LR}'/'${MODEL}
python train_dautomap_calgary.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --sample ${SAMPLE_RATE} --lr ${LEARNING_RATE} 
ACC_FACTOR_6x

#<<ACC_FACTOR_8x
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='8x'
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'acc_${ACC_FACTOR}'/'${LR}'/'${MODEL}
python train_dautomap_calgary.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --sample ${SAMPLE_RATE} --lr ${LEARNING_RATE}  
#ACC_FACTOR_8x