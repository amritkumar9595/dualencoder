MODEL='unet_cardiac'
DATASET_TYPE='cardiac'
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
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/calgary/acc_2x/'${LR}'/unet_calgary'
python train_unet_calgary.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --sample ${SAMPLE_RATE} --lr ${LEARNING_RATE} 
#python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}
ACC_FACTOR_2x



<<ACC_FACTOR_4x
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='4x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/calgary/acc_4x/'${LR}'/unet_calgary'
python train_unet_calgary.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --sample ${SAMPLE_RATE} --lr ${LEARNING_RATE} 
#python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}
ACC_FACTOR_4x


<<ACC_FACTOR_6x
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='6x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/calgary/acc_6x/'${LR}'/unet_calgary'
python train_unet_calgary.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --sample ${SAMPLE_RATE} --lr ${LEARNING_RATE} 
#python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}
ACC_FACTOR_6x


#<<ACC_FACTOR_8x
BATCH_SIZE=1
NUM_EPOCHS=150
DEVICE='cuda:0'
ACC_FACTOR='8x'
EXP_DIR='/media/student1/NewVolume/MR_Reconstruction/experiments/calgary/acc_8x/'${LR}'/unet_calgary'
python train_unet_calgary.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}  --sample ${SAMPLE_RATE} --lr ${LEARNING_RATE} 
#python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE}
#ACC_FACTOR_8x
