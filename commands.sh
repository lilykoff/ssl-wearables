## steps to create conda environment and install requirements 
conda create -n ssl_env python=3.7
conda activate ssl_env
pip install -r req.txt

# to generate the capture24 dataset (processed) 
python data_parsing/make_capture24.py 
# run python mtl.py runtime.gpu=0 data.data_root=capture24_100hz_w10_o0 runtime.is_epoch_data=True data=ssl_capture_24 task=all task.scale=false augmentation=all   model=resnet data.batch_subject_num=5 dataloader=ten_sec
  # data=capture24_30s \

### model fine tuning on capture 24 ### 
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python downstream_task_evaluation.py \
  data=capture24_10s \
  data.data_root=/Users/lilykoffman/Documents/ssl-wearables/capture24_100hz_w10_o0 \
  data.Y_path=/Users/lilykoffman/Documents/ssl-wearables/capture24_100hz_w10_o0/Y.npy \
  report_root=/Users/lilykoffman/Documents/ssl-wearables/capture24_100hz_w10_o0/reports \
  evaluation=all \
  evaluation.flip_net_path=/Users/lilykoffman/Documents/ssl-wearables/mtl_best.mdl \
  gpu=-1 \
  model=resnet 

# make small subset of capture24 for testing locally 
python data_parsing/make_small.py 

# run downstream task evaluation locally with small subset of capture24
python downstream_task_evaluation.py \
  data=capture24_10s \
  data.data_root=/Users/lilykoffman/Documents/ssl-wearables/capture24_100hz_w10_o0_small \
  report_root=/Users/lilykoffman/Documents/ssl-wearables/tmp_reports/ \
  evaluation=all \
  evaluation.flip_net_path=/Users/lilykoffman/Documents/ssl-wearables/mtl_best.mdl \
  gpu=-1 \
  model=resnet \
  evaluation.num_epoch=2 \
  evaluation.patience=1 \
  num_split=1

python downstream_task_evaluation_v2.py \
  data=capture24_10s \
  data.data_root=/Users/lilykoffman/Documents/ssl-wearables/capture24_100hz_w10_o0_small \
  report_root=/Users/lilykoffman/Documents/ssl-wearables/tmp_reports/ \
  evaluation=all \
  evaluation.flip_net_path=/Users/lilykoffman/Documents/ssl-wearables/mtl_best.mdl \
  gpu=-1 \
  model=resnet \
  evaluation.num_epoch=2 \
  evaluation.patience=1 \
  num_split=1

python data_parsing/make_iu.py
python predict_unlabeled.py # generates predictions.csv with predictions for unlabeled data in IU 
