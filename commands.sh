conda create -n ssl_env python=3.7
conda activate ssl_env
pip install -r req.txt

# run data_parsing/make_capture24.py to generate the capture24 dataset (processed)
# run python mtl.py runtime.gpu=0 data.data_root=capture24_100hz_w10_o0 runtime.is_epoch_data=True data=ssl_capture_24 task=all task.scale=false augmentation=all   model=resnet data.batch_subject_num=5 dataloader=ten_sec
  # data=capture24_30s \

python downstream_task_evaluation.py \
  data=capture24_10s \
  data.data_root=/Users/lilykoffman/Documents/ssl-wearables/capture24_100hz_w10_o0 \
  data.Y_path=/Users/lilykoffman/Documents/ssl-wearables/capture24_100hz_w10_o0/Y.npy \
  report_root=/Users/lilykoffman/Documents/ssl-wearables/capture24_100hz_w10_o0/reports \
  evaluation=all \
  evaluation.flip_net_path=/Users/lilykoffman/Documents/ssl-wearables/mtl_best.mdl \
  gpu=-1 \
  model=resnet 
