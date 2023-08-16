fairseq-hydra-train \
	--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
	--config-name base_100h \
	common.user_dir=examples/data2vec \
	common.tensorboard_logdir=/home/work/workspace/fairseq/scripts/whale/outputs/$1 \
	common.log_file=/home/work/workspace/fairseq/scripts/whale/outputs/$1.log \
	task.data=/home/work/workspace/LibriSpeech/manifests \
	task.normalize=true \
	model.w2v_path=/home/work/workspace/models/data2vec_model/audio_base_ls.pt \
	checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/$1 \
	dataset.train_subset=ted2_train \
	dataset.valid_subset=ted2_dev \
	optimization.max_update=80000 \
	optimization.max_epoch=100 \
	optimization.lr=[0.00003] \
	criterion._name=ctc \
	+task.min_sample_size=16000 \
	+model.layer_type=trf_adp \
	+model.adp_num=1 \
	+model.adp_trf_idx=0:8
