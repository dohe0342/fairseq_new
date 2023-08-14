for i in {11..22}; do
	fairseq-hydra-train \
		--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
		--config-name base_100h \
		common.user_dir=examples/data2vec \
		common.tensorboard_logdir=/home/work/workspace/fairseq/scripts/whale/outputs/$1_$i \
		common.log_file=/home/work/workspace/fairseq/scripts/whale/outputs/$1_$i.log \
		task.data=/home/work/workspace/LibriSpeech/manifests \
		task.normalize=true \
		model.w2v_path=/home/work/workspace/models/data2vec_model/audio_base_ls.pt \
		checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/$1_$i \
		dataset.train_subset=ted2_train \
		dataset.valid_subset=ted2_dev \
		optimization.max_update=0 \
		optimization.max_epoch=30 \
		optimization.lr=[0.0001] \
		criterion._name=ctc \
		+task.min_sample_size=16000 \
		+model.layer_type=trf_adp \
		+model.adp_num=1 \
		+model.adp_trf_idx=$i:12
done

: <<'END'
fairseq-hydra-train \
	--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
	--config-name base_100h \
	common.user_dir=examples/data2vec \
	common.tensorboard_logdir=/home/work/workspace/fairseq/scripts/whale/outputs/$2 \
	task.data=/home/work/workspace/LibriSpeech/manifests \
	task.normalize=true \
	model.w2v_path=/home/work/workspace/models/data2vec_model/audio_base_ls.pt \
	checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/$2 \
	dataset.train_subset=cv_train \
	dataset.valid_subset=cv_dev \
	optimization.max_update=20000 \
	optimization.lr=[0.00003] \
	criterion._name=prompt \
	+task.min_sample_size=16000
#model.freeze_finetune_updates=20000 \
END
