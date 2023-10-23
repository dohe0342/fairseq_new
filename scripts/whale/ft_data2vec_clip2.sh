fairseq-hydra-train \
	--config-dir /workspace/fairseq_new/examples/wav2vec/config/finetuning \
	--config-name base_100h \
	common.log_file=/workspace/fairseq_new/scripts/whale/outputs/$1.log \
	common.fp16_no_flatten_grads=true \
	task.data=/DB/manifests \
	model.w2v_path=/workspace/models/wav2vec_small.pt \
	model.freeze_finetune_updates=10000 \
	checkpoint.save_dir=/workspace/fairseq_new/scripts/whale/outputs/$1 \
	dataset.max_tokens=3200000 \
	optimization.update_freq=[2] \
	optimization.debug_param_names=true \
	criterion._name=clip2

#common.wandb_project=lm2am_distill \
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
