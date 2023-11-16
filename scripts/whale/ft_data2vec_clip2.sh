fairseq-hydra-train \
	--config-dir /workspace/fairseq_new/examples/wav2vec/config/finetuning \
	--config-name base_960h \
	common.wandb_project=lm2am_distill \
	common.log_file=/workspace/fairseq_new/scripts/whale/outputs/$1.log \
	distributed_training.distributed_world_size=4 \
	task.data=/DB/manifests \
	model.w2v_path=/workspace/models/wav2vec_small.pt \
	checkpoint.save_dir=/workspace/fairseq_new/scripts/whale/outputs/$1 \
	dataset.max_tokens=3200000 \
	optimization.update_freq=[2] \
	criterion._name=clip3 \
	+criterion.decoder=conv \
	+criterion.lm=gpt2 \
	+criterion.lm_decay=0.2


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
