#exp_name=$1
model_name=$1
for lang in "de"; do
	exp_name="$model_name"_"$lang"
	fairseq-hydra-train \
		--config-dir /workspace/fairseq/examples/wav2vec/config/finetuning \
		--config-name base_100h \
		common.user_dir=examples/data2vec \
		common.wandb_project=sample_reweight \
		common.log_file=/workspace/fairseq/scripts/whale/outputs/$lang.log \
		task.data=/workspace/models/manifests \
		model.w2v_path=/workspace/models/wav2vec_model/$model_name \
		checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/$exp_name \
		dataset.train_subset=cv5.1_"$lang"_train \
		dataset.valid_subset=cv5.1_"$lang"_dev \
		dataset.max_tokens=3200000 \
		optimization.update_freq=[2] \
		criterion._name=ctc \
		distributed_training.distributed_world_size=1 \
		+task.min_sample_size=16000
done

#task.data=/home/work/workspace/LibriSpeech/manifests \
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
