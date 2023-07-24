git pull

fairseq-hydra-train \
	--config-dir /home/work/workspace/fairseq/examples/wav2vec/config/finetuning \
	--config-name base_100h \
	common.user_dir=examples/data2vec \
	task.data=/home/work/workspace/LibriSpeech/manifests \
	task.normalize=true \
	model.w2v_path=/home/work/workspace/models/data2vec_model/audio_base_ls.pt \
	model.layerdrop=0.0 \
	criterion._name=interctc \
	checkpoint.save_dir=/home/work/workspace/fairseq/scripts/whale/outputs/$1 \
	checkpoint.best_checkpoint_metric=wer_11 \
	optimization.lr=[0.00009]
