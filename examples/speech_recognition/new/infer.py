#!/usr/bin/env python -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ast
import hashlib
import logging
import os
import shutil
import sys
import re
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import editdistance
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from examples.speech_recognition.new.decoders.decoder_config import (
    DecoderConfig,
    FlashlightDecoderConfig,
)
from examples.speech_recognition.new.decoders.decoder import Decoder
from fairseq import checkpoint_utils, distributed_utils, progress_bar, tasks, utils
from fairseq.data.data_utils import post_process
from fairseq.dataclass.configs import (
    CheckpointConfig,
    CommonConfig,
    CommonEvalConfig,
    DatasetConfig,
    DistributedTrainingConfig,
    FairseqDataclass,
)
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.logging.progress_bar import BaseProgressBar
from fairseq.models.fairseq_model import FairseqModel
from omegaconf import OmegaConf
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,  
    MultiheadAttention,
    RelPositionalEncoding,
    SamePad,        
    TransposeLast,
)

import hydra
from hydra.core.config_store import ConfigStore

from transformers import (
    GPT2Tokenizer, 
    GPT2Model, 
    BertTokenizer,  
    BertModel,
    AutoTokenizer, 
    MistralModel,
)

import matplotlib.pyplot as plt

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config_path = Path(__file__).resolve().parent / "conf"


@dataclass
class DecodingConfig(DecoderConfig, FlashlightDecoderConfig):
    unique_wer_file: bool = field(
        default=False,
        metadata={"help": "If set, use a unique file for storing WER"},
    )
    results_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "If set, write hypothesis and reference sentences into this directory"
        },
    )


@dataclass
class InferConfig(FairseqDataclass):
    task: Any = None
    decoding: DecodingConfig = DecodingConfig()
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    is_ax: bool = field(
        default=False,
        metadata={
            "help": "if true, assumes we are using ax for tuning and returns a tuple for ax to consume"
        },
    )


def reset_logging():
    root = logging.getLogger()
    for handler in root.handlers:
        root.removeHandler(handler)
    root.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)


class InferenceProcessor:
    cfg: InferConfig

    def __init__(self, cfg: InferConfig) -> None:
        self.cfg = cfg
        self.task = tasks.setup_task(cfg.task)

        models, saved_cfg = self.load_model_ensemble()

        ckpt = torch.load(self.cfg.common_eval.path)
        try: criterion = ckpt['criterion']
        except: criterion = None
        if criterion is not None and 'prompt' in criterion:
            self.prompt = criterion['prompt']
            logger.info('Using prompt...')
        else:
            self.prompt = None
        
        if 1:
            conv_layers = [(768, 5, 2)] * 3
            mode = "layer_norm"
            dropout = 0.0

            def block(
                n_in,
                n_out,
                k,
                stride,
                groups=1,
                is_layer_norm=False,
                is_group_norm=False,
                conv_bias=False,
            ):
                def make_conv():
                    conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias, groups=groups)
                    nn.init.kaiming_normal_(conv.weight)
                    return conv

                assert (
                    is_layer_norm and is_group_norm
                ) == False, "layer norm and group norm are exclusive"

                if is_layer_norm:
                    return nn.Sequential(
                        make_conv(),
                        nn.Dropout(p=dropout),
                        nn.Sequential(
                            TransposeLast(),
                            Fp32LayerNorm(dim, elementwise_affine=True),
                            TransposeLast(),
                        ),
                        nn.GELU(),
                    )
                elif is_group_norm:
                    return nn.Sequential(
                        make_conv(),
                        nn.Dropout(p=dropout),
                        Fp32GroupNorm(dim, dim, affine=True),
                        nn.GELU(),
                    )
                else:
                    return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())
                
            self.lm_decoder = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3, "invalid conv definition: " + str(cl)
                (dim, k, stride) = cl 

                self.lm_decoder.append(
                    block(
                        dim,
                        dim,
                        k,
                        stride,
                        is_layer_norm=(mode == "layer_norm"),
                        is_group_norm=(mode == "default") and i == 0,
                        conv_bias=False,
                    )
                )
            
            self.lm_decoder = self.lm_decoder.to('cuda')
            self.lm_decoder.load_state_dict(criterion, strict=False)
            for n, p in self.lm_decoder.named_parameters():
                print(n)
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.lm = GPT2Model.from_pretrained('gpt2').eval()
            self.lm = self.lm.to('cuda')
        else:
            self.lm_decoder = None

        del ckpt, criterion
        ### LOAD ADAPTER ####
        ckpt_obj = checkpoint_utils.load_checkpoint_to_cpu(self.cfg.common_eval.path)
        if "adapter" in ckpt_obj:
            target_lang = self.cfg.dataset.gen_subset.split(":")[0]
            assert target_lang in ckpt_obj["adapter"]
            
            logger.info(f">>> LOADING ADAPTER: {target_lang}")
            ft_obj = ckpt_obj["adapter"][target_lang]
            ft_model = ft_obj["model"]
            cdevice = models[0].w2v_encoder.proj.weight.device
            cdtype = models[0].w2v_encoder.proj.weight.dtype
            ft_proj_out, ft_proj_in = ft_model["w2v_encoder.proj.weight"].shape
            ft_proj = torch.nn.Linear(ft_proj_in, ft_proj_out, bias=True)
            ft_proj.to(device=cdevice, dtype=cdtype)
            models[0].w2v_encoder.proj = ft_proj
            with torch.no_grad():
                for kk, vv in models[0].named_parameters():
                    if kk in ft_model:
                        vv.copy_(ft_model[kk])
            self.task.load_state_dict(ft_obj["task_state"])
            # overwrite gen_subset with master config
            self.cfg.dataset.gen_subset = re.sub('^[\w-]+:', saved_cfg['task']['multi_corpus_keys']+":", self.cfg.dataset.gen_subset)
        self.models = models
        self.saved_cfg = saved_cfg
        self.tgt_dict = self.task.target_dictionary

        self.task.load_dataset(
            self.cfg.dataset.gen_subset,
            task_cfg=saved_cfg.task,
        )
        self.generator = Decoder(cfg.decoding, self.tgt_dict)
        self.gen_timer = StopwatchMeter()
        self.wps_meter = TimeMeter()
        self.num_sentences = 0
        self.total_errors = 0
        self.total_length = 0

        self.hypo_words_file = None
        self.hypo_units_file = None
        self.ref_words_file = None
        self.ref_units_file = None

        self.progress_bar = self.build_progress_bar()

    def __enter__(self) -> "InferenceProcessor":
        if self.cfg.decoding.results_path is not None:
            self.hypo_words_file = self.get_res_file("hypo.word")
            self.hypo_units_file = self.get_res_file("hypo.units")
            self.ref_words_file = self.get_res_file("ref.word")
            self.ref_units_file = self.get_res_file("ref.units")
        return self

    def __exit__(self, *exc) -> bool:
        if self.cfg.decoding.results_path is not None:
            self.hypo_words_file.close()
            self.hypo_units_file.close()
            self.ref_words_file.close()
            self.ref_units_file.close()
        return False

    def __iter__(self) -> Any:
        for sample in self.progress_bar:
            if not self.cfg.common.cpu:
                sample = utils.move_to_cuda(sample)

            # Happens on the last batch.
            if "net_input" not in sample:
                continue
            yield sample

    def log(self, *args, **kwargs):
        self.progress_bar.log(*args, **kwargs)

    def print(self, *args, **kwargs):
        self.progress_bar.print(*args, **kwargs)

    def get_res_file(self, fname: str) -> None:
        fname = os.path.join(self.cfg.decoding.results_path, fname)
        if self.data_parallel_world_size > 1:
            fname = f"{fname}.{self.data_parallel_rank}"
        return open(fname, "w", buffering=1)

    def merge_shards(self) -> None:
        """Merges all shard files into shard 0, then removes shard suffix."""

        shard_id = self.data_parallel_rank
        num_shards = self.data_parallel_world_size

        if self.data_parallel_world_size > 1:

            def merge_shards_with_root(fname: str) -> None:
                fname = os.path.join(self.cfg.decoding.results_path, fname)
                logger.info("Merging %s on shard %d", fname, shard_id)
                base_fpath = Path(f"{fname}.0")
                with open(base_fpath, "a") as out_file:
                    for s in range(1, num_shards):
                        shard_fpath = Path(f"{fname}.{s}")
                        with open(shard_fpath, "r") as in_file:
                            for line in in_file:
                                out_file.write(line)
                        shard_fpath.unlink()
                shutil.move(f"{fname}.0", fname)

            dist.barrier()  # ensure all shards finished writing
            if shard_id == (0 % num_shards):
                merge_shards_with_root("hypo.word")
            if shard_id == (1 % num_shards):
                merge_shards_with_root("hypo.units")
            if shard_id == (2 % num_shards):
                merge_shards_with_root("ref.word")
            if shard_id == (3 % num_shards):
                merge_shards_with_root("ref.units")
            dist.barrier()

    def optimize_model(self, model: FairseqModel) -> None:
        model.make_generation_fast_()
        if self.cfg.common.fp16:
            model.half()
        if not self.cfg.common.cpu:
            model.cuda()

    def load_model_ensemble(self) -> Tuple[List[FairseqModel], FairseqDataclass]:
        arg_overrides = ast.literal_eval(self.cfg.common_eval.model_overrides)
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(self.cfg.common_eval.path, separator="\\"),
            arg_overrides=arg_overrides,
            task=self.task,
            suffix=self.cfg.checkpoint.checkpoint_suffix,
            strict=(self.cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=self.cfg.checkpoint.checkpoint_shard_count,
        )
        for model in models:
            self.optimize_model(model)
        return models, saved_cfg

    def get_dataset_itr(self, disable_iterator_cache: bool = False) -> None:
        return self.task.get_batch_iterator(
            dataset=self.task.dataset(self.cfg.dataset.gen_subset),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=(sys.maxsize, sys.maxsize),
            ignore_invalid_inputs=self.cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=self.cfg.common.seed,
            num_shards=self.data_parallel_world_size,
            shard_id=self.data_parallel_rank,
            num_workers=self.cfg.dataset.num_workers,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
        ).next_epoch_itr(shuffle=False)

    def build_progress_bar(
        self,
        epoch: Optional[int] = None,
        prefix: Optional[str] = None,
        default_log_format: str = "tqdm",
    ) -> BaseProgressBar:
        return progress_bar.progress_bar(
            iterator=self.get_dataset_itr(),
            log_format=self.cfg.common.log_format,
            log_interval=self.cfg.common.log_interval,
            epoch=epoch,
            prefix=prefix,
            tensorboard_logdir=self.cfg.common.tensorboard_logdir,
            default_log_format=default_log_format,
        )

    @property
    def data_parallel_world_size(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 1
        return distributed_utils.get_data_parallel_world_size()

    @property
    def data_parallel_rank(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 0
        return distributed_utils.get_data_parallel_rank()

    def process_sentence(
        self,
        sample: Dict[str, Any],
        hypo: Dict[str, Any],
        sid: int,
        batch_id: int,
    ) -> Tuple[int, int]:
        speaker = None  # Speaker can't be parsed from dataset.

        if "target_label" in sample:
            toks = sample["target_label"]
        else:
            toks = sample["target"]
        toks = toks[batch_id, :]

        # Processes hypothesis.
        hyp_pieces = self.tgt_dict.string(hypo["tokens"].int().cpu())
        if "words" in hypo:
            hyp_words = " ".join(hypo["words"])
        else:
            hyp_words = post_process(hyp_pieces, self.cfg.common_eval.post_process)

        # Processes target.
        target_tokens = utils.strip_pad(toks, self.tgt_dict.pad())
        tgt_pieces = self.tgt_dict.string(target_tokens.int().cpu())
        tgt_words = post_process(tgt_pieces, self.cfg.common_eval.post_process)

        if self.cfg.decoding.results_path is not None:
            print(f"{hyp_pieces} ({speaker}-{sid})", file=self.hypo_units_file)
            print(f"{hyp_words} ({speaker}-{sid})", file=self.hypo_words_file)
            print(f"{tgt_pieces} ({speaker}-{sid})", file=self.ref_units_file)
            print(f"{tgt_words} ({speaker}-{sid})", file=self.ref_words_file)

        if not self.cfg.common_eval.quiet:
            logger.info(f"HYPO: {hyp_words}")
            logger.info(f"REF: {tgt_words}")
            logger.info("---------------------")

        hyp_words, tgt_words = hyp_words.split(), tgt_words.split()

        return editdistance.eval(hyp_words, tgt_words), len(tgt_words)
    
    def lm_sim(self, sample: Dict[str, Any]):
        ############for distillation###########
        device = "cuda"
        toks_list = sample["target"]
        tgt_list = [] 
        for toks in toks_list:
            # Processes target.
            target_tokens = utils.strip_pad(toks, self.tgt_dict.pad())
            tgt_pieces = self.tgt_dict.string(target_tokens.int().cpu())
            tgt_words = post_process(tgt_pieces, 'letter').lower()

            tgt_list.append(tgt_words)

        print(tgt_list)
   
        lm_input = self.tokenizer(tgt_list, return_tensors='pt', padding=True, return_attention_mask=True).to(device)
        with torch.no_grad():
            lm_output = self.lm(**lm_input)
            lm_output = lm_output['last_hidden_state']

            net_output = self.models[0](**sample["net_input"])
            am_output = net_output['encoder_feat'].transpose(0, 1)
            am_output = am_output.transpose(1, 2).contiguous()
            for i, conv in enumerate(self.lm_decoder):
                am_output = conv(am_output)
            am_output = am_output.transpose(1, 2)
            
            lm_am_sim = torch.bmm(am_output, lm_output.transpose(1, 2))
            lm_am_sim = F.softmax(lm_am_sim, dim=-1)

        for b in range(lm_am_sim.size(0)):
            filename = sample['filename'][b].split('/')[-1].replace('.flac', '')

            plt.matshow(lm_am_sim[b].T.cpu().numpy())
            plt.colorbar()
            if not os.path.exists(f'/home/work/workspace/fairseq/scripts/whale/tc_png'):
                try: os.makedirs(f'/home/work/workspace/fairseq/scripts/whale/tc_png')
                except: pass
            plt.savefig(f'/home/work/workspace/fairseq/scripts/whale/tc_png/{filename}.png')
            plt.close()


    def process_sample(self, sample: Dict[str, Any]) -> None:
        self.gen_timer.start()
        
        if self.prompt is not None:
            device = sample['net_input']['source'].device
            self.prompt = self.prompt.to(device)
            sample['net_input']['prompt'] = self.prompt
            sample['net_input']['filename'] = sample['filename']

        hypos = self.task.inference_step(
            generator=self.generator,
            models=self.models,
            sample=sample,
        )

        self.lm_sim(sample)
        
        if 0:
            net_output = self.models[0](**sample["net_input"])
            am_output = net_output['encoder_feat'].transpose(0, 1) ## T x B x C -> B x T x C
            am_output = am_output.transpose(1, 2).contiguous()
            for i, conv in enumerate(self.lm_decoder):
                try: am_output = conv(am_output)
                except: 
                    conv.to(sample['net_input']['source'].device)
                    am_output = conv(am_output)
            
            '''
            dim1 = F.normalize(am_output[0,:,0].squeeze(), dim=0)
            dim2 = F.normalize(am_output[0,:,1].squeeze(), dim=0)
            dim3 = F.normalize(am_output[0,:,2].squeeze(), dim=0)
            dim4 = F.normalize(am_output[0,:,3].squeeze(), dim=0)
            dim5 = F.normalize(am_output[0,:,4].squeeze(), dim=0)
            dim6 = F.normalize(am_output[0,:,5].squeeze(), dim=0)
            dim7 = F.normalize(am_output[0,:,6].squeeze(), dim=0)
            dim8 = F.normalize(am_output[0,:,7].squeeze(), dim=0)
            dim9 = F.normalize(am_output[0,:,8].squeeze(), dim=0)
            dim10 = F.normalize(am_output[0,:,9].squeeze(), dim=0)
            dim11 = F.normalize(am_output[0,:,10].squeeze(), dim=0)
            dim12 = F.normalize(am_output[0,:,11].squeeze(), dim=0)
            
            sim1 = torch.matmul(dim1, dim2)
            sim2 = torch.matmul(dim1, dim3)
            sim3 = torch.matmul(dim1, dim4)
            sim4 = torch.matmul(dim1, dim5)
            sim5 = torch.matmul(dim1, dim6)
            sim6 = torch.matmul(dim1, dim7)
            sim7 = torch.matmul(dim1, dim8)
            sim8 = torch.matmul(dim1, dim9)
            sim9 = torch.matmul(dim1, dim10)
            sim10 = torch.matmul(dim1, dim11)
            sim11 = torch.matmul(dim1, dim12)

            print(am_output.size(), dim1.size(), sim1.item(), sim2.item(), sim3.item(), sim4.item(), sim5.item(), sim6.item(), sim7.item(), sim8.item(), sim9.item(), sim10.item(), sim11.item())
            '''
        num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
        self.gen_timer.stop(num_generated_tokens)
        self.wps_meter.update(num_generated_tokens)

        for batch_id, sample_id in enumerate(sample["id"].tolist()):
            errs, length = self.process_sentence(
                sample=sample,
                sid=sample_id,
                batch_id=batch_id,
                hypo=hypos[batch_id][0],
            )
            self.total_errors += errs
            self.total_length += length

        self.log({"wps": round(self.wps_meter.avg)})
        if "nsentences" in sample:
            self.num_sentences += sample["nsentences"]
        else:
            self.num_sentences += sample["id"].numel()

    def log_generation_time(self) -> None:
        logger.info(
            "Processed %d sentences (%d tokens) in %.1fs %.2f "
            "sentences per second, %.2f tokens per second)",
            self.num_sentences,
            self.gen_timer.n,
            self.gen_timer.sum,
            self.num_sentences / (self.gen_timer.sum + 1e-6),
            1.0 / (self.gen_timer.avg + 1e-6),
        )


def parse_wer(wer_file: Path) -> float:
    with open(wer_file, "r") as f:
        return float(f.readline().strip().split(" ")[1])


def get_wer_file(cfg: InferConfig) -> Path:
    """Hashes the decoding parameters to a unique file ID."""
    base_path = "wer"
    if cfg.decoding.results_path is not None:
        base_path = os.path.join(cfg.decoding.results_path, base_path)

    if cfg.decoding.unique_wer_file:
        yaml_str = OmegaConf.to_yaml(cfg.decoding)
        fid = int(hashlib.md5(yaml_str.encode("utf-8")).hexdigest(), 16)
        return Path(f"{base_path}.{fid % 1000000}")
    else:
        return Path(base_path)


def main(cfg: InferConfig) -> float:
    """Entry point for main processing logic.

    Args:
        cfg: The inferance configuration to use.
        wer: Optional shared memory pointer for returning the WER. If not None,
            the final WER value will be written here instead of being returned.

    Returns:
        The final WER if `wer` is None, otherwise None.
    """

    yaml_str, wer_file = OmegaConf.to_yaml(cfg.decoding), get_wer_file(cfg)

    # Validates the provided configuration.
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.max_tokens = 4000000
    if not cfg.common.cpu and not torch.cuda.is_available():
        raise ValueError("CUDA not found; set `cpu=True` to run without CUDA")

    logger.info(cfg.common_eval.path)

    with InferenceProcessor(cfg) as processor:
        for sample in processor:
            #processor.process_sample(sample)
            processor.lm_sim(sample)

        processor.log_generation_time()

        if cfg.decoding.results_path is not None:
            processor.merge_shards()

        errs_t, leng_t = processor.total_errors, processor.total_length

        if cfg.common.cpu:
            logger.warning("Merging WER requires CUDA.")
        elif processor.data_parallel_world_size > 1:
            stats = torch.LongTensor([errs_t, leng_t]).cuda()
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            errs_t, leng_t = stats[0].item(), stats[1].item()

        wer = errs_t * 100.0 / leng_t

        if distributed_utils.is_master(cfg.distributed_training):
            with open(wer_file, "w") as f:
                f.write(
                    (
                        f"WER: {wer}\n"
                        f"err / num_ref_words = {errs_t} / {leng_t}\n\n"
                        f"{yaml_str}"
                    )
                )

        return wer


@hydra.main(config_path=config_path, config_name="infer")
def hydra_main(cfg: InferConfig) -> Union[float, Tuple[float, Optional[float]]]:
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    cfg = OmegaConf.create(container)
    OmegaConf.set_struct(cfg, True)

    if cfg.common.reset_logging:
        reset_logging()

    utils.import_user_module(cfg.common)

    # logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    wer = float("inf")

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, main)
        else:
            distributed_utils.call_main(cfg, main)

        wer = parse_wer(get_wer_file(cfg))
    except BaseException as e:  # pylint: disable=broad-except
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! %s", str(e))

    logger.info("Word error rate: %.4f", wer)
    if cfg.is_ax:
        return wer, None

    return wer


def cli_main() -> None:
    try:
        from hydra._internal.utils import (
            get_args,
        )  # pylint: disable=import-outside-toplevel

        cfg_name = get_args().config_name or "infer"
    except ImportError:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "infer"

    cs = ConfigStore.instance()
    cs.store(name=cfg_name, node=InferConfig)

    for k in InferConfig.__dataclass_fields__:
        if is_dataclass(InferConfig.__dataclass_fields__[k].type):
            v = InferConfig.__dataclass_fields__[k].default
            cs.store(name=k, node=v)

    hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    cli_main()
