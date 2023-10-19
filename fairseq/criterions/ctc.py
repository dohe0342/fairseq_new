# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II
from typing import Optional
import numpy as np

import torch
import torch.nn.functional as F

import fairseq
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import post_process
from fairseq.tasks import FairseqTask
from fairseq.logging.meters import safe_round
from fairseq.models.wav2vec.wav2vec2_asr import Linear

from transformers import GPT2Tokenizer, GPT2Model, BertTokenizer, BertModel
from fairseq.data.data_utils import post_process


@dataclass
class CtcCriterionConfig(FairseqDataclass):
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(
        default="letter",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
            "wordpiece, BPE symbols, etc. "
            "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )
    wer_kenlm_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "if this is provided, use kenlm to compute wer (along with other wer_* args)"
        },
    )
    wer_lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "lexicon to use with wer_kenlm_model"},
    )
    wer_lm_weight: float = field(
        default=2.0,
        metadata={"help": "lm weight to use with wer_kenlm_model"},
    )
    wer_word_score: float = field(
        default=-1.0,
        metadata={"help": "lm word score to use with wer_kenlm_model"},
    )
    wer_sil_weight: float = field(
        default=0,
        metadata={"help": "lm word score to use with wer_kenlm_model"},
    )

    wer_args: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)"
        },
    )
    prompt: bool = field(
        default=False,
        metadata={"help": "use prompt as guidance of data augmentation"},
    )


@dataclass
class ClipCriterionConfig(CtcCriterionConfig):
    decoder: float = field(
        default=2.0,
        metadata={"help": "lm weight to use with wer_kenlm_model"},
    )
    prompt: bool = field(
        default=False,
        metadata={"help": "use prompt as guidance of data augmentation"},
    )


@register_criterion("ctc", dataclass=CtcCriterionConfig)
class CtcCriterion(FairseqCriterion):
    def __init__(
        self, cfg: CtcCriterionConfig, task: FairseqTask, rdrop_alpha: int = 0.0
    ):
        super().__init__(task)
        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process

        self.rdrop_alpha = rdrop_alpha

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None and cfg.wer_kenlm_model != "":
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lexicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.sil_weight = cfg.wer_sil_weight
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg

    def forward(self, model, sample, reduce=True, **kwargs):
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder

        # CTC loss is calculated over duplicated inputs
        # sample is already duplicated for R-Drop
        if self.rdrop_alpha > 0:
            for k, v in sample.items():
                if k in ["target", "target_lengths"]:
                    sample[k] = torch.cat([v, v.clone()], dim=0)
                elif k == "net_input":
                    if sample[k]["src_tokens"].size(1) != sample[k]["src_lengths"].size(
                        0
                    ):
                        # for decoder CTC loss
                        sample[k]["src_lengths"] = torch.cat(
                            [
                                sample[k]["src_lengths"],
                                sample[k]["src_lengths"].clone(),
                            ],
                            dim=0,
                        )

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("ctc_reg", dataclass=CtcCriterionConfig)
class CtcCriterion(FairseqCriterion):
    def __init__(
        self, cfg: CtcCriterionConfig, task: FairseqTask, rdrop_alpha: int = 0.0
    ):
        super().__init__(task)
        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process

        self.rdrop_alpha = rdrop_alpha

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None and cfg.wer_kenlm_model != "":
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lexicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.sil_weight = cfg.wer_sil_weight
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg

    def forward(self, model, sample, reduce=True, **kwargs):
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder

        # CTC loss is calculated over duplicated inputs
        # sample is already duplicated for R-Drop
        if self.rdrop_alpha > 0:
            for k, v in sample.items():
                if k in ["target", "target_lengths"]:
                    sample[k] = torch.cat([v, v.clone()], dim=0)
                elif k == "net_input":
                    if sample[k]["src_tokens"].size(1) != sample[k]["src_lengths"].size(
                        0
                    ):
                        # for decoder CTC loss
                        sample[k]["src_lengths"] = torch.cat(
                            [
                                sample[k]["src_lengths"],
                                sample[k]["src_lengths"].clone(),
                            ],
                            dim=0,
                        )

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )
            #loss_l2 = 

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("interctc", dataclass=CtcCriterionConfig)
class InterCtcCriterion(CtcCriterion):
    def __init__(
        self, cfg: CtcCriterionConfig, task: FairseqTask, rdrop_alpha: int = 0.0
    ):
        super().__init__(cfg, task, rdrop_alpha)
        self.inter_ctc_idx = [2, 5, 8, 11]
        
    def forward(self, model, sample, reduce=True, **kwargs):
        net_output = model(**sample["net_input"])
        inter_output_list = [model.w2v_encoder.final_dropout(x[0]) for i, x in enumerate(net_output["layer_results"]) if i in self.inter_ctc_idx]
        inter_output_list = [model.w2v_encoder.proj(x) for x in inter_output_list]
        
        inter_output_dict = []
        for i, x in enumerate(inter_output_list):
           inter_output_dict.append(dict()) 
           inter_output_dict[i]["padding_mask"] = net_output["padding_mask"]
           inter_output_dict[i]["encoder_out"] = x
        
        lprobs_list = [model.get_normalized_probs(
                inter_output, log_probs=True
            ).contiguous() for inter_output in inter_output_dict
        ]
        lprobs = lprobs_list[-1]
        #lprobs = model.get_normalized_probs(
        #    net_output, log_probs=True
        #).contiguous()  # (T, B, C) from the encoder


        # CTC loss is calculated over duplicated inputs
        # sample is already duplicated for R-Drop
        if self.rdrop_alpha > 0:
            for k, v in sample.items():
                if k in ["target", "target_lengths"]:
                    sample[k] = torch.cat([v, v.clone()], dim=0)
                elif k == "net_input":
                    if sample[k]["src_tokens"].size(1) != sample[k]["src_lengths"].size(
                        0
                    ):
                        # for decoder CTC loss
                        sample[k]["src_lengths"] = torch.cat(
                            [
                                sample[k]["src_lengths"],
                                sample[k]["src_lengths"].clone(),
                            ],
                            dim=0,
                        )

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            '''
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )
            '''
            loss_list = [F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            ) for lprobs in lprobs_list]
            
            #loss = sum(loss_list)
            loss = 0.7*loss_list[-1] + 0.3*sum(loss_list[:-1])

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }
        for i, loss_ in enumerate(loss_list):
            logging_output[f"loss_{self.inter_ctc_idx[i]}"] = utils.item(loss_.data)

        if not model.training:
            import editdistance

            with torch.no_grad():
                for i, lprobs in enumerate(lprobs_list):
                    lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                    c_err = 0
                    c_len = 0
                    w_errs = 0
                    w_len = 0
                    wv_errs = 0
                    for lp, t, inp_l in zip(
                        lprobs_t,
                        sample["target_label"]
                        if "target_label" in sample
                        else sample["target"],
                        input_lengths,
                    ):
                        lp = lp[:inp_l].unsqueeze(0)

                        decoded = None
                        if self.w2l_decoder is not None:
                            decoded = self.w2l_decoder.decode(lp)
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]
                                if len(decoded) < 1:
                                    decoded = None
                                else:
                                    decoded = decoded[0]

                        p = (t != self.task.target_dictionary.pad()) & (
                            t != self.task.target_dictionary.eos()
                        )
                        targ = t[p]
                        targ_units = self.task.target_dictionary.string(targ)
                        targ_units_arr = targ.tolist()

                        toks = lp.argmax(dim=-1).unique_consecutive()
                        pred_units_arr = toks[toks != self.blank_idx].tolist()

                        c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                        c_len += len(targ_units_arr)

                        targ_words = post_process(targ_units, self.post_process).split()

                        pred_units = self.task.target_dictionary.string(pred_units_arr)
                        pred_words_raw = post_process(pred_units, self.post_process).split()

                        if decoded is not None and "words" in decoded:
                            pred_words = decoded["words"]
                            w_errs += editdistance.eval(pred_words, targ_words)
                            wv_errs += editdistance.eval(pred_words_raw, targ_words)
                        else:
                            dist = editdistance.eval(pred_words_raw, targ_words)
                            w_errs += dist
                            wv_errs += dist

                        w_len += len(targ_words)

                    logging_output[f"wv_errors_{self.inter_ctc_idx[i]}"] = wv_errs
                    logging_output[f"w_errors_{self.inter_ctc_idx[i]}"] = w_errs
                    logging_output[f"w_total_{self.inter_ctc_idx[i]}"] = w_len
                    logging_output[f"c_errors_{self.inter_ctc_idx[i]}"] = c_err
                    logging_output[f"c_total_{self.inter_ctc_idx[i]}"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        #for idx in self.inter_ctc_idx:
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        
        for idx in [2, 5, 8, 11]:
            loss_sum = utils.item(sum(log.get(f"loss_{idx}", 0) for log in logging_outputs))
            metrics.log_scalar(
            f"loss_{idx}", loss_sum / sample_size / math.log(2), sample_size, round=3
                )
            if sample_size != ntokens:
                metrics.log_scalar(
                    "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
                )

            c_errors = sum(log.get(f"c_errors_{idx}", 0) for log in logging_outputs)
            metrics.log_scalar(f"_c_errors_{idx}", c_errors)
            c_total = sum(log.get(f"c_total_{idx}", 0) for log in logging_outputs)
            metrics.log_scalar(f"_c_total_{idx}", c_total)
            w_errors = sum(log.get(f"w_errors_{idx}", 0) for log in logging_outputs)
            metrics.log_scalar(f"_w_errors_{idx}", w_errors)
            wv_errors = sum(log.get(f"wv_errors_{idx}", 0) for log in logging_outputs)
            metrics.log_scalar(f"_wv_errors_{idx}", wv_errors)
            w_total = sum(log.get(f"w_total_{idx}", 0) for log in logging_outputs)
            metrics.log_scalar(f"_w_total_{idx}", w_total)

            if c_total > 0:
                metrics.log_derived(
                    f"uer_{idx}",
                    lambda meters: safe_round(
                        meters[f"_c_errors_{idx}"].sum * 100.0 / meters[f"_c_total_{idx}"].sum, 3
                    )
                    if meters[f"_c_total_{idx}"].sum > 0
                    else float("nan"),
                )
            if w_total > 0:
                metrics.log_derived(
                    f"wer_{idx}",
                    lambda meters: safe_round(
                        meters[f"_w_errors_{idx}"].sum * 100.0 / meters[f"_w_total_{idx}"].sum, 3
                    )
                    if meters[f"_w_total_{idx}"].sum > 0
                    else float("nan"),
                )
                metrics.log_derived(
                    f"raw_wer_{idx}",
                    lambda meters: safe_round(
                        meters[f"_wv_errors_{idx}"].sum * 100.0 / meters[f"_w_total_{idx}"].sum, 3
                    )
                    if meters[f"_w_total_{idx}"].sum > 0
                    else float("nan"),
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


'''
class AttnHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        return output
'''

@register_criterion("prompt", dataclass=CtcCriterionConfig)
class PromptCtcCriterion(CtcCriterion):
    def __init__(
        self, cfg: CtcCriterionConfig, task: FairseqTask, rdrop_alpha: int = 0.0
    ):
        super().__init__(cfg, task, rdrop_alpha)
        
        if 0:
            statistic = open(f'/home/work/workspace/icefall/egs/librispeech/ASR/conv_feat/1284/1284_statistic.txt', 'r').readlines()
            new_emb = torch.empty(512, 50)
            for i in range(512):
                mean, std = statistic[i].strip().split(' ')
                #print(new_emb[i].size())
                #print(float(mean), float(std))
                new_emb[i] = torch.normal(float(mean), float(std), size=(1,50)).squeeze()
            new_emb = new_emb.transpose(1,0)
        #self.prompt = torch.nn.Parameter(new_emb).half()
        #self.prompt = torch.nn.Parameter(new_emb)
        
        self.prompt = torch.nn.Parameter(torch.randn(120, 512)/10.)
        self.attn_output = []
                
    def hook_fn(self, module, input, output):
        self.attn_output.append(output)

    def forward(self, model, sample, reduce=True, **kwargs):
        '''
        count = 0
        for modules in model.modules():
            if isinstance(modules, fairseq.modules.multihead_attention.MultiheadAttention):
                for module in modules.modules():
                    if isinstance(module, torch.nn.Linear):
                        module.register_forward_hook(self.hook_fn)
                        count += 1
        '''
        device = sample['net_input']['source'].device
        self.prompt = self.prompt.to(device)
        sample['net_input']['prompt'] = self.prompt
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder

        #print(len(self.attn_output))
        #for i, output in enumerate(self.attn_output):
        #    print(i, output.size())

        #lprobs = lprobs[50:, :, :]

        #print(sample["target"])
        #print(sample["target"].size())
        #exit()
        #if 1: lprobs = lprobs[50:, :, :]

        # CTC loss is calculated over duplicated inputs
        # sample is already duplicated for R-Drop
        if self.rdrop_alpha > 0:
            for k, v in sample.items():
                if k in ["target", "target_lengths"]:
                    sample[k] = torch.cat([v, v.clone()], dim=0)
                elif k == "net_input":
                    if sample[k]["src_tokens"].size(1) != sample[k]["src_lengths"].size(
                        0
                    ):
                        # for decoder CTC loss
                        sample[k]["src_lengths"] = torch.cat(
                            [
                                sample[k]["src_lengths"],
                                sample[k]["src_lengths"].clone(),
                            ],
                            dim=0,
                        )

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                #net_output["padding_mask"] = net_output["padding_mask"][:,50:]
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )
        #if input_lengths is not None:
        #    input_lengths -= 50
        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output


@register_criterion("prompt2", dataclass=CtcCriterionConfig)
class Prompt2CtcCriterion(CtcCriterion):
    def __init__(
        self, cfg: CtcCriterionConfig, task: FairseqTask, rdrop_alpha: int = 0.0
    ):
        super().__init__(cfg, task, rdrop_alpha)
        
        if 0:
            statistic = open(f'/home/work/workspace/icefall/egs/librispeech/ASR/conv_feat/1284/1284_statistic.txt', 'r').readlines()
            new_emb = torch.empty(512, 50)
            for i in range(512):
                mean, std = statistic[i].strip().split(' ')
                #print(new_emb[i].size())
                #print(float(mean), float(std))
                new_emb[i] = torch.normal(float(mean), float(std), size=(1,50)).squeeze()
            new_emb = new_emb.transpose(1,0)
        #self.prompt = torch.nn.Parameter(new_emb).half()
        #self.prompt = torch.nn.Parameter(new_emb)
        
        self.prompt = torch.nn.Parameter(torch.randn(2, 120, 512)/10.)
        #torch.nn.init.orthogonal_(self.prompt)
        '''
        ckpt = torch.load('/home/work/workspace/fairseq/scripts/whale/outputs/w2v2_200h_clean+speech_mixed-valid_prompt_prompt-freeze80000_orthogonal/checkpoint_best.pt')
        self.prompt = ckpt['criterion']['prompt']
        self.prompt = torch.nn.Parameter(self.prompt)
        '''
                
    def hook_fn(self, module, input, output):
        self.attn_output.append(output)

    def forward(self, model, sample, reduce=True, **kwargs):
        if model.w2v_encoder.num_updates < 20000:
            self.prompt.requries_grad = False
        else:
            self.prompt.requries_grad = True
        '''
        if model.w2v_encoder.num_updates < 10000:
            for n, p in model.named_parameters():
                if 'w2v_encoder.proj' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        elif model.w2v_encoder.num_updates >= 10000 and model.w2v_encoder.num_updates < 50000:
            self.prompt.requires_grad = False
            for n, p in model.named_parameters():
                p.requires_grad = True
        else:
            self.prompt.requires_grad = True
        '''
        '''
        if model.w2v_encoder.num_updates % 2 == 0:
            self.prompt.requires_grad = False
            for n, p in model.named_parameters():
                p.requires_grad = True
        else:
            self.prompt_requries_grad = True
            for n, p in model.named_parameters():
                if 'norm' in n or 'w2v_encoder.proj' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        '''

        device = sample['net_input']['source'].device
        self.prompt = self.prompt.to(device)
        
        sample['net_input']['prompt'] = self.prompt
        sample['net_input']['filename'] = sample['filename']

        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder

        #lprobs = lprobs[120:, :, :]

        # CTC loss is calculated over duplicated inputs
        # sample is already duplicated for R-Drop
        if self.rdrop_alpha > 0:
            for k, v in sample.items():
                if k in ["target", "target_lengths"]:
                    sample[k] = torch.cat([v, v.clone()], dim=0)
                elif k == "net_input":
                    if sample[k]["src_tokens"].size(1) != sample[k]["src_lengths"].size(
                        0
                    ):
                        # for decoder CTC loss
                        sample[k]["src_lengths"] = torch.cat(
                            [
                                sample[k]["src_lengths"],
                                sample[k]["src_lengths"].clone(),
                            ],
                            dim=0,
                        )

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                #net_output["padding_mask"] = net_output["padding_mask"][:,120:]
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )
        
        #if input_lengths is not None:
        #    input_lengths -= 120

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output


@register_criterion("prefix", dataclass=CtcCriterionConfig)
class PrefixCtcCriterion(CtcCriterion):
    def __init__(
        self, cfg: CtcCriterionConfig, task: FairseqTask, rdrop_alpha: int = 0.0
    ):
        super().__init__(cfg, task, rdrop_alpha)
        key_prefix = [torch.nn.Parameter(torch.rand(50, 768)/10.) for i in range(12)]
        value_prefix = [torch.nn.Parameter(torch.rand(50, 768)/10.) for i in range(12)]
        self.prefix = [key_prefix, value_prefix]
        
    def forward(self, model, sample, reduce=True, **kwargs):
        sample['net_input']['prefix'] = self.prefix
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder
        #if 1: lprobs = lprobs[50:, :, :]

        # CTC loss is calculated over duplicated inputs
        # sample is already duplicated for R-Drop
        if self.rdrop_alpha > 0:
            for k, v in sample.items():
                if k in ["target", "target_lengths"]:
                    sample[k] = torch.cat([v, v.clone()], dim=0)
                elif k == "net_input":
                    if sample[k]["src_tokens"].size(1) != sample[k]["src_lengths"].size(
                        0
                    ):
                        # for decoder CTC loss
                        sample[k]["src_lengths"] = torch.cat(
                            [
                                sample[k]["src_lengths"],
                                sample[k]["src_lengths"].clone(),
                            ],
                            dim=0,
                        )

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )
        #if input_lengths is not None:
        #    input_lengths -= 50
        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output


@register_criterion("clip", dataclass=CtcCriterionConfig)
class ClipCriterion(FairseqCriterion):
    def __init__(
        self, cfg: CtcCriterionConfig, task: FairseqTask, rdrop_alpha: int = 0.0
    ):
        super().__init__(task)
        
        ########### for gpt2
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = GPT2Model.from_pretrained('gpt2')
        
        #self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
        #self.lm = BertModel.from_pretrained("bert-large-uncased-whole-word-masking")
        #self.lm = GPT2Model.from_pretrained('/home/work/workspace/models/checkpoint-420500')
        self.task = task
        self.tgt_dict = task.target_dictionary
        self.lm_linear = Linear(768, 768)
        self.ins_norm = torch.nn.InstanceNorm1d(768)
        ##############################################################
        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process

        self.rdrop_alpha = rdrop_alpha

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None and cfg.wer_kenlm_model != "":
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lmlmtoammicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.sil_weight = cfg.wer_sil_weight
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg

    def forward(self, model, sample, reduce=True, **kwargs):
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder
        
        ############for distillation###########
        device = lprobs.device
        toks_list = sample["target"]
        tgt_list = []
        for toks in toks_list:
            # Processes target.
            target_tokens = utils.strip_pad(toks, self.tgt_dict.pad())
            tgt_pieces = self.tgt_dict.string(target_tokens.int().cpu())
            #tgt_words = post_process(tgt_pieces, 'letter')
            tgt_words = post_process(tgt_pieces, 'letter').lower()

            tgt_list.append(tgt_words)
        
        lm_input = self.tokenizer(tgt_list, return_tensors='pt', padding=True, return_attention_mask=True).to(device)
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                lm_output = self.lm(**lm_input)
                lm_output = lm_output['last_hidden_state']
                if 0:
                    lm_output = lm_output[:,1:,:]
                
                lm_output = F.normalize(lm_output, dim=2)

            am_output = net_output['encoder_feat'].transpose(0, 1) ## T x B x C -> B x T x C
            #am_output = F.gelu(am_output)
            am_output = self.lm_linear(am_output)
            #am_output = self.ins_norm(am_output)
            am_output = F.normalize(am_output, dim=2)
            
            lm_am_sim = torch.bmm(am_output, lm_output.transpose(1, 2))
            if np.random.rand() < 0.1 and 0:
                softmax = F.softmax(lm_am_sim / 3, dim=-1)
                print(softmax.size(), softmax[0][0].size(), softmax[0][0])
                print(softmax[0][100])
                print(softmax[0][200])
                print(softmax[0][300])
                print('-'*20)

            lm_am_sim = F.log_softmax(lm_am_sim, dim=-1)
            #lm_am_sim = F.log_softmax(lm_am_sim / 3, dim=-1)
            lm_am_sim = F.pad(lm_am_sim, (1, 0, 0, 0, 0, 0), value=np.log(np.e**-1))
            lm_am_sim = lm_am_sim.transpose(0, 1).contiguous()

        ##############################

        # CTC loss is calculated over duplicated inputs
        # sample is already duplicated for R-Drop
        if self.rdrop_alpha > 0:
            for k, v in sample.items():
                if k in ["target", "target_lengths"]:
                    sample[k] = torch.cat([v, v.clone()], dim=0)
                elif k == "net_input":
                    if sample[k]["src_tokens"].size(1) != sample[k]["src_lengths"].size(
                        0
                    ):
                        # for decoder CTC loss
                        sample[k]["src_lengths"] = torch.cat(
                            [
                                sample[k]["src_lengths"],
                                sample[k]["src_lengths"].clone(),
                            ],
                            dim=0,
                        )

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)
        
        #############for alignment target ###############################
        #alignment_pad_mask = lm_input["attention_mask"] > 0
        alignment_lengths = torch.sum(lm_input["attention_mask"], 1)
        if 0:
            alignment_lengths -= 1

        alignment_flat = torch.linspace(
                                            1, 
                                            alignment_lengths[0], 
                                            steps=alignment_lengths[0]
                                    ).to(device)
        
        for i in alignment_lengths[1:]:
            temp_target = torch.linspace(1, i, steps=i).to(device)
            alignment_flat = torch.cat([alignment_flat, temp_target])
            alignment_flat = alignment_flat.to(torch.cuda.IntTensor())
        #############for alignment target ###############################

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )
            
            #print('1', lprobs.size())
            #print('2', targets_flat.dtype)
            #print('3', input_lengths.dtype)
            #print('4', target_lengths.dtype)
                
            #print('5', lm_am_sim.size())
            #print('6', alignment_flat.dtype)
            #print('7', input_lengths.dtype)
            #print('8', alignment_lengths.dtype)
            
            distill_loss = F.ctc_loss(
                lm_am_sim,
                alignment_flat,
                input_lengths,
                alignment_lengths,
                blank=0,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

            loss = ctc_loss + 0.1*distill_loss

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ctc_loss": utils.item(ctc_loss.data),  # * sample['ntokens'],
            "distill_loss": utils.item(distill_loss.data),
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ctc_loss_sum = utils.item(sum(log.get("ctc_loss", 0) for log in logging_outputs))
        distill_loss_sum = utils.item(sum(log.get("distill_loss", 0) for log in logging_outputs))

        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "distill_loss", distill_loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("clip2", dataclass=CtcCriterionConfig)
class Clip2Criterion(FairseqCriterion):
    def __init__(
        self, cfg: CtcCriterionConfig, task: FairseqTask, rdrop_alpha: int = 0.0
    ):
        super().__init__(task)
        
        ########### for gpt2
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = GPT2Model.from_pretrained('gpt2')
        
        self.task = task
        self.tgt_dict = task.target_dictionary
        #self.lm_linear = Linear(768, 768)
        #self.ins_norm = torch.nn.InstanceNorm1d(768)

        
        ##############################################################
        self.blank_idx = (
            task.target_dictionary.index(task.blank_symbol)
            if hasattr(task, "blank_symbol")
            else 0
        )
        self.pad_idx = task.target_dictionary.pad()
        self.eos_idx = task.target_dictionary.eos()
        self.post_process = cfg.post_process

        self.rdrop_alpha = rdrop_alpha

        if cfg.wer_args is not None:
            (
                cfg.wer_kenlm_model,
                cfg.wer_lexicon,
                cfg.wer_lm_weight,
                cfg.wer_word_score,
            ) = eval(cfg.wer_args)

        if cfg.wer_kenlm_model is not None and cfg.wer_kenlm_model != "":
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            dec_args = Namespace()
            dec_args.nbest = 1
            dec_args.criterion = "ctc"
            dec_args.kenlm_model = cfg.wer_kenlm_model
            dec_args.lmlmtoammicon = cfg.wer_lexicon
            dec_args.beam = 50
            dec_args.beam_size_token = min(50, len(task.target_dictionary))
            dec_args.beam_threshold = min(50, len(task.target_dictionary))
            dec_args.lm_weight = cfg.wer_lm_weight
            dec_args.word_score = cfg.wer_word_score
            dec_args.sil_weight = cfg.wer_sil_weight
            dec_args.unk_weight = -math.inf
            dec_args.sil_weight = 0

            self.w2l_decoder = W2lKenLMDecoder(dec_args, task.target_dictionary)
        else:
            self.w2l_decoder = None

        self.zero_infinity = cfg.zero_infinity
        self.sentence_avg = cfg.sentence_avg

    def forward(self, model, sample, reduce=True, **kwargs):
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True
        ).contiguous()  # (T, B, C) from the encoder
        
        ############for distillation###########
        device = lprobs.device
        toks_list = sample["target"]
        tgt_list = []
        for toks in toks_list:
            # Processes target.
            target_tokens = utils.strip_pad(toks, self.tgt_dict.pad())
            tgt_pieces = self.tgt_dict.string(target_tokens.int().cpu())
            #tgt_words = post_process(tgt_pieces, 'letter')
            tgt_words = post_process(tgt_pieces, 'letter').lower()

            tgt_list.append(tgt_words)
        
        lm_input = self.tokenizer(tgt_list, return_tensors='pt', padding=True, return_attention_mask=True).to(device)
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                lm_output = self.lm(**lm_input)
                lm_output = lm_output['last_hidden_state']
                if 0:
                    lm_output = lm_output[:,1:,:]
                
                lm_output = F.normalize(lm_output, dim=2)

            am_output = net_output['encoder_feat'].transpose(0, 1) ## T x B x C -> B x T x C
            #am_output = F.gelu(am_output)
            am_output = self.lm_linear(am_output)
            #am_output = self.ins_norm(am_output)
            am_output = F.normalize(am_output, dim=2)
            
            lm_am_sim = torch.bmm(am_output, lm_output.transpose(1, 2))
            if np.random.rand() < 0.1 and 0:
                softmax = F.softmax(lm_am_sim / 3, dim=-1)
                print(softmax.size(), softmax[0][0].size(), softmax[0][0])
                print(softmax[0][100])
                print(softmax[0][200])
                print(softmax[0][300])
                print('-'*20)

            lm_am_sim = F.log_softmax(lm_am_sim, dim=-1)
            #lm_am_sim = F.log_softmax(lm_am_sim / 3, dim=-1)
            lm_am_sim = F.pad(lm_am_sim, (1, 0, 0, 0, 0, 0), value=np.log(np.e**-1))
            lm_am_sim = lm_am_sim.transpose(0, 1).contiguous()

        ##############################

        # CTC loss is calculated over duplicated inputs
        # sample is already duplicated for R-Drop
        if self.rdrop_alpha > 0:
            for k, v in sample.items():
                if k in ["target", "target_lengths"]:
                    sample[k] = torch.cat([v, v.clone()], dim=0)
                elif k == "net_input":
                    if sample[k]["src_tokens"].size(1) != sample[k]["src_lengths"].size(
                        0
                    ):
                        # for decoder CTC loss
                        sample[k]["src_lengths"] = torch.cat(
                            [
                                sample[k]["src_lengths"],
                                sample[k]["src_lengths"].clone(),
                            ],
                            dim=0,
                        )

        if "src_lengths" in sample["net_input"]:
            input_lengths = sample["net_input"]["src_lengths"]
        else:
            if net_output["padding_mask"] is not None:
                non_padding_mask = ~net_output["padding_mask"]
                input_lengths = non_padding_mask.long().sum(-1)
            else:
                input_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )

        pad_mask = (sample["target"] != self.pad_idx) & (
            sample["target"] != self.eos_idx
        )
        targets_flat = sample["target"].masked_select(pad_mask)
        if "target_lengths" in sample:
            target_lengths = sample["target_lengths"]
        else:
            target_lengths = pad_mask.sum(-1)
        
        #############for alignment target ###############################
        #alignment_pad_mask = lm_input["attention_mask"] > 0
        alignment_lengths = torch.sum(lm_input["attention_mask"], 1)
        if 0:
            alignment_lengths -= 1

        alignment_flat = torch.linspace(
                                            1, 
                                            alignment_lengths[0], 
                                            steps=alignment_lengths[0]
                                    ).to(device)
        
        for i in alignment_lengths[1:]:
            temp_target = torch.linspace(1, i, steps=i).to(device)
            alignment_flat = torch.cat([alignment_flat, temp_target])
            alignment_flat = alignment_flat.to(torch.cuda.IntTensor())
        #############for alignment target ###############################

        with torch.backends.cudnn.flags(enabled=False):
            ctc_loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )
            
            #print('1', lprobs.size())
            #print('2', targets_flat.dtype)
            #print('3', input_lengths.dtype)
            #print('4', target_lengths.dtype)
                
            #print('5', lm_am_sim.size())
            #print('6', alignment_flat.dtype)
            #print('7', input_lengths.dtype)
            #print('8', alignment_lengths.dtype)
            
            distill_loss = F.ctc_loss(
                lm_am_sim,
                alignment_flat,
                input_lengths,
                alignment_lengths,
                blank=0,
                reduction="sum",
                zero_infinity=self.zero_infinity,
            )

            loss = ctc_loss + 0.1*distill_loss

        ntokens = (
            sample["ntokens"] if "ntokens" in sample else target_lengths.sum().item()
        )

        sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
        logging_output = {
            "loss": utils.item(loss.data),  # * sample['ntokens'],
            "ctc_loss": utils.item(ctc_loss.data),  # * sample['ntokens'],
            "distill_loss": utils.item(distill_loss.data),
            "ntokens": ntokens,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        if not model.training:
            import editdistance

            with torch.no_grad():
                lprobs_t = lprobs.transpose(0, 1).float().contiguous().cpu()

                c_err = 0
                c_len = 0
                w_errs = 0
                w_len = 0
                wv_errs = 0
                for lp, t, inp_l in zip(
                    lprobs_t,
                    sample["target_label"]
                    if "target_label" in sample
                    else sample["target"],
                    input_lengths,
                ):
                    lp = lp[:inp_l].unsqueeze(0)

                    decoded = None
                    if self.w2l_decoder is not None:
                        decoded = self.w2l_decoder.decode(lp)
                        if len(decoded) < 1:
                            decoded = None
                        else:
                            decoded = decoded[0]
                            if len(decoded) < 1:
                                decoded = None
                            else:
                                decoded = decoded[0]

                    p = (t != self.task.target_dictionary.pad()) & (
                        t != self.task.target_dictionary.eos()
                    )
                    targ = t[p]
                    targ_units = self.task.target_dictionary.string(targ)
                    targ_units_arr = targ.tolist()

                    toks = lp.argmax(dim=-1).unique_consecutive()
                    pred_units_arr = toks[toks != self.blank_idx].tolist()

                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)

                    targ_words = post_process(targ_units, self.post_process).split()

                    pred_units = self.task.target_dictionary.string(pred_units_arr)
                    pred_words_raw = post_process(pred_units, self.post_process).split()

                    if decoded is not None and "words" in decoded:
                        pred_words = decoded["words"]
                        w_errs += editdistance.eval(pred_words, targ_words)
                        wv_errs += editdistance.eval(pred_words_raw, targ_words)
                    else:
                        dist = editdistance.eval(pred_words_raw, targ_words)
                        w_errs += dist
                        wv_errs += dist

                    w_len += len(targ_words)

                logging_output["wv_errors"] = wv_errs
                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ctc_loss_sum = utils.item(sum(log.get("ctc_loss", 0) for log in logging_outputs))
        distill_loss_sum = utils.item(sum(log.get("distill_loss", 0) for log in logging_outputs))

        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "ctc_loss", ctc_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "distill_loss", distill_loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        metrics.log_scalar("ntokens", ntokens)
        metrics.log_scalar("nsentences", nsentences)
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_c_errors", c_errors)
        c_total = sum(log.get("c_total", 0) for log in logging_outputs)
        metrics.log_scalar("_c_total", c_total)
        w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_w_errors", w_errors)
        wv_errors = sum(log.get("wv_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_wv_errors", wv_errors)
        w_total = sum(log.get("w_total", 0) for log in logging_outputs)
        metrics.log_scalar("_w_total", w_total)

        if c_total > 0:
            metrics.log_derived(
                "uer",
                lambda meters: safe_round(
                    meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                )
                if meters["_c_total"].sum > 0
                else float("nan"),
            )
        if w_total > 0:
            metrics.log_derived(
                "wer",
                lambda meters: safe_round(
                    meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )
            metrics.log_derived(
                "raw_wer",
                lambda meters: safe_round(
                    meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                )
                if meters["_w_total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


def Linear(in_features, out_features, bias=True):
    m = torch.nn.Linear(in_features, out_features, bias)
    torch.nn.init.xavier_uniform_(m.weight)
    if bias:
        torch.nn.init.constant_(m.bias, 0.0)
    return m
