import os
from tqdm import tqdm
import argparse
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import json
from torch.utils.data import DataLoader
import ipdb

ENCODER_TYPE = [
    "bert-base-uncased",
    "roberta-base",
    "ProsusAI/finbert",
    "yiyanghkust/finbert-tone",
    "yiyanghkust/finbert-pretrain",
]
# ENCODER_TYPE = ["yiyanghkust/finbert-pretrain"]


def generate_embedding(args, tokenizer, model):
    data_list = os.listdir(args.data_path)
    if args.debug_mode:
        data_list = data_list[:10]
    output_dct = {}
    all_sent_num = []
    for data_dir in tqdm(data_list):
        output_dct[data_dir] = {}
        # text_path = args.data_path + data_dir + '/TextSequence.txt'
        text_path = os.path.join(
            args.data_path, data_dir, f"{args.file_type}.txt"
        )  # For ec
        text_file = open(text_path)
        all_sent_for_one_text = []
        for line in text_file.readlines():
            line = line.strip()
            all_sent_for_one_text += [line]
        all_sent_num += [len(all_sent_for_one_text)]
        try:
            input = tokenizer(
                all_sent_for_one_text,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
        except:
            print("wrong tokenize, sent_list_from: {}".format(data_dir))
            continue
        with torch.no_grad():
            model_out = model(input["input_ids"].cuda(), input["attention_mask"].cuda())
            # model_out = model(input['input_ids'].cuda(), input['token_type_ids'].cuda(), input['attention_mask'].cuda())
        # 今回は不要なので抜いておく，これで試してみるもありかも
        # last_hidden_out = model_out['last_hidden_state'].mean(1)
        # second_last_hidden_out = model_out['hidden_states'][-2].mean(1)
        out_for_on_text = model_out["pooler_output"]
        num_sent, ptm_size = out_for_on_text.shape
        # for key, emb in {'last_hidden_out': last_hidden_out, 'pooler_output': out_for_on_text}.items():
        for key, emb in {"pooler_output": out_for_on_text}.items():
            if args.max_sent > num_sent:
                emb = torch.cat(
                    [emb, torch.zeros(args.max_sent - num_sent, ptm_size).cuda()], dim=0
                )
            else:
                emb = emb[: args.max_sent, :]
            output_dct[data_dir] = emb.cpu().detach().numpy()

    if not args.not_save:
        if args.bert_type == "yiyanghkust/finbert-tone":
            args.bert_type = "finbert-tone"
        elif args.bert_type == "yiyanghkust/finbert-pretrain":
            args.bert_type = "finbert-pretrain"
        save_dir = os.path.join(args.save_path, args.bert_type)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if args.data_type == "ectsum":
            save_path = os.path.join(
                save_dir, "_".join([args.data_type, args.file_type + ".npz"])
            )
        else:
            save_path = os.path.join(save_dir, args.file_type + ".npz")
        np.savez(save_path, **output_dct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get embedding")

    parser.add_argument(
        "--ptm_model",
        default="roberta-large",
        type=str,
    )
    parser.add_argument(
        "--not_save", action="store_true", help="If set, do not save the output."
    )
    parser.add_argument(
        "--debug_mode", action="store_true", help="If set, run in debug mode."
    )
    parser.add_argument(
        "--data_type",
        default="acl19",
        type=str,
    )
    parser.add_argument("--max_sent", default=520, type=int)
    parser.add_argument(
        "--save_path",
        default="/home/m2021ttakayanagi/Documents/HTML-Hierarchical-Transformer-based-Multi-task-Learning-for-Volatility-Prediction/Data/ptm_embeddings/",
        type=str,
    )
    parser.add_argument(
        "--file_type",
        choices=[
            "TextSequence",
            "ECT",
            "gpt_summary",
            "gpt_summary_overweight",
            "gpt_summary_underweight",
            "gpt_analysis_overweight",
            "gpt_analysis_underweight",
        ],
        default="TextSequence",
        help="Type of the file to process (default: %(default)s)",
    )

    args = parser.parse_args()

    print(args)
    args.device = "cuda"
    assert args.data_type in ["ectsum", "acl19"], "Invalid data type."
    data_path_base = "/home/m2021ttakayanagi/Documents/HTML-Hierarchical-Transformer-based-Multi-task-Learning-for-Volatility-Prediction/Data/"
    if args.data_type == "acl19":
        args.data_path = os.path.join(data_path_base, "ACL19_Release/")
    elif args.data_type == "ectsum":
        args.data_path = os.path.join(data_path_base, "ectsum_ours_test")
    else:
        raise ValueError("Invalid data type.")

    if not args.debug_mode:
        for file_type in [
            "analyst_report"
            # # "TextSequence",
            # # "ECT",
            # "gt_summary_bullet",  # Ectsumのデータに対して
            # # "gpt_summary",
            # # "gpt_summary_overweight",
            # # "gpt_summary_underweight",
            # # "gpt_analysis_overweight",
            # # "gpt_analysis_underweight",
            # # "gpt_promotion_overweight",
            # # "gpt_promotion_underweight",
            # "gpt4_summary",
            # "gpt4_summary_overweight",
            # "gpt4_summary_underweight",
            # # "gpt4_analysis_overweight",
            # # "gpt4_analysis_underweight",
            # "gpt4_promotion_overweight",
            # "gpt4_promotion_underweight",
        ]:
            for bert_type in ENCODER_TYPE:
                tokenizer = AutoTokenizer.from_pretrained(bert_type)
                ptm = AutoModel.from_pretrained(bert_type).to(args.device)
                args.file_type = file_type
                args.bert_type = bert_type
                generate_embedding(args, tokenizer, ptm)
    # else:
    #     generate_embedding(args, tokenizer, ptm)
