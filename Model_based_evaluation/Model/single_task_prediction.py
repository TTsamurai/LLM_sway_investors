import sys
import os

sys.path.append("../../Model")
import torch
import random
import transformers
import run_task
import numpy as np
import pandas as pd
import easydict
from argparse import ArgumentParser


def set_seed(seed):
    """Set the seed for reproducibility."""
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # PyTorch
    if (
        torch.cuda.is_available()
    ):  # If you're using PyTorch with a CUDA-capable device (GPU)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups


FILE_TYPES = [
    # "TextSequence",
    "ECT",
    # "gpt_summary",
    # "gpt_summary_overweight",
    # "gpt_summary_underweight",
    # "gpt_analysis_overweight",
    # "gpt_analysis_underweight",
    # "gpt_promotion_overweight",
    # "gpt_promotion_underweight",
    "gpt4_summary",
    # "gpt4_summary_overweight",
    # "gpt4_summary_underweight",
    # "gpt4_analysis_overweight",
    # "gpt4_analysis_underweight",
    # "gpt4_promotion_overweight",
    # "gpt4_promotion_underweight",
    # "analysis_underweight_and_overweight",
    # "summary_underweight_and_overweight",
    # "analysis_and_summary_underweight_and_overweight",
]
# GPT_FILE_TYPES = [
#     # "ECT", # Test typeがacl19ならこちら
#     # "gpt_summary",
#     # "gpt_summary_overweight",
#     # "gpt_summary_underweight",
#     # "gpt_analysis_overweight",
#     # "gpt_analysis_underweight",
#     # "gpt_promotion_overweight",
#     # "gpt_promotion_underweight",
#     # "gpt4_summary",
#     # "gpt4_summary_overweight",
#     # "gpt4_summary_underweight",
#     # "gpt4_analysis_overweight",
#     # "gpt4_analysis_underweight",
#     # "gpt4_promotion_overweight",
#     # "gpt4_promotion_underweight",
#     "gt_summary_bullet",
#     "gpt4_summary",
#     "gpt4_summary_overweight",
#     "gpt4_summary_underweight",
#     # "gpt4_analysis_overweight",
#     # "gpt4_analysis_underweight",
#     "gpt4_promotion_overweight",
#     "gpt4_promotion_underweight",
#     # "analysis_underweight_and_overweight",
#     # "summary_underweight_and_overweight",
#     # "analysis_and_summary_underweight_and_overweight",
# ]
GPT_FILE_TYPES = ["analyst_report"]
DURATIONS = [3, 7, 15, 30]
# DURATIONS = [3]
# EMBEDDINGS_TYPE = ["bert-base-uncased", "roberta-base", "ProsusAI/finbert"]
EMBEDDINGS_TYPE = [
    "bert-base-uncased",
    "roberta-base",
    "ProsusAI/finbert",
    "finbert-tone",
    "finbert-pretrain",
]
TEST_TYPE = ["ectsum", "acl19"]
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="stock_movement_prediction",
        choices=[
            "stock_price_prediction",
            "stock_movement_prediction",
            "volatility_prediction",
            "stock_return_prediction",
        ],
    )
    args = parser.parse_known_args()[0]
    # Convert args to a mutable EasyDict
    args = easydict.EasyDict(vars(args))
    # for train_normal_test_various in [False, True]:
    test_type = TEST_TYPE[0]
    for train_normal_test_various in [True]:
        for file_type in FILE_TYPES:
            # testデータだけ変更するとき
            if train_normal_test_various:
                if file_type not in ["TextSequence", "ECT", "gpt4_summary"]:
                    continue
            for embeddings_type in EMBEDDINGS_TYPE:
                for dur in DURATIONS:
                    best_alpha = {
                        "best": [],
                        "seed": [],
                        "prediction": [],
                        "actual": [],
                        "text file predict label": [],
                    }
                    various_best_alpha = {
                        other_file: {
                            "best": [],
                            "seed": [],
                            "prediction": [],
                            "actual": [],
                            "text file predict label": [],
                        }
                        for other_file in GPT_FILE_TYPES
                    }
                    # for seed in range(10):
                    for seed in range(10):
                        args.update(
                            {
                                "num_epochs": 20,  # 元々は5
                                "batch_size": 16,
                                "lr": 1e-5,  # 元々は 1e-4
                                "tb_dir": "./runs",
                                "final": False,
                                "max_pool": False,
                                "embedding_size": 768,  # 1024,  # 1024(textual feature)
                                "max_length": 520,
                                "num_heads": 2,
                                "depth": 2,
                                "seed": 1,
                                "lr_warmup": 1000,
                                "gradient_clipping": 1.0,
                                "file_name": file_type,
                                "data_dir": "../../Data/",
                                "embeddings_type": embeddings_type,
                                "input_dir": f"./ptm_embeddings/{embeddings_type}/{file_type}.npz",
                                "price_data_dir": "./price_data/",
                                # "alpha": i,
                                "gpu": True,
                                "save": False,
                                "duration": dur,
                                # "duration": 30,
                                "vocab_size": None,
                                "cuda_id": "0",
                                "train_normal_test_various": train_normal_test_various,
                                "test_type": test_type,
                            }
                        )
                        if args.train_normal_test_various:
                            evaluation, evaluation_various = run_gpu_single_task.go(
                                easydict.EasyDict(args)
                            )
                        else:
                            evaluation = run_gpu_single_task.go(easydict.EasyDict(args))

                        print(evaluation)
                        best_alpha["best"].append(evaluation["Test Loss"].iloc[0])
                        best_alpha["seed"].append(seed)
                        best_alpha["prediction"].append(evaluation["Outputs"].iloc[0])
                        best_alpha["actual"].append(evaluation["Actual"].iloc[0])
                        best_alpha["text file predict label"].append(
                            evaluation["Text File Predict Label"].iloc[0]
                        )

                        if args.train_normal_test_various:
                            for other_file, evaluation in evaluation_various.items():
                                various_best_alpha[other_file]["best"].append(
                                    evaluation["Test Loss"].iloc[0]
                                )
                                various_best_alpha[other_file]["seed"].append(seed)
                                various_best_alpha[other_file]["prediction"].append(
                                    evaluation["Outputs"].iloc[0]
                                )
                                various_best_alpha[other_file]["actual"].append(
                                    evaluation["Actual"].iloc[0]
                                )
                                various_best_alpha[other_file][
                                    "text file predict label"
                                ].append(evaluation["Text File Predict Label"].iloc[0])

                    best_alpha = pd.DataFrame(best_alpha)
                    best_alpha.sort_values(["best"], ascending=True, inplace=True)
                    if not args.train_normal_test_various:
                        save_dir = f"./results/{args.task}/{args.file_name}/{args.duration}/{args.embeddings_type}/"
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        best_alpha.to_csv(save_dir + f"result.csv")
                    else:
                        save_dir_base = f"./results/{test_type}/train_normal_test_various_lr_{str(args.lr)}/"
                        save_dir = os.path.join(
                            save_dir_base
                            + f"{args.task}/{args.file_name}/{args.file_name}/{args.duration}/{args.embeddings_type}/"
                        )
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        best_alpha.to_csv(save_dir + f"result.csv")
                        for (
                            other_file_type,
                            various_alpha,
                        ) in various_best_alpha.items():
                            save_dir = os.path.join(
                                save_dir_base,
                                f"{args.task}/{args.file_name}/{other_file_type}/{args.duration}/{args.embeddings_type}/",
                            )
                            best_alpha = pd.DataFrame(various_alpha)
                            best_alpha.sort_values(
                                ["best"], ascending=True, inplace=True
                            )
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                            best_alpha.to_csv(save_dir + f"result.csv")
