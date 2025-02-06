import os
import pandas as pd
import ipdb
from survey_quantification import get_quantify_data
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score
import krippendorff

ROUND_NUMS = 30

# The first two are demo
PARTICIPANTS = [
    "5pukicue",
    "a5c9a7l5",
    # "b2a6kjz7",
    "d441lscl",
    "zy5illjx",
    "dxko1d46",
    "gfqxqayo",
    # "77v72ocw",
    "7rl2vvd0",
    "vyvfqtbl",
    "rc40anah",
    "5sgmhkv5",
    "nl01a1y9",
    "hzpxcb35",
    "n5kuo094",
]

SESSION_ID = {
    "A": [
        "nl01a1y9",
        "hzpxcb35",
        "n5kuo094",
    ],
    "B": [
        "vyvfqtbl",
        "rc40anah",
        "5sgmhkv5",
    ],
    "C": [
        "gfqxqayo",
        "77v72ocw",
        "7rl2vvd0",
    ],
    "D": [
        "d441lscl",
        "zy5illjx",
        "dxko1d46",
    ],
    "E": [
        "5pukicue",
        "a5c9a7l5",
        "b2a6kjz7",
    ],
}
ID_SESSION = {id_: session for session, ids in SESSION_ID.items() for id_ in ids}


# Time data preparation
def get_time_data(time_out):
    time_out = time_out[time_out["participant_code"].isin(PARTICIPANTS)]
    time_out["time"] = pd.to_datetime(time_out["epoch_time_completed"], unit="s")
    # time_out["time_diff_minutes"] = (
    #     time_out.groupby("participant_code")["time"].diff().dt.total_seconds() / 60
    # )
    time_out = time_out.query("app_name=='Task_Financial_Decision_Making'")
    filtered_df = time_out[time_out["page_name"].isin(["Wait_page", "Third_page"])]
    pivot_df = filtered_df.pivot_table(
        index=["participant_code", "round_number"],
        columns="page_name",
        values="time",
        aggfunc="first",
    )

    # Calculate the time difference in minutes
    pivot_df["time_diff_minutes"] = (
        pivot_df["Third_page"] - pivot_df["Wait_page"]
    ).dt.total_seconds() / 60

    # Reset index to turn MultiIndex into columns
    pivot_df.reset_index(inplace=True)
    pivot_df.rename(
        columns={"round_number": "round", "participant_code": "participant_id"},
        inplace=True,
    )
    return pivot_df


def save_time_data(time_df, save_file="time_diff_histogram.png"):
    plt.figure()  # Create a new figure
    time_df["time_diff_minutes"].hist()
    # Adding title
    plt.title("The Histogram of Time Used for Each Period")
    # Save the histogram
    plt.savefig((f"./output/{save_file}"))


# Data Preparation
def quantify_financial_decision(df):
    financial_decision_num = {"Increase (上昇)": 1, "Decrease (下降)": 0}
    for round in range(1, ROUND_NUMS + 1):
        df[
            f"Task_Financial_Decision_Making.{round}.player.first_financial_decision"
        ] = df[
            f"Task_Financial_Decision_Making.{round}.player.first_financial_decision"
        ].apply(
            lambda x: financial_decision_num.get(x)
        )
        df[
            f"Task_Financial_Decision_Making.{round}.player.final_financial_decision"
        ] = df[
            f"Task_Financial_Decision_Making.{round}.player.final_financial_decision"
        ].apply(
            lambda x: financial_decision_num.get(x)
        )
    return df


def take_task_data(df):
    task_data_list = [
        "first_doc",
        "second_doc",
        "first_financial_decision",
        "final_financial_decision",
        "Grammaticality",
        "Convincing",
        "Logical_causality",
        "Usefulness",
        "Question",
    ]
    column_necessary = ["participant.code"]
    new_column = ["participant_id"]
    for round in range(1, 31):
        base_name = f"Task_Financial_Decision_Making.{round}.player."
        for task_data in task_data_list:
            column_necessary.append(base_name + task_data)
            new_column.append(f"{round}_{task_data}")
    return df.filter(items=column_necessary).rename(
        columns=dict(zip(column_necessary, new_column))
    )


def format_long_format(task_data):
    # Splitting the task_variable column into separate columns
    # Assuming `task_data` is your DataFrame
    long_format = task_data.melt(
        id_vars=["participant_id"], var_name="task_variable", value_name="value"
    )
    long_format[["round", "task"]] = long_format["task_variable"].str.split(
        "_", expand=True, n=1
    )
    # Converting task number to integer
    long_format["round"] = long_format["round"].astype(int)
    return long_format.drop(columns=["task_variable"])


def format_wide_data(long_data):
    # pivot tableとassertion
    grouped_data = long_data.groupby(["participant_id", "round"]).size()
    duplicates = grouped_data[grouped_data > 9]
    assert duplicates.empty, f"There are duplicates in the data: {duplicates}"
    wide_data = long_data.pivot_table(
        index=["participant_id", "round"],
        columns="task",
        values="value",
        aggfunc="first",
        dropna=False,
    )
    wide_data.reset_index(inplace=True)
    return wide_data


def merge_wide_and_document_info(wide_data, document_info):
    # Document infoをwide_dataにmerge
    wide_data["session_group"] = wide_data["participant_id"].map(ID_SESSION)
    merged_data = pd.merge(
        wide_data,
        document_info,
        left_on=["session_group", "round"],
        right_on=["source", "round"],
        how="inner",
    )
    # merged_data.drop(
    #     columns=["first_anonymized", "second_anonymized", "first_doc", "second_doc"],
    #     inplace=True,
    # )
    merged_data.drop(
        columns=["first_doc", "second_doc"],
        inplace=True,
    )
    return merged_data


def score_decision(merged_human_data, which_decision):
    assert which_decision in ["first_financial_decision", "final_financial_decision"]
    if which_decision == "first_financial_decision":
        return (
            merged_human_data.future_label_3
            == merged_human_data.first_financial_decision
        ) * 1
    else:
        return (
            merged_human_data.future_label_3
            == merged_human_data.final_financial_decision
        ) * 1


# Data Analysis
# Analyst the differences between the first decision and the second decision
def take_difference_round(df, round):
    financial_decision_num = {"Increase (上昇)": 1, "Decrease (下降)": 0}
    first_decision = df[
        f"Task_Financial_Decision_Making.{round}.player.first_financial_decision"
    ].apply(lambda x: financial_decision_num.get(x))
    second_decision = df[
        f"Task_Financial_Decision_Making.{round}.player.final_financial_decision"
    ].apply(lambda x: financial_decision_num.get(x))
    return second_decision - first_decision


def get_merged_data(file_path, document_path):
    data = pd.read_csv(file_path)
    document_info = pd.read_csv(document_path)
    human_data = get_quantify_data(file_path)
    data = data[data["participant.code"].isin(PARTICIPANTS)]
    quantified_data = quantify_financial_decision(data)
    task_data = take_task_data(quantified_data)
    long_data = format_long_format(task_data)
    wide_data = format_wide_data(long_data)
    merged_data = merge_wide_and_document_info(wide_data, document_info)
    # Human dataとmergeする
    merged_human_data = pd.merge(
        merged_data, human_data, on="participant_id", how="left"
    )

    # 正解を計算
    merged_human_data["score"] = score_decision(merged_human_data)
    return merged_human_data


def get_annotation_difference_veteran_expert_students(analysis_data, standardize=False):
    # 分析．専門家・非専門家からのAnalyst Report, Summary, Promotionに対するレポートの評価
    # 2 * 3の六パターンできる
    # 結果．専門家の方が非専門家よりもAnalyst Report, Summaryを高い評価．学生はPromotionを高く評価．
    # 例）
    if standardize:
        cols = [
            "standard_Grammaticality",
            "standard_Convincing",
            "standard_Logical_causality",
            "standard_Usefulness",
        ]
    else:
        cols = [
            "Grammaticality",
            "Convincing",
            "Logical_causality",
            "Usefulness",
        ]
    mean_promotion_with_vet = (
        analysis_data.query(q_promotion).query(q_veteran_experience)[cols].mean()
    )
    mean_summary_with_vet = (
        analysis_data.query(q_summary).query(q_veteran_experience)[cols].mean()
    )
    mean_report_with_vet = (
        analysis_data.query(q_analyst_report).query(q_veteran_experience)[cols].mean()
    )

    mean_promotion_with_exp = (
        analysis_data.query(q_promotion).query(q_financial_experience)[cols].mean()
    )
    mean_summary_with_exp = (
        analysis_data.query(q_summary).query(q_financial_experience)[cols].mean()
    )
    mean_report_with_exp = (
        analysis_data.query(q_analyst_report).query(q_financial_experience)[cols].mean()
    )
    mean_promotion_without_exp = (
        analysis_data.query(q_promotion).query(q_no_financial_experience)[cols].mean()
    )
    mean_summary_without_exp = (
        analysis_data.query(q_summary).query(q_no_financial_experience)[cols].mean()
    )
    mean_report_without_exp = (
        analysis_data.query(q_analyst_report)
        .query(q_no_financial_experience)[cols]
        .mean()
    )

    # Create a DataFrame
    data = {
        "Document Type": [
            "Analyst Report",
            "Summary",
            "Promotion",
            "Analyst Report",
            "Summary",
            "Promotion",
            "Analyst Report",
            "Summary",
            "Promotion",
        ],
        "Financial Experience": [
            "Veteran",
            "Veteran",
            "Veteran",
            "With Experience",
            "With Experience",
            "With Experience",
            "Without Experience",
            "Without Experience",
            "Without Experience",
        ],
        "Mean Annotations": [
            mean_report_with_vet,  # Analyst Report with financial experience
            mean_summary_with_vet,  # Summary with financial experience
            mean_promotion_with_vet,  # Promotion with financial experience
            mean_report_with_exp,  # Analyst Report with financial experience
            mean_summary_with_exp,  # Summary with financial experience
            mean_promotion_with_exp,  # Promotion with financial experience
            mean_report_without_exp,  # Analyst Report without financial experience
            mean_summary_without_exp,  # Summary without financial experience
            mean_promotion_without_exp,  # Promotion without financial experience
        ],
    }

    result_annotations = pd.DataFrame(data)

    # Pivot the data for plotting
    pivot_df = result_annotations.pivot(
        index="Document Type", columns="Financial Experience", values="Mean Annotations"
    )
    return pivot_df


def plot_annotation_distribution(annotation_name, analysis_data, save_dir="./output"):
    # Create histograms
    students = analysis_data.query(q_no_financial_experience)[annotation_name]
    experts = analysis_data.query(q_financial_experience)[annotation_name]

    plt.figure(figsize=(10, 6))

    # Plot normalized histograms
    n_bins = range(1, 7)  # Bins from 1 to 6
    weights_students = [100 / len(students)] * len(students)
    weights_experts = [100 / len(experts)] * len(experts)

    plt.hist(
        students,
        bins=n_bins,
        weights=weights_students,
        alpha=0.5,
        label="Students",
        color="blue",
        edgecolor="black",
    )
    plt.hist(
        experts,
        bins=n_bins,
        weights=weights_experts,
        alpha=0.5,
        label="Experts",
        color="green",
        edgecolor="black",
    )

    # Add title and labels
    plt.title(f"Distribution of {annotation_name} Scores")
    plt.xlabel(f"{annotation_name} Score")
    plt.ylabel("Percentage of Responses")
    plt.legend()

    # Ensure directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the figure
    save_path = os.path.join(save_dir, f"{annotation_name}_distribution.png")
    plt.savefig(save_path)

    # Show the plot (optional, for visual confirmation)
    plt.show()


def get_top_n_most_convincing_data(analysis_data, n=5):
    annotation_num_data = [
        "Grammaticality",
        "Convincing",
        "Logical_causality",
        "Usefulness",
    ]
    most_convincing = (
        analysis_data[annotation_num_data + ["second_anonymized"]]
        .groupby("second_anonymized")
        .mean()
        .sort_values(by="Convincing", ascending=False)
    ).head(n)
    return pd.merge(
        most_convincing,
        analysis_data[
            ["second_anonymized", "recommendation", "second_document_type"]
        ].drop_duplicates(),
        left_index=True,
        right_on="second_anonymized",
        how="inner",
    )


def plot_and_save(df, col, path):
    weights = pd.Series([1] * len(df))
    plt.figure(figsize=(8, 6))
    # df[col].hist(bins=10, alpha=0.7, color="blue", edgecolor="black")
    df[col].hist(bins=5, alpha=0.7, color="blue", edgecolor="black", weights=weights)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(False)
    plt.savefig(path)


def personality_high_low(analysis_data):
    # Define personality columns
    personality_cols = [
        "extraversion",
        "agreeableness",
        "conscientiousness",
        "neuroticism",
        "openness",
    ]

    # Dictionary to store results
    results = {}

    # Iterate over each personality trait to calculate median and categorize
    for person in personality_cols:
        mid = analysis_data[person].median()
        analysis_data[f"{person}_high"] = (analysis_data[person] >= mid).astype(int)
        analysis_data[f"{person}_low"] = (analysis_data[person] < mid).astype(int)

        # Calculate means for high and low groups
        high_difference = analysis_data.loc[
            analysis_data[f"{person}_high"] == 1, "difference"
        ].mean()
        high_up_difference = analysis_data.loc[
            analysis_data[f"{person}_high"] == 1, "up_difference"
        ].mean()
        high_down_difference = analysis_data.loc[
            analysis_data[f"{person}_high"] == 1, "down_difference"
        ].mean()
        low_difference = analysis_data.loc[
            analysis_data[f"{person}_low"] == 1, "difference"
        ].mean()
        low_up_difference = analysis_data.loc[
            analysis_data[f"{person}_low"] == 1, "up_difference"
        ].mean()
        low_down_difference = analysis_data.loc[
            analysis_data[f"{person}_low"] == 1, "down_difference"
        ].mean()

        # Store results in dictionary
        results[person] = {
            "high_difference": high_difference,
            "high_up_difference": high_up_difference,
            "high_down_difference": high_down_difference,
            "low_difference": low_difference,
            "low_up_difference": low_up_difference,
            "low_down_difference": low_down_difference,
        }

    # Print the results
    for personality, scores in results.items():
        print(f"Results for {personality.capitalize()}:")
        for score_type, value in scores.items():
            print(f"{score_type}: {value:.2f}")
        print()


def investor_difference_predict_difference(data, queries):
    for description, query_string in queries.items():
        try:
            # Execute the query
            filtered_data = data.query(query_string)

            # Calculate means
            difference_mean = filtered_data["difference"].mean()
            up_difference_mean = filtered_data["up_difference"].mean()
            down_difference_mean = filtered_data["down_difference"].mean()

            # Print results
            print(f"Results for {description}:")
            print(f"Mean Difference: {difference_mean:.2f}")
            print(f"Mean Up Difference: {up_difference_mean:.2f}")
            print(f"Mean Down Difference: {down_difference_mean:.2f}")
            print()
        except Exception as e:
            print(f"An error occurred while processing {description}: {str(e)}")


def investor_document_predict_difference(data, queries):
    for description, query_string in queries.items():
        try:
            # Execute the query
            filtered_data = data.query(query_string)

            # Calculate means
            analyst_mean = filtered_data.query(q_analyst_report)["difference"].mean()
            summary_mean = filtered_data.query(q_summary)["difference"].mean()
            promotion_mean = filtered_data.query(q_promotion)["difference"].mean()

            # Print results
            print(f"Results for {description}:")
            print(f"Mean Analyst Report Difference: {analyst_mean:.2f}")
            print(f"Mean Summary Difference: {summary_mean:.2f}")
            print(f"Mean Promotion Difference: {promotion_mean:.2f}")
            print()
        except Exception as e:
            print(f"An error occurred while processing {description}: {str(e)}")


def print_question_participant_id(data, participant_id):
    for i in data.query(f"participant_id == '{participant_id}'")["Question"].values:
        print(i)
        print("\n")


def print_question_pair_id(data, pair_id):
    data = data.query(f"pair_id == {pair_id}")
    data.sort_values(by="financial_working_years", inplace=True)
    for i, row in data.iterrows():
        print(row["participant_id"])
        print(row["Question"])
        print("\n")
    print(data.query(f"pair_id == {pair_id}")["first_document_type"].values[0])
    print(data.query(f"pair_id == {pair_id}")["second_document_type"].values[0])
    print(data.query(f"pair_id == {pair_id}")["recommendation"].values[0])


def correlation_annotation(df):
    # Convert the columns to numeric types if they are not already
    df["Convincing"] = pd.to_numeric(df["Convincing"], errors="coerce")
    df["Grammaticality"] = pd.to_numeric(df["Grammaticality"], errors="coerce")
    df["Logical_causality"] = pd.to_numeric(df["Logical_causality"], errors="coerce")
    df["Usefulness"] = pd.to_numeric(df["Usefulness"], errors="coerce")

    # Calculate the correlation matrix again
    correlation_matrix = df.corr()
    return correlation_matrix


# analysis_dataとの融合
def get_sentiment_score_row(row, ect_sentiment, acl_sentiment):
    change = {
        "summary": "gpt4_summary",
        "promotion": "gpt4_promotion",
        "analyst": "analyst_report",
    }

    positive_query = "_".join(
        [
            "positive_score",
            change.get(row["second_document_type"], row["second_document_type"]),
        ]
    )
    negative_query = "_".join(
        [
            "negative_score",
            change.get(row["second_document_type"], row["second_document_type"]),
        ]
    )
    if "analyst" not in row["second_document_type"]:
        positive_query = "_".join([positive_query, row["recommendation"]])
        negative_query = "_".join([negative_query, row["recommendation"]])
    if row["company_name"] in ect_sentiment.index:
        positive_score = ect_sentiment.loc[row["company_name"], positive_query]
        negative_score = ect_sentiment.loc[row["company_name"], negative_query]

    else:
        positive_score = acl_sentiment.loc[row["company_name"], positive_query]
        negative_score = acl_sentiment.loc[row["company_name"], negative_query]
    return positive_score, negative_score


def get_sentiment(analysis_data, sentiment_type="LM_sentiment_scores.csv"):
    sentiment_data_dir = "/home/m2021ttakayanagi/Documents/FinancialOpinionGeneration/EMNLP2024/preliminary_output"
    sentiment_ect_path = os.path.join(sentiment_data_dir, sentiment_type)
    sentiment_acl_path = os.path.join(
        sentiment_data_dir, "LM_sentiment_scores_acl2019.csv"
    )
    ect_sentiment = pd.read_csv(sentiment_ect_path, index_col=0).filter(regex="score")
    acl_sentiment = pd.read_csv(sentiment_acl_path, index_col=0).filter(regex="score")
    analysis_data[["positive_score", "negative_score"]] = analysis_data.apply(
        lambda row: pd.Series(
            get_sentiment_score_row(
                row, ect_sentiment=ect_sentiment, acl_sentiment=acl_sentiment
            )
        ),
        axis=1,
    )
    return analysis_data


def standarize_annotation_among_participants(df):
    annotation_num_data = [
        "Grammaticality",
        "Convincing",
        "Logical_causality",
        "Usefulness",
    ]
    standarized_num_data = [
        "standard_Grammaticality",
        "standard_Convincing",
        "standard_Logical_causality",
        "standard_Usefulness",
    ]
    df[standarized_num_data] = (
        df[annotation_num_data + ["participant_id"]]
        .groupby("participant_id")
        .transform(
            lambda x: StandardScaler().fit_transform(x.values[:, np.newaxis]).ravel()
        )
    )
    return df


def second_document_stance(analysis_data):
    document_types = {
        "Summary": q_summary,
        "Promotion": q_promotion,
        "Analyst Report": q_analyst_report,
    }
    investment_stances = {
        "Both": None,
        "Overweight": q_overweight,
        "Underweight": q_underweight,
    }
    results = []
    for doc_type, doc_query in document_types.items():
        for stance, stance_query in investment_stances.items():
            if stance == "Both":
                mean_difference = analysis_data.query(doc_query)["difference"].mean()
            else:
                mean_difference = (
                    analysis_data.query(doc_query)
                    .query(stance_query)["difference"]
                    .mean()
                )
            results.append(
                {
                    "Document Type": doc_type,
                    "Investment Stance": stance,
                    "Mean Difference": mean_difference,
                }
            )
    return pd.DataFrame(results)


def first_document_stance(analysis_data):
    document_types = {
        "GT Summary": q_gt_summary,
        "GPT Summary": q_gpt4_summary,
    }
    investment_stances = {
        "Both": None,
        "Overweight": q_overweight,
        "Underweight": q_underweight,
    }
    results = []
    for doc_type, doc_query in document_types.items():
        for stance, stance_query in investment_stances.items():
            if stance == "Both":
                mean_difference = analysis_data.query(doc_query)["difference"].mean()
            else:
                mean_difference = (
                    analysis_data.query(doc_query)
                    .query(stance_query)["difference"]
                    .mean()
                )
            results.append(
                {
                    "Document Type": doc_type,
                    "Investment Stance": stance,
                    "Mean Difference": mean_difference,
                }
            )
    return pd.DataFrame(results)


def first_and_second_document_type(analysis_data):
    first_document_types = {
        "GT Summary": q_gt_summary,
        "GPT Summary": q_gpt4_summary,
    }
    second_document_types = {
        "Summary": q_summary,
        "Promotion": q_promotion,
        "Analyst Report": q_analyst_report,
    }
    investment_stances = {
        "Both": None,
        "Overweight": q_overweight,
        "Underweight": q_underweight,
    }
    results = []
    for first_doc_type, first_doc_query in first_document_types.items():
        for second_doc_type, second_doc_query in second_document_types.items():
            for stance, stance_query in investment_stances.items():
                if stance == "Both":
                    mean_difference = (
                        analysis_data.query(first_doc_query)
                        .query(second_doc_query)["difference"]
                        .mean()
                    )
                else:
                    mean_difference = (
                        analysis_data.query(first_doc_query)
                        .query(second_doc_query)
                        .query(stance_query)["difference"]
                        .mean()
                    )
                results.append(
                    {
                        "First Document Type": first_doc_type,
                        "Second Document Type": second_doc_type,
                        "Investment Stance": stance,
                        "Mean Difference": mean_difference,
                    }
                )
    return pd.DataFrame(results)


def custom_format(x):
    if x == 0:
        return "0.000"
    else:
        return "{:.3g}".format(x)


def print_agreement(ids, name):
    annotations = {}
    for ann in annotation_num_data:
        annotation_missing = pd.DataFrame(np.arange(1, 76), columns=["pair_id"])
        for ex in ids:
            annotation_missing = annotation_missing.merge(
                annotation_data.query(f"participant_id=='{ex}'")[
                    ["pair_id", ann]
                ].rename(columns={ann: ex}),
                on="pair_id",
                how="left",
            )
        annotations[ann] = annotation_missing
    for ann, val in annotations.items():
        alpha = krippendorff.alpha(
            reliability_data=val.drop(columns="pair_id").values.astype("float"),
            level_of_measurement="ordinal",
        )
        print(f"Among {name}: {ann} Krippendorf Alpha: {alpha}")


def annotations_diff(analysis_data, diffs=True):
    if diffs:
        analysis_data = analysis_data.query("difference==1")
    else:
        analysis_data = analysis_data.query("difference==0")
    investor_types = {
        "Overall": None,
        "Student": q_no_financial_experience,
        "Expert": q_financial_experience,
        "Veteran": q_veteran_experience,
    }
    cols = annotation_num_data_standarized + [
        "sum_sentiment",
        "positive_score",
        "negative_score",
    ]
    res = pd.DataFrame()

    for investor_type, investor_query in investor_types.items():
        if investor_type == "Overall":
            mean_data = analysis_data[cols].mean()
        else:
            mean_data = analysis_data.query(investor_query)[cols].mean()

        # Convert Series to DataFrame and transpose
        mean_data = pd.DataFrame(mean_data).transpose()
        # Assigning column names after transposing
        mean_data.columns = cols
        # Naming the index with the type of investor
        mean_data.index = [investor_type]

        # Concatenate along columns
        res = pd.concat([res, mean_data], axis=0)
    return res


pd.options.display.float_format = custom_format
if __name__ == "__main__":
    file_path = "./data/all_apps_wide_2024-06-01.csv"
    time_out_path = "./data/PageTimes-2024-06-01.csv"
    document_path = "./data/all_document_with_price_data.csv"
    data = pd.read_csv(file_path)
    time_out = pd.read_csv(time_out_path)
    time_df = get_time_data(time_out)
    document_info = pd.read_csv(document_path)
    human_data = get_quantify_data(file_path)
    human_data = human_data[
        human_data["participant_id"].isin(PARTICIPANTS)
    ].reset_index(drop=True)
    data = data[data["participant.code"].isin(PARTICIPANTS)]
    time_out = time_out[time_out["participant_code"].isin(PARTICIPANTS)]
    # それぞれのラウンドで分析に使いそうなデータ, Data Preparation
    # first_finanial_decision, second_financial_decision, Grammaticality, Convincing, Logical_causality, Usefulness, Question
    quantified_data = quantify_financial_decision(data)
    task_data = take_task_data(quantified_data)
    long_data = format_long_format(task_data)
    wide_data = format_wide_data(long_data)
    merged_data = merge_wide_and_document_info(wide_data, document_info)

    # Human dataとmergeする
    merged_human_data = pd.merge(
        merged_data, human_data, on="participant_id", how="left"
    )

    # Time dataとmergeする
    merged_human_data = pd.merge(
        merged_human_data, time_df, on=["participant_id", "round"], how="left"
    )
    # 時間をTHRESHOLDでフィルタリング
    # THRESHOLD = 2
    # merged_human_data = merged_human_data[merged_human_data["time_diff_minutes"] > THRESHOLD]

    # 正解を計算
    merged_human_data["first_score"] = score_decision(
        merged_human_data, "first_financial_decision"
    )
    merged_human_data["final_score"] = score_decision(
        merged_human_data, "final_financial_decision"
    )

    # Analysis

    # validなデータのみをデータのみを取得
    ANNOTATIONS = [
        "Convincing",
        "Grammaticality",
        "Logical_causality",
        "Question",
        "Usefulness",
    ]
    DECISIONS = ["first_financial_decision", "final_financial_decision"]
    DOCUMENT = ["rank", "first_document_type", "second_document_type", "recommendation"]

    # validなデータのみをデータのみを取得
    analysis_data = merged_human_data[
        merged_human_data["final_financial_decision"].notnull()
    ].reset_index(drop=True)

    # スコアを集計
    analysis_data["difference"] = (
        (
            analysis_data["final_financial_decision"]
            - analysis_data["first_financial_decision"]
        )
        .abs()
        .astype("int")
    )
    analysis_data["recommendation"] = analysis_data["recommendation"].str.lower()
    raw_difference = (
        analysis_data["final_financial_decision"]
        - analysis_data["first_financial_decision"]
    )
    analysis_data["raw_difference"] = raw_difference.values.astype("int")
    analysis_data["up_difference"] = raw_difference.apply(lambda x: 1 if x > 0 else 0)
    analysis_data["down_difference"] = raw_difference.apply(lambda x: 1 if x < 0 else 0)
    # sentimentを加える (psotive_score, negative_score)
    analysis_data = get_sentiment(
        analysis_data, sentiment_type="LM_sentiment_scores.csv"
    )
    analysis_data = standarize_annotation_among_participants(analysis_data)

    # Query

    # 専門家・非専門家
    q_veteran_experience = "financial_working_years >= 10"
    q_financial_experience = "financial_experience_flag == 1"
    q_no_financial_experience = "financial_experience_flag == 0"
    # First Documentの種類に関して
    q_gt_summary = "first_document_type == 'gt_summary'"
    q_gpt4_summary = "first_document_type == 'gpt4_summary'"
    # Second DocumentのInvestment Stance
    q_overweight = "recommendation == 'overweight'"
    q_underweight = "recommendation == 'underweight'"
    # Second Documentの種類に対して
    q_analyst_report = "second_document_type == 'analyst'"
    q_promotion = "second_document_type == 'promotion'"
    q_summary = "second_document_type == 'summary'"
    # DocumentのRank
    q_rank_top = "rank == 'top'"
    q_rank_bottom = "rank == 'bottom'"
    q_rank_none = "@pd.isna(rank)"

    # 時間の可視化
    save_time_data(time_df)

    # 予測精度に対して
    analysis_data.query(q_financial_experience)["first_score"].mean()
    analysis_data.query(q_financial_experience)["final_score"].mean()
    analysis_data.query(q_veteran_experience)["first_score"].mean()
    analysis_data.query(q_no_financial_experience)["first_score"].mean()
    analysis_data.query(q_no_financial_experience)["final_score"].mean()
    analysis_data.query(q_veteran_experience)["final_score"].mean()

    # 予測変化に対して
    # 分析 Second DocumentがOVerweightとUnderweightの時にどのような予測変化が起こっているか
    # 2パターン
    # 結果 Underweightの方が動きやすい．結果は同じで，Earning conference callsはそもそもPositiveに書かれていることが多いから．
    analysis_data.query(q_underweight)["difference"].mean()
    analysis_data.query(q_overweight)["difference"].mean()

    # 分析 Second Document Typeにより予測の変化に違いが生じるか
    analysis_data.query(q_analyst_report)["difference"].mean()
    analysis_data.query(q_promotion)["difference"].mean()
    analysis_data.query(q_summary)["difference"].mean()

    both = second_document_stance(analysis_data).rename(
        columns={"Mean Difference": "Overall"}
    )
    students = second_document_stance(
        analysis_data.query(q_no_financial_experience)
    ).rename(columns={"Mean Difference": "Student"})
    experts = second_document_stance(
        analysis_data.query(q_financial_experience)
    ).rename(columns={"Mean Difference": "Expert"})
    veterans = second_document_stance(analysis_data.query(q_veteran_experience)).rename(
        columns={"Mean Difference": "Veteran"}
    )
    both.merge(students, on=["Document Type", "Investment Stance"]).merge(
        experts, on=["Document Type", "Investment Stance"]
    ).merge(veterans, on=["Document Type", "Investment Stance"])

    # 分析 First Document TYpeにより予測の傾向に違いが生じるか
    analysis_data.query(q_gt_summary)["difference"].mean()
    analysis_data.query(q_gpt4_summary)["difference"].mean()

    both = first_document_stance(analysis_data).rename(
        columns={"Mean Difference": "Overall"}
    )
    students = first_document_stance(
        analysis_data.query(q_no_financial_experience)
    ).rename(columns={"Mean Difference": "Student"})
    experts = first_document_stance(analysis_data.query(q_financial_experience)).rename(
        columns={"Mean Difference": "Expert"}
    )
    veterans = first_document_stance(analysis_data.query(q_veteran_experience)).rename(
        columns={"Mean Difference": "Veteran"}
    )
    both.merge(students, on=["Document Type", "Investment Stance"]).merge(
        experts, on=["Document Type", "Investment Stance"]
    ).merge(veterans, on=["Document Type", "Investment Stance"])

    # FirstとSecond両方
    both = first_and_second_document_type(analysis_data).rename(
        columns={"Mean Difference": "Overall"}
    )
    students = first_and_second_document_type(
        analysis_data.query(q_no_financial_experience)
    ).rename(columns={"Mean Difference": "Student"})
    experts = first_and_second_document_type(
        analysis_data.query(q_financial_experience)
    ).rename(columns={"Mean Difference": "Expert"})
    veterans = first_and_second_document_type(
        analysis_data.query(q_veteran_experience)
    ).rename(columns={"Mean Difference": "Veteran"})
    both.merge(
        students,
        on=["First Document Type", "Second Document Type", "Investment Stance"],
    ).merge(
        experts, on=["First Document Type", "Second Document Type", "Investment Stance"]
    ).merge(
        veterans,
        on=["First Document Type", "Second Document Type", "Investment Stance"],
    )

    ######################################Statistics########################################
    # 投資家のStatistics

    # Distributionをとる
    dist_cols = [
        "age",
        "financial_working_years",
        "risk_tolerance",
        "cognitive_test",
        "extraversion",
        "agreeableness",
        "conscientiousness",
        "neuroticism",
        "openness",
    ]
    out_dir = "./statistics_output/investor_statistics"
    for var_col in dist_cols:
        plot_and_save(analysis_data, var_col, f"{out_dir}/{var_col}_distribution.png")

    # 文書のStatistics

    ###########################################################################################

    ######################################投資家属性と投資家行動の比較#########################################
    # 投資家の行動：Accuracy, Prediction (Up, Down, Stay), Annotations
    # 投資家の分類：Experience, risk tolerance, cognitive test, personality

    # Investor difference and prediction accuracy
    # Experience
    no_financial_experience_first = analysis_data.query(q_no_financial_experience)[
        "first_score"
    ].mean()
    no_financial_experience_final = analysis_data.query(q_no_financial_experience)[
        "final_score"
    ].mean()
    financial_experience_first = analysis_data.query(q_financial_experience)[
        "first_score"
    ].mean()
    financial_experience_final = analysis_data.query(q_financial_experience)[
        "final_score"
    ].mean()
    veteran_experience_first = analysis_data.query(q_veteran_experience)[
        "first_score"
    ].mean()
    veteran_experience_final = analysis_data.query(q_veteran_experience)[
        "final_score"
    ].mean()

    # Cognitive Test
    cognitive_test_2_first = analysis_data.query("cognitive_test == 2")[
        "first_score"
    ].mean()
    cognitive_test_2_final = analysis_data.query("cognitive_test == 2")[
        "final_score"
    ].mean()
    cognitive_test_3_first = analysis_data.query("cognitive_test == 3")[
        "first_score"
    ].mean()
    cognitive_test_3_final = analysis_data.query("cognitive_test == 3")[
        "final_score"
    ].mean()

    # # Printing results
    # print("Experience")
    # print(f"No Financial Experience - First Score: {no_financial_experience_first}")
    # print(f"No Financial Experience - Final Score: {no_financial_experience_final}")
    # print(f"Financial Experience - First Score: {financial_experience_first}")
    # print(f"Financial Experience - Final Score: {financial_experience_final}")
    # print(f"Veteran Experience - First Score: {veteran_experience_first}")
    # print(f"Veteran Experience - Final Score: {veteran_experience_final}")

    # print("\nCognitive Test")
    # print(f"Cognitive Test == 2 - First Score: {cognitive_test_2_first}")
    # print(f"Cognitive Test == 2 - Final Score: {cognitive_test_2_final}")
    # print(f"Cognitive Test == 3 - First Score: {cognitive_test_3_first}")
    # print(f"Cognitive Test == 3 - Final Score: {cognitive_test_3_final}")

    # Investor difference and prediction difference
    # Personality
    # personality_high_low(analysis_data=analysis_data.query(q_no_financial_experience))

    # Experience
    queries_investor_difference_prediction_difference = {
        "No Financial Experience": q_no_financial_experience,
        "Financial Experience": q_financial_experience,
        "Veteran Experience": q_veteran_experience,
        "High Cognitive Ability": "cognitive_test == 3",
        "Low Cognitive Ability": "cognitive_test == 2",
        "High Risk Tolerance": "risk_tolerance >= 2",
        "Low Risk Tolerance": "risk_tolerance < 2",
    }
    # investor_difference_predict_difference(
    #     analysis_data, queries_investor_difference_prediction_difference
    # )

    # Investor difference and annotations
    # 分析．専門家・非専門家からのAnalyst Report, Summary, Promotionに対するレポートの評価
    # 2 * 3の六パターンできる
    # 結果．専門家の方が非専門家よりもAnalyst Report, Summaryを高い評価．学生はPromotionを高く評価．
    # 例）
    expert_students_annotations = get_annotation_difference_veteran_expert_students(
        analysis_data
    )
    expert_students_annotations_standarized = (
        get_annotation_difference_veteran_expert_students(
            analysis_data, standardize=True
        )
    )
    # expert_students_annotations.to_csv("./output/expert_students_annotations.csv")
    # expert_students_annotations_standarized.to_csv(
    #     "./output/expert_students_annotations_standarized.csv"
    # )
    # Annotationデータの分布の可視化
    # 例）Convincingの分布
    annotation_num_data = [
        "Grammaticality",
        "Convincing",
        "Logical_causality",
        "Usefulness",
    ]
    annotation_num_data_standarized = [
        "standard_Grammaticality",
        "standard_Convincing",
        "standard_Logical_causality",
        "standard_Usefulness",
    ]
    # ここをコメントアウトすると，全てのAnnotationの分布を可視化する
    # for ann in annotation_num_data + annotation_num_data_standarized:
    #     plot_annotation_distribution(ann, analysis_data)

    ###########################################################################################

    ######################################文書属性と投資家行動の比較#########################################

    # 文書属性：Document Type, Rank, Recommendation
    queries_investor_document_predict_difference = {
        "No financial experience": q_no_financial_experience,
        "Financial experience": q_financial_experience,
        "Veteran experience": q_veteran_experience,
    }
    # investor_document_predict_difference(
    #     analysis_data, queries_investor_document_predict_difference
    # )
    ###########################################################################################

    ######################################Investor ExperimentとModel Experimentの比較#########################################
    # Expertに最もConvincingなDocumentを取ってくる
    # 分析　どのようなドキュメントに影響されているのか？
    # 影響されたか否かをxに，sentimentのスコアとANNOTATIONsのスコアをxにとってロジスティック回帰をしてみる？

    # Annotationsの分析
    #  Krippendorff's Alpha
    annotation_data = analysis_data[annotation_num_data + ["pair_id", "participant_id"]]
    all_ids = analysis_data.participant_id.unique()
    student_ids = analysis_data.query(q_no_financial_experience).participant_id.unique()
    expert_ids = analysis_data.query(q_financial_experience).participant_id.unique()
    veteran_ids = analysis_data.query(q_veteran_experience).participant_id.unique()
    print_agreement(all_ids, "All")
    print("\n")
    print_agreement(student_ids, "Student")
    print("\n")
    print_agreement(expert_ids, "Expert")
    print("\n")
    print_agreement(veteran_ids, "Veteran")

    # Model Changing document and Human Change
    analysis_data.query(q_rank_top)["difference"].mean()
    analysis_data.query(q_rank_bottom)["difference"].mean()

    # どのようなドキュメントに影響を受けるか？
    analysis_data["sum_sentiment"] = (
        analysis_data["positive_score"] + analysis_data["negative_score"]
    )
    annotation_analysis_diff = annotations_diff(analysis_data, diffs=True)
    annotations_analysis_non_diff = annotations_diff(analysis_data, diffs=False)
    pd.options.display.float_format = custom_format
    ipdb.set_trace()
    # Model Changing and Human Annotations
    use_cols_most_changing = (
        annotation_num_data
        + annotation_num_data_standarized
        + [
            "positive_score",
            "negative_score",
            "difference",
            "up_difference",
            "down_difference",
            "second_anonymized",
            "rank",
        ]
    )
    analysis_data[use_cols_most_changing].query(q_rank_bottom)[
        annotation_num_data_standarized
    ].mean()
    analysis_data[use_cols_most_changing].query(q_rank_bottom)[
        annotation_num_data_standarized
    ].mean()

    # Investorsにとって最もConvincingなDocumentを取ってくる
    # PCAをとってannotationを真ん中にする
    analysis_data[annotation_num_data] = analysis_data[
        annotation_num_data
    ].values.astype("int")
    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(analysis_data[annotation_num_data_standarized])
    analysis_data["pca_1"] = pca_result[:, 0]
    analysis_data["pca_2"] = pca_result[:, 1]
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)

    # # Investorsの行動との相関
    # output_investor_base = "./output_investor_correlation/"
    # analysis_data.groupby("second_anonymized")[
    #     annotation_num_data_standarized
    #     + annotation_num_data
    #     + [
    #         "raw_difference",
    #         "difference",
    #         "up_difference",
    #         "down_difference",
    #         "positive_score",
    #         "negative_score",
    #         "pca_1",
    #         "pca_2",
    #     ]
    # ].mean().corr().to_csv(output_investor_base + "overall_correlation.csv")

    # analysis_data.query(q_veteran_experience).groupby("second_anonymized")[
    #     annotation_num_data_standarized
    #     + annotation_num_data
    #     + [
    #         "raw_difference",
    #         "difference",
    #         "up_difference",
    #         "down_difference",
    #         "positive_score",
    #         "negative_score",
    #         "pca_1",
    #         "pca_2",
    #     ]
    # ].mean().corr().to_csv(output_investor_base + "veteran_correlation.csv")

    # analysis_data.query(q_financial_experience).groupby("second_anonymized")[
    #     annotation_num_data_standarized
    #     + annotation_num_data
    #     + [
    #         "raw_difference",
    #         "difference",
    #         "up_difference",
    #         "down_difference",
    #         "positive_score",
    #         "negative_score",
    #         "pca_1",
    #         "pca_2",
    #     ]
    # ].mean().corr().to_csv(output_investor_base + "expert_correlation.csv")

    # analysis_data.query(q_no_financial_experience).groupby("second_anonymized")[
    #     annotation_num_data_standarized
    #     + annotation_num_data
    #     + [
    #         "raw_difference",
    #         "difference",
    #         "up_difference",
    #         "down_difference",
    #         "positive_score",
    #         "negative_score",
    #         "pca_1",
    #         "pca_2",
    #     ]
    # ].mean().corr().to_csv(output_investor_base + "student_correlation.csv")

    # # Most Convincing Documentを取ってくる
    # # Second DocumentごとにConvincingの正解を持ってくる
    # most_convincing = get_top_n_most_convincing_data(analysis_data)
    # veteran_most_convinving = get_top_n_most_convincing_data(analysis_data.query(q_veteran_experience))
    # expert_most_convincing = get_top_n_most_convincing_data(analysis_data.query(q_financial_experience))
    # students_most_convincing = get_top_n_most_convincing_data(analysis_data.query(q_no_financial_experience))

    ###########################################################################################

    # Modelの分析との比較：Document RankがBottomとTopで差が出るのか？
    analysis_data.query(q_rank_bottom)["difference"].mean()
    analysis_data.query(q_rank_top)["difference"].mean()
    analysis_data.query(q_rank_none)["difference"].mean()
    ipdb.set_trace()
    # print("Done")
