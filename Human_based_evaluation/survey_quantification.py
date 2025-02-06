import pandas as pd
import ipdb
import numpy as np

ROUND_NUMS = 30


def participant_id(survey_data: pd.DataFrame) -> pd.DataFrame:
    return survey_data.filter(items=["participant.code"]).rename(
        columns={"participant.code": "participant_id"}
    )


def age(survey_data: pd.DataFrame) -> pd.DataFrame:
    return survey_data.filter(items=["Survey.1.player.Age"]).rename(
        columns={"Survey.1.player.Age": "age"}
    )


def gender(survey_data: pd.DataFrame) -> pd.DataFrame:
    # Male: 1, Female: 0
    return (
        survey_data.filter(items=["Survey.1.player.Gender"])
        .replace({"Male": 1, "Female": 0})
        .rename(columns={"Survey.1.player.Gender": "gender"})
    )


def student_flag(survey_data: pd.DataFrame) -> pd.DataFrame:
    # Student: 1, Not student: 0
    return survey_data.filter(items=["Survey.1.player.Student"]).rename(
        columns={"Survey.1.player.Student": "student_flag"}
    )


def financial_experimence_flag(survey_data: pd.DataFrame) -> pd.DataFrame:
    # Experienced: 1, Not experienced: 0
    return survey_data.filter(items=["Survey.1.player.Financial_industry"]).rename(
        columns={"Survey.1.player.Financial_industry": "financial_experience_flag"}
    )


def financial_working_years(survey_data: pd.DataFrame) -> pd.DataFrame:
    return survey_data.filter(items=["Survey.1.player.Financial_year"]).rename(
        columns={"Survey.1.player.Financial_year": "financial_working_years"}
    )


def objective_financial_literacy_calculate(survey_data: pd.DataFrame) -> pd.DataFrame:
    return (
        survey_data.filter(
            items=[
                "Survey.1.player.financial_literacy_1",
                "Survey.1.player.financial_literacy_2",
                "Survey.1.player.financial_literacy_3",
                "Survey.1.player.financial_literacy_4",
                "Survey.1.player.financial_literacy_5",
            ]
        )
        .replace(
            {
                "Survey.1.player.financial_literacy_1": {
                    " more than $102": 1,
                    " exactly $102": 0,
                    " less than $102": 0,
                },
                "Survey.1.player.financial_literacy_2": {
                    " more than today": 0,
                    " exactly the same": 0,
                    " less than today": 1,
                },
                "Survey.1.player.financial_literacy_3": {
                    " they will rise": 0,
                    " they will fall": 1,
                    " they will remain the same": 0,
                    "there is no relationship between bond prices and the interest rate": 0,
                },
                "Survey.1.player.financial_literacy_4": {" True": 1, " False": 0},
                "Survey.1.player.financial_literacy_5": {" True": 0, " False": 1},
            }
        )
        .sum(axis=1)
        .to_frame(name="objective_financial_literacy")
    )


def subjective_financial_literacy_calculate(survey_data: pd.DataFrame) -> pd.DataFrame:
    return survey_data.filter(items=["Survey.1.player.sub_financial_literacy"]).rename(
        columns={
            "Survey.1.player.sub_financial_literacy": "subjective_financial_literacy"
        }
    )


def risk_tolerance(survey_data: pd.DataFrame) -> pd.DataFrame:
    # Risk averse: 0, Risk neutral: 1, Risk taker 2, Risk loving: 3
    return (
        survey_data.filter(items=["Survey.1.player.risk_tolerance"])
        .replace(
            {
                "Survey.1.player.risk_tolerance": {
                    " Low returns without risk of losing principal": 0,
                    " Fair returns with high safety for my principal": 1,
                    " High returns with a fair degree of principal safety": 2,
                    " Very high returns, even with a high risk of losing part of my principal": 3,
                }
            }
        )
        .rename(columns={"Survey.1.player.risk_tolerance": "risk_tolerance"})
    )


def cognitive_test(survey_data: pd.DataFrame) -> pd.DataFrame:
    survey_data["Survey.1.player.cog_test_1"] = np.where(
        survey_data["Survey.1.player.cog_test_1"] == 0.05, 1, 0
    )
    survey_data["Survey.1.player.cog_test_2"] = np.where(
        survey_data["Survey.1.player.cog_test_2"] == 5, 1, 0
    )
    survey_data["Survey.1.player.cog_test_3"] = np.where(
        survey_data["Survey.1.player.cog_test_3"] == 47, 1, 0
    )
    return (
        survey_data.filter(
            items=[
                "Survey.1.player.cog_test_1",
                "Survey.1.player.cog_test_2",
                "Survey.1.player.cog_test_3",
            ]
        )
        .sum(axis=1)
        .to_frame(name="cognitive_test")
    )


def big_five_calculate(survey_data: pd.DataFrame) -> pd.DataFrame:
    personality_data = survey_data.filter(regex="personality")
    extraversion = (
        personality_data["Survey.1.player.personality_1"]
        + (8 - personality_data["Survey.1.player.personality_6"])
    ) / 2
    agreeableness = (
        (8 - personality_data["Survey.1.player.personality_2"])
        + personality_data["Survey.1.player.personality_7"]
    ) / 2
    conscientiousness = (
        personality_data["Survey.1.player.personality_3"]
        + (8 - personality_data["Survey.1.player.personality_8"])
    ) / 2
    neuroticism = (
        (8 - personality_data["Survey.1.player.personality_4"])
        + personality_data["Survey.1.player.personality_9"]
    ) / 2
    openness = (
        personality_data["Survey.1.player.personality_5"]
        + (8 - personality_data["Survey.1.player.personality_10"])
    ) / 2
    big_five_personality = pd.DataFrame(
        {
            "extraversion": extraversion,
            "agreeableness": agreeableness,
            "conscientiousness": conscientiousness,
            "neuroticism": neuroticism,
            "openness": openness,
        }
    )
    return big_five_personality


def quantify_survey(data):
    # Surveyの定量化に関して
    participant_id_data = participant_id(data)
    age_data = age(data)
    gender_data = gender(data)
    student_flag_data = student_flag(data)
    financial_experimence_flag_data = financial_experimence_flag(data)
    financial_working_years_data = financial_working_years(data)
    objective_financial_literacy_data = objective_financial_literacy_calculate(data)
    subjective_financial_literacy_data = subjective_financial_literacy_calculate(data)
    risk_tolerance_data = risk_tolerance(data)
    cognitive_test_data = cognitive_test(data)
    big_five_data = big_five_calculate(data)

    return pd.concat(
        [
            participant_id_data,
            age_data,
            gender_data,
            student_flag_data,
            financial_experimence_flag_data,
            financial_working_years_data,
            objective_financial_literacy_data,
            subjective_financial_literacy_data,
            risk_tolerance_data,
            cognitive_test_data,
            big_five_data,
        ],
        axis=1,
    )


def get_quantify_data(data_path):
    data = pd.read_csv(data_path)
    survey_data = quantify_survey(data)
    return survey_data


if __name__ == "__main__":

    survey_data = quantify_survey(data)

    ipdb.set_trace()
    print("Done")
