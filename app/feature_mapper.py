import pandas as pd

# Mapping from simplified API field names to original column names
# Based on CSV column structure
FEATURE_MAPPING = {
    # Demographics
    "age": "What is your age? - Age (years)",
    "gender": "What is your gender? - Selected Choice",
    "study_area": 'What is your area of study? If you are not a student, please select "not a student". - Selected Choice',
    "study_years": "How many years of tertiary study have you had?",
    
    # Knowledge questions
    "recognize_psychosis": "What do you think John is experiencing? - Selected ChoiceWhat do you think Jenny is experiencing? - Selected Choice",
    "professional_help_helpful": "Do you think a psychiatrist/psychologist will be helpful?",
    
    # Stigma questions (columns 7-13 in CSV)
    "john_could_snap_out": "The next few questions contain statements about John's problem.  Please indicate how strongly YOU PERSONALLY 2 or 4 with each statement. - John could snap out of it if he wantedThe next few questions contain statements about Jenny's problem.  P",
    "john_weakness": "The next few questions contain statements about John's problem.  Please indicate how strongly YOU PERSONALLY 2 or 4 with each statement. - John's problem is a sign of personal weakness.The next few questions contain statements about Jenny's pro",
    "john_not_real_illness": "The next few questions contain statements about John's problem.  Please indicate how strongly YOU PERSONALLY 2 or 4 with each statement. - John's problem is not a real medical illness.The next few questions contain statements about Jenny's prob",
    "john_dangerous": "The next few questions contain statements about John's problem.  Please indicate how strongly YOU PERSONALLY 2 or 4 with each statement. - John is dangerous. By 'dangerous' we mean 'dangerous to others' The next few questions contain statements",
    "avoid_john": "The next few questions contain statements about John's problem.  Please indicate how strongly YOU PERSONALLY 2 or 4 with each statement. - It is best to avoid John so that you don't develop this problem yourself.The next few questions contain s",
    "john_unpredictable": "The next few questions contain statements about John's problem.  Please indicate how strongly YOU PERSONALLY 2 or 4 with each statement. - John's problem makes him unpredictable.The next few questions contain statements about Jenny's problem.",
    "not_tell_anyone": "The next few questions contain statements about John's problem.  Please indicate how strongly YOU PERSONALLY 2 or 4 with each statement. - You would not tell anyone if you had a problem like John's.The next few questions contain statements abou",
    
    # Social interaction questions (columns 14-18)
    "go_out_weekend": "The following questions ask how you would feel about spending time with John. Please indicate how likely you would perform each statement. - To go out with John on the weekend?The following questions ask how you would feel about spending time with Jenny.",
    "work_on_project": "The following questions ask how you would feel about spending time with John. Please indicate how likely you would perform each statement. - To work on a project with John?The following questions ask how you would feel about spending time with Jenny. Plea",
    "invite_to_house": "The following questions ask how you would feel about spending time with John. Please indicate how likely you would perform each statement. - To invite John around to your house?The following questions ask how you would feel about spending time with Jenny.",
    "go_to_johns_house": "The following questions ask how you would feel about spending time with John. Please indicate how likely you would perform each statement. - To go to John's house?The following questions ask how you would feel about spending time with Jenny. Please indica",
    "develop_friendship": "The following questions ask how you would feel about spending time with John. Please indicate how likely you would perform each statement. - To develop a close friendship with John?The following questions ask how you would feel about spending time with Je",
    
    # First aid intention questions (columns 19-30)
    "ask_harm_thoughts": "How 4 is it that you would take the following actions with John? - Ask if they have been having thoughts of harming themselves or others.How 4 is it that you would take the following actions with Jenny? - Ask if they have been having thoughts of",
    "listen_restate": "How 4 is it that you would take the following actions with John? - Let them know you are listening to what they are saying by restating and summarising what they      have said.How 4 is it that you would take the following actions with Jenny? -",
    "convey_hope": "How 4 is it that you would take the following actions with John? - Convey a message of hope by telling them help is available and things can get better.How 4 is it that you would take the following actions with Jenny? - Convey a message of hope",
    "discuss_professional_options": "How 4 is it that you would take the following actions with John? - Discuss their options for seeking professional helpHow 4 is it that you would take the following actions with Jenny? - Discuss their options for seeking professional help",
    "ask_supportive_people": "How 4 is it that you would take the following actions with John? - Ask whether they have other supportive people they can rely onHow 4 is it that you would take the following actions with Jenny? - Ask whether they have other supportive people th",
    "ask_suicide_thoughts": "How 4 is it that you would take the following actions with John? - Ask if they have been thinking about suicide.How 4 is it that you would take the following actions with Jenny? - Ask if they have been thinking about suicide.",
    "ask_suicide_plan": "How 4 is it that you would take the following actions with John? - Ask if they have a plan for suicide – for example, how, when and where they intend to die.How 4 is it that you would take the following actions with Jenny? - Ask if they have a p",
    "encourage_professional_help": "How 4 is it that you would take the following actions with John? - Encourage them to get appropriate professional help as soon as possible – for example, see a mental health professional or someone at a mental health service.How 4 is it that you",
    "acknowledge_frightened": "How 4 is it that you would take the following actions with John? - Acknowledge they might be frightened by what they are experiencing.How 4 is it that you would take the following actions with Jenny? - Acknowledge they might be frightened by wha",
    "convince_false_beliefs": "How 4 is it that you would take the following actions with John? - Try to convince them that their beliefs and perceptions are false.How 4 is it that you would take the following actions with Jenny? - Try to convince them that their beliefs and",
    "listen_unreal_experiences": "How 4 is it that you would take the following actions with John? - Listen to them talk about their experiences even though you know they are not based in reality.How 4 is it that you would take the following actions with Jenny? - Listen to them",
    "find_reasons_no_help": "How 4 is it that you would take the following actions with John? - Find out if there are specific reasons why they do not want to seek professional help.How 4 is it that you would take the following actions with Jenny? - Find out if there are sp"
}

def map_api_to_model(api_input: dict) -> dict:
    """
    Convert API input (simplified field names) to model input (original column names)
    """
    model_input = {}
    
    for api_field, value in api_input.items():
        if api_field in FEATURE_MAPPING:
            model_field = FEATURE_MAPPING[api_field]
            model_input[model_field] = value
        else:
            # Keep the field as-is if no mapping exists
            model_input[api_field] = value
    
    return model_input

def map_model_to_api(model_output: dict) -> dict:
    """
    Convert model output (original column names) to API output (simplified field names)
    """
    api_output = {}
    
    # Reverse mapping
    reverse_mapping = {v: k for k, v in FEATURE_MAPPING.items()}
    
    for model_field, value in model_output.items():
        if model_field in reverse_mapping:
            api_field = reverse_mapping[model_field]
            api_output[api_field] = value
        else:
            api_output[model_field] = value
    
    return api_output

def get_expected_feature_order():
    """
    Get the feature order that the model expects
    This should match the order used during training
    """
    # Load the actual feature order from training
    try:
        with open("models/saved_models/feature_names.txt", 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        return feature_names
    except:
        # Fallback to the order in CSV (excluding the target column)
        return list(FEATURE_MAPPING.values())