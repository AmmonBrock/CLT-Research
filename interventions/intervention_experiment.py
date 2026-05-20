import torch
from configs.config_data import NetworkConfig
from pathlib import Path
import argparse
import os
import glob
import pandas as pd


states = [
    "Alabama", "Alaska", "Arizona", "California", "Colorado", 
    "Connecticut", "Florida", "Georgia", "Illinois", 
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", 
    "Michigan", "Minnesota", "Mississippi", "Montana", 
    "Nebraska", "Nevada", "New Jersey", "New Mexico", "New York", 
    "North Carolina", "North Dakota", "Ohio", "Oregon", "Pennsylvania", 
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", 
    "Washington", "Wisconsin", "Wyoming"
]
state_capitals = [
    "Montgomery", "Juneau", "Phoenix", "Sacramento", "Denver", 
    "Hartford", "Tallahassee", "Atlanta", "Springfield", 
    "Topeka", "Frankfort", "Baton Rouge", "Augusta", "Annapolis", 
    "Lansing", "St. Paul", "Jackson", "Helena", 
    "Lincoln", "Carson City", "Trenton", "Santa Fe", "Albany", 
    "Raleigh", "Bismarck", "Columbus", "Salem", "Harrisburg", 
    "Pierre", "Nashville", "Austin", "Salt Lake City", "Montpelier", 
    "Olympia", "Madison", "Cheyenne"
]
state_cities = [
    "Birmingham", "Anchorage", "Tucson", "Fresno", "Boulder", 
    "Bridgeport", "Miami", "Savannah", "Chicago", 
    "Wichita", "Louisville", "Shreveport", "Bangor", "Baltimore", 
    "Detroit", "Minneapolis", "Biloxi", "Bozeman", 
    "Omaha", "Reno", "Princeton", "Albuquerque", "Buffalo", 
    "Charlotte", "Fargo", "Cleveland", "Eugene", "Philadelphia", 
    "Deadwood", "Memphis", "Dallas", "Provo", "Burlington", 
    "Seattle", "Milwaukee", "Laramie"
]
initial_df = pd.DataFrame({"state": states, "capital": state_capitals, "clue": state_cities})

def get_human_response():
    full_options={"n": "No", "y": "Yes","b": "Go Back", "o": "Display More Options", "x": "Quit and Save", "q": "Quit without Saving"}
    default_options = {"n": "No", "y": "Yes", "o": "Display More Options"}
    def display_options(options):
        for k, v in options.items():
            print(f"{k}: {v}")
    
    print("Is the model's response correct?")
    display_options(default_options)
    while True:
        response = input("Enter your choice: ").strip().lower()
        if response in "nybxq":
            return response
        elif response == "o":
            display_options(full_options)
        else:
            print("Invalid choice. Please try again.")
            display_options(default_options)
    return response 


def filter_by_correct(config, df):
    """"""
    device = config.device
    cache_dir = config.model_storage_absolute
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snapshot_paths = glob.glob(f"{cache_dir}/models--{config.study_model_name.replace('/', '--')}/snapshots/*/")
    assert len(snapshot_paths) > 0, f"No snapshots found for model {config.study_model_name} in cache directory {cache_dir}. Please make sure the model is downloaded and the path is correct."
    absolute_model_path = snapshot_paths[0]
    tokenizer = AutoTokenizer.from_pretrained(absolute_model_path, local_files_only = True)
    print(config.study_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = config.study_model_name,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    ).to(device)

    # Split into batches
    prompts = [f"The capital of the state containing {city} is" for city in df["clue"]]
    batch_size = 16
    answers = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        
        # Tokenize the batch with padding and truncation
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        
        # Generate exactly up to 7 new tokens
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=7,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode only the newly generated tokens for each prompt in the batch
        completions = []
        for j, prompt_text in enumerate(batch_prompts):
            input_len = inputs.input_ids[j].shape[0]
            # output_ids[j] includes the padding tokens, so we slice from the end of the input sequence
            new_tokens = output_ids[j][input_len:]
            completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            completions.append(completion.strip())
        

        answers.extend(completions)

    df["model_answer"] = answers
    df.to_csv(config.CLT_dir / "interventions" / "results.csv", index=False)
    return

def human_eval(config, df):
    answers = df.model_answer.tolist()
    
    human_evals = [None] * len(answers)
    assert(len(answers) == len(df)), "Number of answers does not match number of rows in dataframe"
    i = 0
    while i < len(answers):
        answer = answers[i]
        state_capital = df.iloc[i]["capital"]
        state = df.iloc[i]["state"]
        print(f"Response: {answer} | Truth: {state_capital} | State: {state}")
        eval = get_human_response()
        if eval == "y":
            human_evals[i] = True
        elif eval == "n":
            human_evals[i] = False
        elif eval == "b":
            if i > 0:
                i -= 1
                continue
            else:
                print("Already at the first response, cannot go back.")
        elif eval == "x":
            print("Saving progress and exiting...")
            df["human_eval"] = human_evals
            df.to_csv(config.CLT_dir / "interventions" / "state_hop_evals.csv", index=False)
            return
        elif eval == "q":
            print("Exiting without saving...")
            return
        i += 1

    df["human_eval"] = human_evals
    df.to_csv(config.CLT_dir / "interventions" / "state_hop_evals.csv", index=False)
    return

def state_hop(config):


    evaluated_df = pd.read_csv(config.CLT_dir / "interventions" / "state_hop_evals.csv")

    device = config.device
    cache_dir = config.model_storage_absolute
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    from circuit_tracer import ReplacementModel
    from transformers import AutoTokenizer

    snapshot_paths = glob.glob(f"{cache_dir}/models--{config.study_model_name.replace('/', '--')}/snapshots/*/")
    assert len(snapshot_paths) > 0, f"No snapshots found for model {config.study_model_name} in cache directory {cache_dir}. Please make sure the model is downloaded and the path is correct."
    absolute_model_path = snapshot_paths[0]
    tokenizer = AutoTokenizer.from_pretrained(absolute_model_path, local_files_only = True)
    model = ReplacementModel.from_pretrained(
        model_name = config.study_model_name,
        transcoder_set = config.feature_tool_name,
        dtype = torch.bfloat16,
        device = device,
        local_files_only = True,
        tokenizer = tokenizer)
    

    clt = model.transcoders
    hook_name_base = clt.feature_input_hook
    hook_names = [f"blocks.{layer}.{hook_name_base}" for layer in range(clt.n_layers)]


    
    # verified_states = []
    # verified_cities = []
    # verified_capitals = []
    # for city, correct_capital, state in zip(state_cities, state_capitals, states):
    #     prompt = f"The capital of the state containing {city} is"
    #     answer = model.generate(prompt)
    #     if answer == correct_capital:
    #         verified_states.append(state)
    #         verified_cities.append(city)
    #         verified_capitals.append(correct_capital)

    


def main():
    parser = argparse.ArgumentParser(description="Compute coactivation stats for a specific layer.")
    parser.add_argument("--config", type=str, required = True, help="Name of config yaml file")
    args = parser.parse_args()
    clt_dir = Path(__file__).resolve().parent.parent
    config_path = clt_dir / "configs" / args.config
    config = NetworkConfig.from_yaml(config_path)
    config.validate_params()

    filter_by_correct(config, initial_df)



    


if __name__ == "__main__":
    main()
