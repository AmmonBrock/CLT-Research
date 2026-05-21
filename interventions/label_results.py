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
            df.to_csv("interventions/state_hop_evals.csv", index=False)
            return
        elif eval == "q":
            print("Exiting without saving...")
            return
        i += 1

    df["human_eval"] = human_evals
    df.to_csv("interventions/state_hop_evals.csv", index=False)
    return