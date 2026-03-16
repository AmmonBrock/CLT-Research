from virtual_weights import VirtualWeightNeighbors
import numpy as np



weights_dir = "twera_small_sample_12M"
sample_indices_path = "feature_filtering/sampled_features_small.npy"
prefix = "twera_"
suffix = ".safetensors"
tensor_prefix = "TWERA_"
index_in_sampled = True

class LabelEvaluation:
    def __init__(self, weights_dir, sample_indices_path, prefix, suffix, tensor_prefix, max_layer = 25):
        self.twera_weight_network = VirtualWeightNeighbors(save_dir = weights_dir, sample_indices_path = sample_indices_path, prefix = prefix, suffix = suffix, tensor_prefix = tensor_prefix)
        self.n_sampled_features = self.twera_weight_network.sample_indices.shape[1]
        self.max_layer = max_layer

    def sample_features_to_evaluate(self, layer, num_features):
        """Returns a numpy array of random feature indices to evaluate for a given layer. Assumes sample_indices_path is provided."""
        assert (self.twera_weight_network.sample_indices is not None), "Sample indices not loaded. Please provide a valid sample_indices_path."
        sampled_indices = self.twera_weight_network.sample_indices[layer]
        selected_indices = np.random.choice(sampled_indices, size=num_features, replace=False)
        return selected_indices.tolist()
    
    def get_downstream_neighbor_quiz(self, layer, feature_idx, m = .05, n_questions = 30):
        if layer == self.max_layer:
            raise ValueError(f"Layer {self.max_layer} is the output layer and has no downstream neighbors.")
        
        assert m <= .5, "m should be less than or equal to .5 to ensure there is a difference in distribution between top and bottom neighbors."
    
        n_downstream_features = self.n_sampled_features * (self.max_layer - layer)
        k = int(n_downstream_features * m)
        assert k > n_questions, f"m is too small to generate {n_questions} questions. Increase m or decrease n_questions."


        k_downstream_top = self.twera_weight_network.get_k_downstream_neighbors(layer = layer, feature_idx = feature_idx, k = k, method = "top",max_layer = self.max_layer, index_in_sampled = True)
        k_downstream_bottom = self.twera_weight_network.get_k_downstream_neighbors(layer = layer, feature_idx = feature_idx, k = k, method = "abs_bottom", max_layer = self.max_layer, index_in_sampled = True)
        

        # I should sample more than n_questions so that I can drop the neighbors that have nan descriptions. int(1.5 *n_questions) should do the trick
        to_sample = int(1.5 * n_questions)
        if len(k_downstream_top) < to_sample or len(k_downstream_bottom) < to_sample:
            raise ValueError(f"Tried to sample {to_sample} neighbors, but only found {len(k_downstream_top)} top neighbors and {len(k_downstream_bottom)} bottom neighbors. Decrease n_questions or increase m.")
        k_downstream_top_sampled = np.array(k_downstream_top)[np.random.choice(len(k_downstream_top), size=to_sample, replace=False), :]
        k_downstream_bottom_sampled = np.array(k_downstream_bottom)[np.random.choice(len(k_downstream_bottom), size=to_sample, replace=False), :]

        # Get labels for sampled neighbors
        top_neighbors, source = self.twera_weight_network.get_labels_for_neighbors(layer = layer, feature_idx = feature_idx, neighbor_results = k_downstream_top_sampled, index_in_sampled = True, additional_label_info=['typeName'], show_source_feature = False, show_neighbors=False)
        bottom_neighbors, _ = self.twera_weight_network.get_labels_for_neighbors(layer = layer, feature_idx = feature_idx, neighbor_results = k_downstream_bottom_sampled, index_in_sampled = True, additional_label_info=['typeName'], show_source_feature = False, show_neighbors = False)
        top_neighbors.dropna(subset = ["description"], inplace = True)
        bottom_neighbors.dropna(subset = ["description"], inplace = True)

        # Create a priority score column that helps us know which labels to keep if there are duplicates
        priority_map = {"np_max-act": 1, "np_max-act-logits": 2}
        top_neighbors["priority"] = top_neighbors["typeName"].map(priority_map).fillna(3)
        bottom_neighbors["priority"] = bottom_neighbors["typeName"].map(priority_map).fillna(3)
        source["priority"] = source["typeName"].map(priority_map).fillna(3)
        top_neighbors_sorted = top_neighbors.sort_values(by=["original_feature_idx", "layer", "priority"])
        bottom_neighbors_sorted = bottom_neighbors.sort_values(by=["original_feature_idx", "layer", "priority"])
        source_sorted = source.sort_values(by=["original_feature_idx", "layer", "priority"])
        top_neighbors_deduped = top_neighbors_sorted.drop_duplicates(subset=["original_feature_idx", "layer"], keep="first")
        bottom_neighbors_deduped = bottom_neighbors_sorted.drop_duplicates(subset=["original_feature_idx", "layer"], keep="first")
        source_deduped = source_sorted.drop_duplicates(subset=["original_feature_idx", "layer"], keep="first")

        # Only keep n_questions
        quiz_len = min(len(top_neighbors_deduped), len(bottom_neighbors_deduped), n_questions)
        if quiz_len < n_questions:
            print("Warning: Did not find n_questions with valid descriptions")
        top_neighbors = top_neighbors_deduped.iloc[:quiz_len]
        bottom_neighbors = bottom_neighbors_deduped.iloc[:quiz_len]

        if len(source_deduped) == 0:
            raise ValueError("Source feature has no valid description. Cannot generate quiz.")
        source_description = source_deduped["description"].iloc[0] 
        top_descriptions = top_neighbors["description"].tolist()
        bottom_descriptions = bottom_neighbors["description"].tolist()

        return source_description, top_descriptions, bottom_descriptions
    
    def display_quiz(self, source_description, top_descriptions, bottom_descriptions):
        quiz_len = len(top_descriptions)
        assert quiz_len == len(bottom_descriptions), "Number of top and bottom descriptions should be the same."
        print(f"Source feature description: {source_description}\n")
        correct_answers = np.random.choice(['a', 'b'], size=quiz_len)

        options = {'a': "choose answer a",
                   'b': "choose answer b",
                   'x': "exit quiz",
                   "r": "go back one question"
                   }

        answer_choices = []

        i=0
        while i < quiz_len:
            print(f"Question {i+1}:")
            print("Source feature description:", source_description)
            if correct_answers[i] == 'a':
                print("a)", top_descriptions[i])
                print("b)", bottom_descriptions[i])
            else:
                print("a)", bottom_descriptions[i])
                print("b)", top_descriptions[i])

            choice = input("Which neighbor is more similar to the source feature? (a/b): ")
            while choice not in options.keys():
                choice = input("Invalid choice. Please enter 'a' or 'b': ")
            if choice == "x":
                print("Exiting quiz.")
                break
            elif choice == "r":
                if i > 0:
                    i -= 1
                    answer_choices.pop()
                else:
                    print("Already at the first question. Cannot go back further.")
            else:
                answer_choices.append(choice)
                i += 1
        
        quiz_complete = (len(answer_choices) == quiz_len)
        score = sum([1 for user_choice, correct_answer in zip(answer_choices, correct_answers) if user_choice == correct_answer])/quiz_len
        return quiz_complete, score
    
    def take_quiz_for_feature(self, layer, feature_idx, m =.05, n_questions = 30):
        source_description, top_descriptions, bottom_descriptions = self.get_downstream_neighbor_quiz(layer = layer, feature_idx = feature_idx, m = m, n_questions = n_questions)
        quiz_complete, score = self.display_quiz(source_description, top_descriptions, bottom_descriptions)
        print(f"Quiz complete: {quiz_complete}, Your score: {score:.2f}")
        return quiz_complete, score



        




if __name__ == "__main__":
    label_evaluator = LabelEvaluation(weights_dir = weights_dir, sample_indices_path = sample_indices_path, prefix = prefix, suffix = suffix, tensor_prefix = tensor_prefix)
    # Got 20/30 correct with m = .05, layer = 12, feature_idx = 4
    label_evaluator.take_quiz_for_feature(layer = 12, feature_idx = 10, m = .05, n_questions = 30)        

        

        # Sample n_questions from the top and bottom neighbors before getting the feature labels


        


