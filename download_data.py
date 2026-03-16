from datasets import load_dataset

# Download the sample
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

# Save it to a local folder
# This creates a folder containing Arrow files and a json state file
dataset.save_to_disk("data/fineweb_10bt_offline")

print("Saved to data/fineweb_10bt_offline. Transfer this folder to your GPU node.")