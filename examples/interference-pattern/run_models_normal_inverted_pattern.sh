#!/bin/bash

# Define an array of model names
models=("meta-llama/Meta-Llama-3-8B-Instruct") #"microsoft/phi-2" "microsoft/Phi-3-mini-4k-instruct" "meta-llama/Llama-2-7b-chat-hf" "meta-llama/Meta-Llama-3-8B-Instruct") #"microsoft/phi-2" "microsoft/Phi-3-mini-4k-instruct" "meta-llama/Llama-2-7b-chat-hf" 
output_path="./results/paper/greedy_sampler_agent_inverted_pattern_full.json"

# Loop through the model names
for model in "${models[@]}"
do
  echo "Running script with model: $model"
  # Call your Python script with the current model name as an argument
  python cognitive-llm_agent.py --model_name "$model" --output_path $output_path --random "false" --noise "false"
done
