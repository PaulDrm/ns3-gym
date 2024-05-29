#!/bin/bash

# Define an array of model names
models=("meta-llama/Meta-Llama-3-8B-Instruct") #"microsoft/phi-2" "microsoft/Phi-3-mini-4k-instruct" "meta-llama/Llama-2-7b-chat-hf"  #"microsoft/phi-2" "microsoft/Phi-3-mini-4k-instruct" "meta-llama/Llama-2-7b-chat-hf" 
#models=("openchat/openchat-3.5-0106")
#output_path="./llama3_greedy_sampler_agent_noise_reasoning.json"
output_path="./greedy_sampler_agent_noise_pattern_full.json"

#output_path="./llama3_greedy_sampler_agent_random.json"
# Loop through the model names
for model in "${models[@]}"
do
  echo "Running script with model: $model"
  # Call your Python script with the current model name as an argument
  python cognitive-llm_agent.py --model_name "$model" --output_path $output_path --random "false" --noise "True"
done
