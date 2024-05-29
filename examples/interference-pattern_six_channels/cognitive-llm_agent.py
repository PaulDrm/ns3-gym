import gym
from ns3gym import ns3env

import numpy as np
import pandas as pd 
import json
import argparse

import copy
from abc import ABC, abstractmethod
import re 
import sys 
sys.path.append('../../')
sys.path.append('../')
import utils.agents as agents
import torch
from LlamaGym.llamagym import Agent

from transformers import AutoTokenizer, AutoModelForCausalLM#, AutoModelForTokenClassification,  LlamaForCausalLM, AutoTokenizer #, OPTForCausalLM, AutoModelForCausalLM
#from transformers import StoppingCriteria, StoppingCriteriaList

from transformers import GenerationConfig
from trl import AutoModelForCausalLMWithValueHead

def generate_response(
    prompt,
    model = None,
    tokenizer = None,
    stopping_criteria = None,
    device = 'cuda',
    **kwargs,
    ):

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        **kwargs,
    )
    with torch.no_grad():
        
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            stopping_criteria=stopping_criteria,
            output_hidden_states= True,
            output_scores=True,
            #output_attentions=True,
        )
        s = generation_output.sequences[0]
    output = tokenizer.decode(s)

    #print(output)

    return output, generation_output

def load_model(model_name, load_in_8_bit=False, load_in_4_bit=False, value_head=True): 

    LOAD_8BIT = load_in_8_bit
    LOAD_4BIT = load_in_4_bit

    assert not(LOAD_4BIT and LOAD_8BIT), "Can't load 4bit and 8bit at the same"

    BASE_MODEL = model_name
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if value_head:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
                BASE_MODEL,
                load_in_8bit=LOAD_8BIT,
                load_in_4bit=LOAD_4BIT,
                torch_dtype=torch.float16,
                trust_remote_code = True,
            # device_map="auto",
            )
        
    else: 
        model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                load_in_8bit=LOAD_8BIT,
                load_in_4bit=LOAD_4BIT,
                torch_dtype=torch.float16,
                trust_remote_code = True,
            # device_map="auto",
            )

    #model.cuda()
    if "openchat" in model.config._name_or_path:
        if value_head:
            model.pretrained_model.generation_config.pad_token_id = 0
        else:
            model.generation_config.pad_token_id = 0

    elif "Qwen" in model.config._name_or_path:
        
        if value_head:
            model.pretrained_model.generation_config.pad_token_id = 151643

        else:
            model.generation_config.pad_token_id = 151643


    elif "Phi-3" in model.config._name_or_path:
        #model.pretrained_model.generation_config.eos_token_id = 32007
        if value_head:
            tokenizer.eos_token_id = 32007
        
        else:
            model.generation_config.pad_token_id = 32007

    elif "Meta-Llama-3-8B-Instruct" in model.config._name_or_path:
        #model.pretrained_model.generation_config.eos_token_id = 32007
        if value_head:
            model.pretrained_model.generation_config.pad_token_id = tokenizer.eos_token_id
            
        else:
            model.generation_config.pad_token_id = 32007
        
        tokenizer.eos_token_id = 128009
        tokenizer.chat_template = """
{% set loop_messages = messages %}
{% for message in loop_messages %}
{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>'+ message['content'] | trim + '<|eot_id|>' %}
{% if loop.index0 == 0 %} {% set content = bos_token + content %} {% endif %}
{{ content }}
{% endfor %}
        """

    elif "phi-2" in model.config._name_or_path:
        tokenizer.chat_template = tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = message['content']%}{% if loop.index0 == 0 %} {% set content = bos_token + content%}{% endif %}{{content+eos_token}}{%endfor %}"
    
    if not load_in_8_bit:
        model = model.cuda()
        
    return tokenizer, model 

def add_and_pop_if_needed(mylist, item, max_length=8):
    # Step 2: Add the new item
    mylist.append(item)
    
    # Step 3: Check the length and remove the first item if needed
    if len(mylist) > max_length:
        mylist.pop(0)  # This removes the first item

def add_to_history(observations, ground_truth): 
    
    ground_truth_str = "|".join([str(num) for num in ground_truth])
    addition = f"""Here is the continuation of the time step table:

| Time Step | Channel A | Channel B | Channel C | Channel D |
|-----------|-----------|-----------|-----------|-----------|
|{len(observations)+1}|{ground_truth_str}"""

    return addition

def run_episode(env, agent, max_env_steps=40, warm_up=20, max_history_length=20, random=False):
    if random == True:
        print(f"Running random run")
    else:
        print(f"Running episode for max history {max_history_length} and warm-up steps {warm_up}...")

    # Initialize the rewards and time histories
    time_history = []
    reward_history = []
    states_history = []
    actions_history = []

    total_steps = max_env_steps # warm_up

    # Setting environment's maximum steps
    env._max_episode_steps = total_steps
    
    # Get the action and observation space sizes
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    rewardsum = 0
    states = []

    #with pipes() as (out, err):
    state = env.reset()
    state = np.reshape(state, [1, s_size])
    states_history.append(state.squeeze().astype(int).tolist())


    # Initialize states being used by the model
    add_and_pop_if_needed(states, state.squeeze(), max_length=max_history_length)

    for time in range(total_steps):
        print("\n")
        print("Time: ", time)
        print("Observation: ", state)
        if random or time < warm_up:
            action = np.random.randint(a_size)
            print("Action: ", action) if time < warm_up else None
            zero_index = action
        else:
            try:
                #action = predict_next_action(states, model, tokenizer)

                #states = # Find min and max for each column
                # if noise: 
                #     mins = np.min(states, axis=0)
                #     maxs = np.max(states, axis=0)
                #     # Scale each column
                #     scaled_observations = (states - mins) / (maxs - mins) * 10
                #     scaled_observations = np.round(scaled_observations).astype(int).tolist()
                #     #states = scaled_observations
                #     #print(states)
                #     action = agent.act(scaled_observations)
                # else:
                action = agent.act(states)
                print()
                try:
                    assert len(action) == a_size, f"Expected {a_size} actions, got {len(action)} Action:{action}"
                    print("Agent decision: ", action)
                    print("LLM decision: ", action)
                    actions_history.append(action)
                    #zero_index = action.index(0)

                    zero_index = np.argmin(action)

                except AssertionError:

                    print("Returned actions smaller action size")
                    actions_history.append(action)
                    print("LLM decision: ", action)
                    zero_index = 0
                except TypeError: 
                    print("Something went wrong... Action is: ", action)
                    actions_history.append(action)
                    #print("LLM decision: ", action)
                    zero_index = 0

            except ValueError as e:
                print(e)
                print("Fallback to random action")

                zero_index = np.random.randint(a_size)
                action = zero_index
                actions_history.append(-100)
                
        print("Passing action: ", action)
        next_state, reward, done, _ = env.step(zero_index)



        if noise: 
            ### add this to history 
            """<|start_header_id|>assistant<|end_header_id|>Here is the continuation of the time step table:

| Time Step | Channel A | Channel B | Channel C | Channel D |
|-----------|-----------|-----------|-----------|-----------|
|10|18|50|22|25|<|eot_id|>"""

            agent.current_episode_messages = agent.current_episode_messages[:-1]

            ground_truth = add_to_history(states, next_state) +"|"# "<|eot_id|>"
            #print(ground_truth)
            agent.current_episode_messages.append({"role": "assistant", "content": ground_truth})

        #if action == next_state.squeeze().tolist():
        if np.argmax(action) == np.argmax(next_state.squeeze()):
            reward = +1

        else: 
            reward = -1
            
        print("next_state: ", next_state, " reward: ", reward)
        add_and_pop_if_needed(states, next_state.squeeze(), max_length=max_history_length)
            
        state = next_state
        states_history.append(state.squeeze().astype(int).tolist())

        if time >= warm_up:
            rewardsum += reward

        print("__________________")
        if done:
            print(f"Time: {time}, Reward: {rewardsum}")
            #break

        #epsilon = max(epsilon * epsilon_decay, epsilon_min)  # Update epsilon
        time_history.append(time)
        reward_history.append(rewardsum)

    return time_history, reward_history, actions_history, states_history

# Redirect the output of the ns3gym environment creation
#with pipes() as (out, err):
env = gym.make('ns3-v0')

# Your code that uses the ns3gym environment goes here
print("ns3gym environment created successfully!")

# 
#models = ["microsoft/phi-2", "microsoft/Phi-3-mini-4k-instruct","meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B"]

#results = pd.read_json("results.json").to_dict()
results = []

#model_name =  "Nexusflow/Starling-LM-7B-beta"#"Qwen/Qwen1.5-14B"#" "h2oai/h2o-danube2-1.8b-base"#
#
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='llama_7B', help='model name')
parser.add_argument("--output_path", type=str, default='overall_results.json', help='output path')
parser.add_argument("--noise",  type=lambda x: (str(x).lower() == 'true'), default='true')

parser.add_argument("--max_env_steps", type=int, default=40)

parser.add_argument("--random", type=lambda x: (str(x).lower() == 'true'), default='true')
args = parser.parse_args()

# Print all arguments
print("Parsed arguments:")
for arg, value in vars(args).items():
    print(f"{arg}: {value}")

model_name = args.model_name
#model_name = "microsoft/Phi-3-mini-4k-instruct" # "meta-llama/Llama-2-7b-chat-hf"#"openchat/openchat-3.5-0106"
#model_name = "openchat/openchat-3.5-0106"
#for model_name in models:
print(model_name)

noise = args.noise
MAX_NEW_TOKENS=50
generation_args = {"max_new_tokens":MAX_NEW_TOKENS,
        "do_sample": True, 
        "num_beams" : 1, 
        "num_return_sequences" : 1, 
        "temperature": 0.01,# 0.8, 
        "top_p": 0.95,
        #"min_new_tokens": 256, 
        #"no_repeat_ngram_size": 12, 
        #"begin_suppress_tokens": [2], 
        }

if noise: 
    MAX_NEW_TOKENS=600
    generation_args = {"max_new_tokens":MAX_NEW_TOKENS,
                "do_sample": True, 
                "num_beams" : 1, 
                "num_return_sequences" : 1, 
                "temperature": 0.01,# 0.8, 
                "top_p": 0.95,
                "min_new_tokens": 256, 
                #[128009]
                #"no_repeat_ngram_size": 12, 
                "begin_suppress_tokens": [128009], 
                }
# Example usage
# Define `model`, `tokenizer`, `device` variables appropriately.
# Instantiate the agent and use it in a simulation or training loop.

device = "cuda"
tokenizer, model = load_model(model_name, load_in_8_bit=True, value_head=False)

if noise: 
    agent = agents.llm_agent_outline_noise_2(model, tokenizer, device, generate_config_dict=generation_args)
else:
    agent = agents.llm_agent_outline(model, tokenizer, device, generate_config_dict=generation_args)

warm_ups = [0]#[0, 10, 20] #0,10,
max_lengths = [10 , 20]#, 20, 30] #10, 20, 40]#[0, 10, 20] #0,10, #,30,40


        #break
if args.random:
    for i in range(3):
            time_history, reward_history, action_history, states_history = run_episode(env, agent, warm_up=0, max_history_length=0, random=args.random)
            results.append({"warm_up": 0, "max_length": 0, "time_history": time_history, "reward_history": reward_history, "random": True})
else:
    random = False
    for warm_up in warm_ups:
        for max_length in max_lengths:
            time_history, reward_history, action_history, states_history = run_episode(env, agent, max_env_steps=20, warm_up=warm_up, max_history_length=max_length, random=random)
            temp = {"model": model_name, "warm_up": warm_up, "max_length": max_length, "time_history": time_history, "reward_history": reward_history, "action_history": action_history, "states_history": states_history, "random": random}
            results.append(temp)
df = pd.DataFrame(results)
#df.to_json(f"results_llama2.json", orient="records", indent=4)

df.to_json(f"results.json", orient="records", indent=4)

#print(states_history)
# Path to your JSON file
json_file_path = args.output_path#f'overall_results.json'
# Load existing data fr<om the JSON file
try:
    with open(json_file_path, 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    data = []

print(len(data))
# Add the new data to the existing list (or create a new list if the file was not found)
[data.append(res) for res in results]

# Write the updated data back to the JSON file
with open(json_file_path, 'w') as file:
    json.dump(data, file, indent=4)





