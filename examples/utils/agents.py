from transformers import AutoTokenizer#, AutoModelForTokenClassification, AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer #, OPTForCausalLM, AutoModelForCausalLM
#from transformers import StoppingCriteria, StoppingCriteriaList
import gym
from transformers import GenerationConfig
from trl import AutoModelForCausalLMWithValueHead
from LlamaGym.llamagym import Agent
import torch
class LLM_agent_1(Agent):
    def __init__(self, model, tokenizer, device, generate_config_dict=None, ppo_config_dict=None):
        super().__init__(model, tokenizer, device, generate_config_dict, ppo_config_dict)
        self.stopping_criteria = None 
    def get_system_prompt(self) -> str:
        # Starting system prompt or any initialization text if needed
        return "Initialize system state."

    def format_observation(self, observations) -> str:
        interference_map = "\nTime Step Channel A Channel B Channel C Channel D\n"
        for i, state in enumerate(observations):
            interference_map += f"{i+1}\t{state[0]}\t{state[1]}\t{state[2]}\t{state[3]}\n"
        #interference_map += "\n"
        return interference_map

    def extract_action(self, response: str) -> gym.core.ActType:
        # Example pattern match and action extraction logic
        pattern = r'\{"next_row": "(.*?)"\}'
        match = re.search(pattern, response)
        if match:
            number_sequence = match.group(1)
            print("Prediction : ", number_sequence)
            number_array = number_sequence.split()
            number_array_int = [int(num) for num in number_array]
            
            #action = number_array_int.index(0)  # finding the first zero as an example action
            return number_array_int #action
        raise ValueError("Failed to extract action from response.")
    
    def parse_output(self, prompt, output):

        output =output[len(self.tokenizer.decode(self.tokenizer.encode(prompt))):]
        return output


    def llm(self,
        prompt,
        ):

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        generation_config = GenerationConfig(
             **self.generate_config_dict,
         )    
        with torch.no_grad():
            
            generation_output = self.model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                generation_config=generation_config,
                #stopping_criteria=self.stopping_criteria,
                #output_hidden_states= True,
                #output_scores=True,
                #output_attentions=True,
                ## Bug generation configuratio args need to be added as keyword arguments
                #**self.generate_config_dict
            )
            s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)

        #print(output)

        return output
    
    def create_prompt(self, message):#, user_message="GPT4 Correct User: ", system_message= "GPT4 Correct Assistant:", end_message= "<|end_of_turn|>"): 
        
        pattern_1 ="""
        Time Step Channel A Channel B Channel C Channel D
        1| 1 1 0 0
        2| 1 1 0 0
        3| 0 0 1 1
        4| 0 0 1 1
        5| 1 1 0 0"""
        answer_1 = """{"next_row": "1 1 0 0"}"""
        self.current_episode_messages = [{"role": "user", "content": pattern_1},
                {"role": "assistant", "content": answer_1}]
        
        self.current_episode_messages.append({"role": "user", "content": message})
        self.current_episode_messages.append({"role": "assistant", "content": '{"next_row": "'})
                    
        prompt = self.tokenizer.apply_chat_template(
                    self.current_episode_messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False)

        #print(prompt)
        prompt = prompt.strip()
        prompt = prompt[:-len(self.tokenizer.eos_token)]##prompt.replace(self.tokenizer.eos_token, "")
        #print(prompt)
        prompt = prompt[len(self.tokenizer.bos_token):]
        #print(prompt)
        return prompt

    def act(self, observations):
        message = self.format_observation(observations)
        #self.current_episode_messages += [{"role": "user", "content": message}]
        prompt = self.create_prompt(message)
        print(prompt)
        response = self.llm(prompt)
        response = self.parse_output(prompt, response)
        response = '{"next_row": "'+ response
        print(response)
        try:
            action = self.extract_action(response)
        except Exception as e:
            print(e)
            raise

        #self.current_episode_messages += [{"role": "assistant", "content": response}]
        return action

import outlines
import re
class llm_agent_outline(Agent):
    def __init__(self, model, tokenizer, device, generate_config_dict=None, ppo_config_dict=None):
        super().__init__(model, tokenizer, device, generate_config_dict, ppo_config_dict)
        self.stopping_criteria = None 
    def get_system_prompt(self) -> str:
        # Starting system prompt or any initialization text if needed
        return "Initialize system state."

    def format_observation(self, observations) -> str:
        header = "| Time Step | " + " | ".join([f"Channel {chr(65 + i)}" for i in range(len(observations[0]))]) + " |\n"
        separator = "|-----------" + "|-----------" * len(observations[0]) + "|\n"
        header = header + separator 
        interference_map = ""

        #print(interference_map)
        # Loop through the states and build the formatted string
        for i, state in enumerate(observations):
            row = f"|{i+1}|" + "|".join(map(str, state)) + "|\n"
            interference_map += row
        interference_map = header + interference_map
        return interference_map, header
    
    def create_prompt(self, message, header):#, user_message="GPT4 Correct User: ", system_message= "GPT4 Correct Assistant:", end_message= "<|end_of_turn|>"): 
        
        # pattern_1 ="""
        # Time Step Channel A Channel B Channel C Channel D
        # 1| 1 1 0 0
        # 2| 1 1 0 0
        # 3| 0 0 1 1
        # 4| 0 0 1 1
        # 5| 1 1 0 0"""
        # answer_1 = """{"next_row": "1 1 0 0"}"""
        # self.current_episode_messages = [{"role": "user", "content": pattern_1},
        #         {"role": "assistant", "content": answer_1}]
        
        #self.current_episode_messages.append({"role": "assistant", "content": '{"next_row": "'})

        self.current_episode_messages = [{"role": "user", "content": message}]

        pattern = r"^\|\s*(\d+)\s*\|"
        matches = re.findall(pattern, message, re.MULTILINE)
        last_time_step = int(matches[-1])
        started_assistant_message = f"""Here is the continuation of the time step table:

{header}
|{last_time_step+1}|"""
        
        self.current_episode_messages.append({"role": "assistant", "content": started_assistant_message})
        

        prompt = self.tokenizer.apply_chat_template(
                    self.current_episode_messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False)

        prompt = prompt.strip()
        prompt = prompt[:-len(self.tokenizer.eos_token)]##prompt.replace(self.tokenizer.eos_token, "")

        prompt = prompt[len(self.tokenizer.bos_token):]
        #print(prompt)
        return prompt

    def extract_action(self, response: str) -> gym.core.ActType:
        # Example pattern match and action extraction logic
        #pattern = r'\{"next_row": "(.*?)"\}'
        #match = re.search(pattern, response)
        #if match:
        number_sequence = response
        print("Prediction : ", number_sequence)
        number_array = number_sequence.split("|")
        number_array_int = [int(num.replace(" ", "")) for num in number_array]
        #action = number_array_int.index(0)  # finding the first zero as an example action
        action = number_array_int
        return action
        #raise ValueError("Failed to extract action from response.")
    
    def parse_output(self, prompt, output):

        output =output[len(self.tokenizer.decode(self.tokenizer.encode(prompt))):]
        return output

    def llm(self,
        prompt,
        action_space_len,
        ):

        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # input_ids = inputs["input_ids"].to(self.device)
        # generation_config = GenerationConfig(
        #      **self.generate_config_dict,
        #  )    
        # with torch.no_grad():
            
        #     generation_output = self.model.generate(
        #         input_ids=input_ids,
        #         return_dict_in_generate=True,
        #         generation_config=generation_config,
        #         #stopping_criteria=self.stopping_criteria,
        #         #output_hidden_states= True,
        #         #output_scores=True,
        #         #output_attentions=True,
        #         ## Bug generation configuratio args need to be added as keyword arguments
        #         #**self.generate_config_dict
        #     )
        #     s = generation_output.sequences[0]
        # output = self.tokenizer.decode(s)

        #print(output)

        model_out = outlines.models.Transformers(self.model, self.tokenizer)
        sampler = outlines.samplers.GreedySampler()
        # Function to generate the regex pattern
        def generate_pattern(length):
            pattern = ""
            for i in range(length):
                pattern += "[0-1]"
                if i < length - 1:
                    pattern += "\\|"
            return f"({pattern})"

        # Generate the pattern for a specific length
        length = action_space_len
        dynamic_pattern = generate_pattern(length)
        generator = outlines.generate.regex(
            model_out,
            regex_str = dynamic_pattern,
            #r"([0-1]\|[0-1]\|[0-1]\|[0-1])",
            sampler = sampler)

        output = generator(prompt, max_tokens=30)
        return output

    def act(self, observations):
        message, header = self.format_observation(observations)
        #self.current_episode_messages += [{"role": "user", "content": message}]
        prompt = self.create_prompt(message,header)
        print(prompt)
        response = self.llm(prompt, len(observations[0]))
        #response = self.parse_output(prompt, response)
        #response = '{"next_row": "'+ response
        print(response)
        try:
            action = self.extract_action(response)
        except Exception as e:
            print("Exception catched ", e)
            return None

        #self.current_episode_messages += [{"role": "assistant", "content": response}]
        return action
    
import numpy as np


# class llm_agent_outline_noise(Agent):
#     def __init__(self, model, tokenizer, device, generate_config_dict=None, ppo_config_dict=None):
#         super().__init__(model, tokenizer, device, generate_config_dict, ppo_config_dict)
#         self.stopping_criteria = None 
#         self.current_episode_messages = []
#         self.max_messages = 4

#     def add_message(self, message):
#         self.current_episode_messages.append(message)
#         # Ensure the length of current_episode_messages does not exceed max_messages
#         while len(self.current_episode_messages) > self.max_messages:
#             self.current_episode_messages.pop(0)  # Remove the oldest message until the length constraint is met

#     def get_system_prompt(self) -> str:
#         # Starting system prompt or any initialization text if needed
#         return "Initialize system state."

#     def format_observation(self, observations) -> str:
#         interference_map = """| Time Step | Channel A | Channel B | Channel C | Channel D |
# |-----------|-----------|-----------|-----------|-----------|\n"""

#         for i, state in enumerate(observations):
#             interference_map += f"|{i+1}|{state[0]}|{state[1]}|{state[2]}|{state[3]}|\n"
#         #interference_map += "\n"
#         return interference_map
    
#     def create_prompt(self, message):

#         self.current_episode_messages.append({"role": "user", "content": message})

#         pattern = r"^\|\s*(\d+)\s*\|"
#         matches = re.findall(pattern, message, re.MULTILINE)
#         last_time_step = int(matches[-1])
#         started_assistant_message = f"""Here is the continuation of the time step table:

# | Time Step | Channel A | Channel B | Channel C | Channel D |
# |-----------|-----------|-----------|-----------|-----------|
# |{last_time_step+1}|"""
        
#         #self.current_episode_messages.append({"role": "assistant", "content": started_assistant_message})
        
#         self.add_message({"role": "assistant", "content": started_assistant_message})

#         prompt = self.tokenizer.apply_chat_template(
#                     self.current_episode_messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False)

#         #print(prompt)
#         prompt = prompt.strip()
#         prompt = prompt[:-len(self.tokenizer.eos_token)]##prompt.replace(self.tokenizer.eos_token, "")
#         #print(prompt)
#         prompt = prompt[len(self.tokenizer.bos_token):]
#         #print(prompt)
#         return prompt

#     def extract_action(self, response: str) -> gym.core.ActType:
        
#         number_sequence = response
#         print("Prediction : ", number_sequence)
#         number_array = number_sequence.split("|")
#         number_array_int = [int(num) for num in number_array]
#         #action = number_array_int.index(0)  # finding the first zero as an example action
#         action = number_array_int
#         return action
#         #raise ValueError("Failed to extract action from response.")
    
#     def parse_output(self, prompt, output):

#         output =output[len(self.tokenizer.decode(self.tokenizer.encode(prompt))):]
#         return output

#     def llm(self,
#         prompt,
#         ):

#         # inputs = self.tokenizer(prompt, return_tensors="pt")
#         # input_ids = inputs["input_ids"].to(self.device)
#         # generation_config = GenerationConfig(
#         #      **self.generate_config_dict,
#         #  )    
#         # with torch.no_grad():
            
#         #     generation_output = self.model.generate(
#         #         input_ids=input_ids,
#         #         return_dict_in_generate=True,
#         #         generation_config=generation_config,
#         #         #stopping_criteria=self.stopping_criteria,
#         #         #output_hidden_states= True,
#         #         #output_scores=True,
#         #         #output_attentions=True,
#         #         ## Bug generation configuratio args need to be added as keyword arguments
#         #         #**self.generate_config_dict
#         #     )
#         #     s = generation_output.sequences[0]
#         # output = self.tokenizer.decode(s)

#         #print(output)

#         model_out = outlines.models.Transformers(self.model, self.tokenizer)


#         #[1-9]|[1-9][0-9]\| #
#         #    r"([0-1]\|[0-1]\|[0-1]\|[0-1])")#,
#         generator = outlines.generate.regex(
#             model_out,
#             r"([0-9]|[1-9][0-9])(\|([0-9]|[1-9][0-9]))(\|([0-9]|[1-9][0-9]))(\|([0-9]|[1-9][0-9]))")
#         output = generator(prompt, max_tokens=30)
#         return output

#     def act(self, observations):
#         message = self.format_observation(observations)
#         #self.current_episode_messages += [{"role": "user", "content": message}]
#         prompt = self.create_prompt(message)
#         print(prompt)
#         response = self.llm(prompt)
#         #response = self.parse_output(prompt, response)
#         #response = '{"next_row": "'+ response
#         print(response)
#         try:
#             action = self.extract_action(response)
#         except Exception as e:
#             print("Exception catched ", e)
#             return None

#         #self.current_episode_messages += [{"role": "assistant", "content": response}]
#         return action

class llm_agent_outline_noise_2(Agent):
    def __init__(self, model, tokenizer, device, generate_config_dict=None, ppo_config_dict=None):
        super().__init__(model, tokenizer, device, generate_config_dict, ppo_config_dict)
        self.stopping_criteria = None 
        self.current_episode_messages = []
        self.max_messages = 2

    def add_message(self, message):
        self.current_episode_messages.append(message)
        # Ensure the length of current_episode_messages does not exceed max_messages
        while len(self.current_episode_messages) > self.max_messages:
            self.current_episode_messages.pop(0)  # Remove the oldest message until the length constraint is met

    def get_system_prompt(self) -> str:
        # Starting system prompt or any initialization text if needed
        return "Initialize system state."

    def format_observation(self, observations) -> str:
        interference_map = """| Time Step | Channel A | Channel B | Channel C | Channel D |
|-----------|-----------|-----------|-----------|-----------|\n"""

        for i, state in enumerate(observations):
            interference_map += f"|{i+1}| {" ".join([digit for digit in str(state[0])])}| {" ".join([digit for digit in str(state[1])])}| {" ".join([digit for digit in str(state[2])])}| {" ".join([digit for digit in str(state[3])])}|\n"
        #interference_map += "\n"
        return interference_map
    
    def create_prompt(self, message):

        self.current_episode_messages.append({"role": "user", "content": message})

        pattern = r"^\|\s*(\d+)\s*\|"
        matches = re.findall(pattern, message, re.MULTILINE)
        last_time_step = int(matches[-1])
        started_assistant_message = f"""Here is the continuation of the time step table:

| Time Step | Channel A | Channel B | Channel C | Channel D |
|-----------|-----------|-----------|-----------|-----------|
|{last_time_step+1}|"""
        
        #self.current_episode_messages.append({"role": "assistant", "content": started_assistant_message})
        
        self.add_message({"role": "assistant", "content": started_assistant_message})

        prompt = self.tokenizer.apply_chat_template(
                    self.current_episode_messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False)

        #print(prompt)
        prompt = prompt.strip()
        prompt = prompt[:-len(self.tokenizer.eos_token)]##prompt.replace(self.tokenizer.eos_token, "")
        #print(prompt)
        prompt = prompt[len(self.tokenizer.bos_token):]
        #print(prompt)
        return prompt

    def extract_action(self, response: str) -> gym.core.ActType:
        
        number_sequence = response
        print("Prediction : ", number_sequence)
        number_array = number_sequence.split("|")
        number_array_int = [int(num.replace(" ", "")) for num in number_array]

        #action = number_array_int.index(0)  # finding the first zero as an example action
        action = number_array_int
        return action
        #raise ValueError("Failed to extract action from response.")
    
    def parse_output(self, prompt, output):

        output =output[len(self.tokenizer.decode(self.tokenizer.encode(prompt))):]
        return output

    def llm(self,
        prompt,
        ):

        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # input_ids = inputs["input_ids"].to(self.device)
        # generation_config = GenerationConfig(
        #      **self.generate_config_dict,
        #  )    
        # with torch.no_grad():
            
        #     generation_output = self.model.generate(
        #         input_ids=input_ids,
        #         return_dict_in_generate=True,
        #         generation_config=generation_config,
        #         #stopping_criteria=self.stopping_criteria,
        #         #output_hidden_states= True,
        #         #output_scores=True,
        #         #output_attentions=True,
        #         ## Bug generation configuratio args need to be added as keyword arguments
        #         #**self.generate_config_dict
        #     )
        #     s = generation_output.sequences[0]
        # output = self.tokenizer.decode(s)

        #print(output)

        model_out = outlines.models.Transformers(self.model, self.tokenizer)


        #[1-9]|[1-9][0-9]\| #
        #    r"([0-1]\|[0-1]\|[0-1]\|[0-1])")#,
        generator = outlines.generate.regex(
            model_out,
            r"([0-9]|[1-9] [0-9])(\|([0-9]|[1-9] [0-9]))(\|([0-9]|[1-9] [0-9]))(\|([0-9]|[1-9] [0-9]))")
        output = generator(prompt, max_tokens=30)
        return output

    def act(self, observations):
        message = self.format_observation(observations)
        #self.current_episode_messages += [{"role": "user", "content": message}]
        prompt = self.create_prompt(message)
        print(prompt)
        response = self.llm(prompt)
        #response = self.parse_output(prompt, response)
        #response = '{"next_row": "'+ response
        print(response)
        try:
            action = self.extract_action(response)
        except Exception as e:
            print("Exception catched ", e)
            return None

        #self.current_episode_messages += [{"role": "assistant", "content": response}]
        return action


class llm_agent_open_reasoning_noise(Agent):
    def __init__(self, model, tokenizer, device, generate_config_dict=None, ppo_config_dict=None):
        super().__init__(model, tokenizer, device, generate_config_dict, ppo_config_dict)
        self.stopping_criteria = None 
    def get_system_prompt(self) -> str:
        # Starting system prompt or any initialization text if needed
        return "Initialize system state."

    def format_observation(self, observations) -> str:
        interference_map = """OBSERVATIONS:| Time Step | Channel A | Channel B | Channel C | Channel D |
|-----------|-----------|-----------|-----------|-----------|\n"""

        # | Time Step | Channel A | Channel B | Channel C | Channel D |
        # |-----------|-----------|-----------|-----------|-----------|
        # |1|1|0|0|0|
        # |2|0|1|0|0|
        # |3|0|0|1|0|
        # |4|0|0|0|1|
        # |5|1|0|0|0|
        # |6|0|1|0|0|
        # |7|0|0|1|0|
        # |8|0|0|0|1|
        # |9|1|0|0|0|
        # |10|0|1|0|0|
        # Continue<|eot_id|>
        # <|start_header_id|>assistant<|end_header_id|>
        # Here is the continuation of the time step table:

        # | Time Step | Channel A | Channel B | Channel C | Channel D |
        # |-----------|-----------|-----------|-----------|-----------|

        for i, state in enumerate(observations):
            interference_map += f"|{i+1}|{state[0]}|{state[1]}|{state[2]}|{state[3]}|\n"

        def autocorrelation(data, lag=4):
            n = len(data)
            mean = np.mean(data)
            c0 = np.sum((data - mean) ** 2)
            c1 = np.sum((data[:-lag] - mean) * (data[lag:] - mean))
            if c0 == 0:  # This is to avoid division by zero error
                return 0
            return c1 / c0
        
        #METRICS: 
        # Channel A Autocorrelation with the values 4 time steps ago: 0.87
        # Channel B Autocorrelation with the values 4 time steps ago: 0.87
        # Channel C Autocorrelation with the values 4 time steps ago: 0.83
        # Channel D Autocorrelation with the values 4 time steps ago: 0.82
        data = np.array(observations).T
        interference_map += "METRICS:\n"
        lags = [1,2,3,4]
        channels = ["A", "B", "C", "D"]
        for lag in lags:
            # Calculate and print autocorrelation for each column with lag 1
            column_autocorrelations = [autocorrelation(col, lag=lag) for col in data]
            for index, ac in enumerate(column_autocorrelations):
                #print(f"Autocorrelation of Channel {channels[index]} with lag {lag}: {ac:.4f}")
                if ac > 0.4: 
                    ac = "High"
                elif ac > 0.2:
                    ac = "Medium"
                else:
                    ac = "Low"
                #interference_map +=f"Channel {channels[index]} Autocorrelation with the values {lag} time steps ago: {ac:.2f} \n"
                interference_map +=f"Channel {channels[index]} Autocorrelation with the values {lag} time steps ago: {ac} \n"
       
        interference_map += "\n"
        return interference_map
    
    def create_prompt(self, message):#, user_message="GPT4 Correct User: ", system_message= "GPT4 Correct Assistant:", end_message= "<|end_of_turn|>"): 
        
        # pattern_1 ="""
        # Time Step Channel A Channel B Channel C Channel D
        # 1| 1 1 0 0
        # 2| 1 1 0 0
        # 3| 0 0 1 1
        # 4| 0 0 1 1
        # 5| 1 1 0 0"""
        # answer_1 = """{"next_row": "1 1 0 0"}"""
        # self.current_episode_messages = [{"role": "user", "content": pattern_1},
        #         {"role": "assistant", "content": answer_1}]
        
        #self.current_episode_messages.append({"role": "assistant", "content": '{"next_row": "'})

        self.current_episode_messages = [{"role": "user", "content": message}]

        pattern = r"^\|\s*(\d+)\s*\|"
        matches = re.findall(pattern, message, re.MULTILINE)
        last_time_step = int(matches[-1])
#         started_assistant_message = f"""Here is the continuation of the time step table:

# | Time Step | Channel A | Channel B | Channel C | Channel D |
# |-----------|-----------|-----------|-----------|-----------|
# |{last_time_step+1}|"""

        start_assistant_message= "Given the observations and metrics, I'll try to predict the next time step."
        
        self.current_episode_messages.append({"role": "assistant", "content": start_assistant_message})
        

        prompt = self.tokenizer.apply_chat_template(
                    self.current_episode_messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False)

        #print(prompt)
        prompt = prompt.strip()
        prompt = prompt[:-len(self.tokenizer.eos_token)]##prompt.replace(self.tokenizer.eos_token, "")
        #print(prompt)
        prompt = prompt[len(self.tokenizer.bos_token):]
        #print(prompt)
        return prompt

    def extract_action(self, response: str) -> gym.core.ActType:
        pattern = r"^\|\s*(\d+)\s*\|.*"
        matches = list(re.finditer(pattern, response, re.MULTILINE))

        if matches:
            last_match = matches[-1]
            last_match = last_match.group(0)
            # print(f'Full match: {last_match.group(0)}')
            # print(f'Number: {last_match.group(1)}')
        else:
            print("No matches found.")
        number_sequence = last_match
        print("Prediction : ", number_sequence)
        number_array = number_sequence.split("|")
        number_array_int = [int(num.strip()) for num in number_array if num.strip() != ""]
        #action = number_array_int.index(0)  # finding the first zero as an example action
        action = number_array_int[-4:]
        return action
        #raise ValueError("Failed to extract action from response.")
    
    def parse_output(self, prompt, output):

        output =output[len(self.tokenizer.decode(self.tokenizer.encode(prompt))):]
        return output

    def llm(self,
        prompt,
        ):

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        generation_config = GenerationConfig(
             **self.generate_config_dict,
         )    
        with torch.no_grad():
            
            generation_output = self.model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                generation_config=generation_config,
                #stopping_criteria=self.stopping_criteria,
                #output_hidden_states= True,
                #output_scores=True,
                #output_attentions=True,
                ## Bug generation configuratio args need to be added as keyword arguments
                #**self.generate_config_dict
            )
            s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)

        # #print(output)

        # model_out = outlines.models.Transformers(model, tokenizer)


        # #[1-9]|[1-9][0-9]\|
        # generator = outlines.generate.regex(
        #     model_out,
        #    r"([1-9]|[1-9][0-9])(\|([1-9]|[1-9][0-9]))(\|([1-9]|[1-9][0-9]))(\|([1-9]|[1-9][0-9]))")
        # output = generator(prompt, max_tokens=30)
        return output

    def act(self, observations):
        message = self.format_observation(observations)
        #self.current_episode_messages += [{"role": "user", "content": message}]
        prompt = self.create_prompt(message)
        print(prompt)
        response = self.llm(prompt)
        response = self.parse_output(prompt, response)
        #response = '{"next_row": "'+ response
        print(response)
        try:
            action = self.extract_action(response)
        except Exception as e:
            print("Exception catched ", e)
            return None

        #self.current_episode_messages += [{"role": "assistant", "content": response}]
        return action