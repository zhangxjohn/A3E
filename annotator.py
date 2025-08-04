import warnings
warnings.filterwarnings('ignore')

import os
import re
import time
import json
from typing import List, Union

import httpx
try:
    from openai import OpenAI, AzureOpenAI
except:
    pass
from prompts import system_message_en, entities_analysis_prompt_en, entities_analysis_prompt_en_llama3_8b, sentences_alignment_prompt_en, sentences_causality_prompt_en, sentences_causality_prompt_en_llama_8b, conclusions_analogy_prompt_en, modify_dissimilar_prompt_en, conclusions_analogy_prompt_abbr_en

PROXY = os.environ.get('HTTPS_PROXY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
AZURE_ENDPOINT = os.environ.get('AZURE_ENDPOINT')
AZURE_API_KEY = os.environ.get('AZURE_API_KEY')

class AnalogyAutoAnnotator:
    """
    Parameters
    -----------
    llm_name: str or List[str], default='llama3'
        The name of the language model. Optional values include 'llama3', 'gpt-4-0125-preview', 'gpt-4-turbo-2024-04-09', 'gpt-4', 'gpt-3.5-turbo-0125', 'gpt-4o'.
        If it's a List[str], it represents using multiple language models for joint reasoning.
    temperature: float, default=0.3, range=[0, 1]
        The randomness of the generated text. The higher the value, the more diverse the generated text.
    max_tokens: int, default=2048
        The maximum length of the generated text.
    modify_dsrs: bool, default=False
        Whether to modify the reason and type of dissimilar groups.
    summarize_by_llm: bool, default=False
        Whether to summarize the conclusion by the language model.
    verbose: bool, default=False
        Whether to print the generated text.
    e2e: bool, default=False
        Whether to use end-to-end generation, i.e., only call the model once to determine the type of analogy. This mode requires higher capabilities for LLM.
    """
    def __init__(self,
                llm_name: Union[str, List[str]]='llama3',
                temperature: float=0.3,
                max_tokens: int=2048,
                modify_dsrt: bool=False,
                summarize_by_llm: bool=False,
                verbose: bool=False,
                e2e: bool=False,
                en_prompt: bool=False,
        ):
        if isinstance(llm_name, str):
            self.llm_name = [llm_name]*4
        else:
            assert len(llm_name) == 4, "The length of llm_name should be 4."
            self.llm_name = llm_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.modify_dsrt = modify_dsrt
        self.summarize_by_llm = summarize_by_llm
        self.verbose = verbose
        self.e2e = e2e
        self.en_prompt = en_prompt

        self.llama3_tokens_usages = {"prompt": 0, "completion": 0, "total": 0}
        self.gpt4_tokens_usages = {"prompt": 0, "completion": 0, "total": 0}
        self.tokens_usages = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.time_cost = 0

    def run(self, sample: dict):
        """
        Run the auto-annotation for the given sample.

        Parameters
        -----------
        sample: dict
            A sample with keys 'base', 'target'.

        Returns
        ---------
        analogy_type: str
            The predicted analogy type.
        reason_process: str
            The reasoning process of the auto-annotation.
        """
        start_time = time.time()
        base, target = sample['base'], sample['target']
        modify_dsrt_process = ''
        analogy_type_check_response = ''
        entities, sentences, causalities = '', '', ''

        try:
            if self.e2e:
                E2E = e2e_analogy_prompt.format(base=base, target=target)
                if self.verbose:
                    print(E2E)
                    print("--"*30)
                analogy_type_check_response, usage_tokens = self.get_gpt_completion(
                    prompt=E2E,
                    model=self.llm_name[0],
                    temperature=self.temperature, 
                    max_tokens=self.max_tokens,
                    count_tokens=True)
                self.count_usage_tokens(llm_name=self.llm_name[0], usage_tokens=usage_tokens)
                if self.verbose: 
                    print(analogy_type_check_response)
                    print("--"*30)
                parsing_json_data = self.extract_json_from_string(analogy_type_check_response)
            else:
                if not self.en_prompt:
                    EAP = entities_analysis_prompt.format(base=base, target=target)
                else:
                    if self.llm_name[0] not in ["Meta-Llama-3.1-8B-Instruct"]:
                        EAP = entities_analysis_prompt_en.format(base=base, target=target)
                    else:
                        EAP = entities_analysis_prompt_en_llama3_8b.format(base=base, target=target)
                if self.verbose: 
                    print(EAP)
                    print("--"*30)
                entities, usage_tokens = self.get_gpt_completion(
                    prompt=EAP,
                    model=self.llm_name[0], 
                    temperature=self.temperature,
                    max_tokens=self.max_tokens, 
                    count_tokens=True)
                self.count_usage_tokens(llm_name=self.llm_name[0], usage_tokens=usage_tokens)
                if self.verbose: 
                    print(entities)
                    print("--"*30)

                if not self.en_prompt:
                    SAP = sentences_alignment_prompt.format(base=base, target=target)
                else:
                    SAP = sentences_alignment_prompt_en.format(base=base, target=target)
                if self.verbose: 
                    print(SAP)
                    print("--"*30)
                sentences, usage_tokens = self.get_gpt_completion(
                    prompt=SAP,
                    model=self.llm_name[1],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    count_tokens=True)
                self.count_usage_tokens(llm_name=self.llm_name[1], usage_tokens=usage_tokens)
                if self.verbose: 
                    print(sentences)
                    print("--"*30)

                if not self.en_prompt:
                    SCP = sentences_causality_prompt.format(base=base, target=target, sentences=sentences)
                else:
                    if self.llm_name[0] not in ["Meta-Llama-3.1-8B-Instruct"]:
                        SCP = sentences_causality_prompt_en.format(base=base, target=target, sentences=sentences)
                    else:
                        SCP = sentences_causality_prompt_en_llama_8b.format(base=base, target=target, sentences=sentences)
                if self.verbose: 
                    print(SCP)
                    print("--"*30)
                causalities, usage_tokens = self.get_gpt_completion(
                    prompt=SCP,
                    model=self.llm_name[2],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    count_tokens=True)
                self.count_usage_tokens(llm_name=self.llm_name[2], usage_tokens=usage_tokens)
                if self.verbose: 
                    print(causalities)
                    print("--"*30)

                try:
                    entities_info = self.extract_json_from_string(entities)
                    similar_group = self.extract_type_group_list(causalities, default='相似组')
                    dissimilar_group = self.extract_type_group_list(causalities, default='不相似组')
                    irrelevant_group = self.extract_type_group_list(causalities, default='无关组')
                    extract_summarize_info = 'succeed'
                except:
                    extract_summarize_info = 'failure'

                if not self.en_prompt:
                    CAP = conclusions_analogy_prompt.format(base=base, target=target, entities=entities, sentences=sentences, causalities=causalities)
                else:
                    CAP = conclusions_analogy_prompt_en.format(base=base, target=target, entities=entities, sentences=sentences, causalities=causalities)
                if self.summarize_by_llm or extract_summarize_info == 'failure':
                    if self.verbose:
                        print(CAP)
                        print("--"*30)
                    analogy_type_check_response, usage_tokens = self.get_gpt_completion(
                        prompt=CAP,
                        model=self.llm_name[3],
                        temperature=self.temperature, 
                        max_tokens=self.max_tokens,
                        count_tokens=True)
                    self.count_usage_tokens(llm_name=self.llm_name[3], usage_tokens=usage_tokens)
                    if self.verbose: 
                        print(analogy_type_check_response)
                        print("--"*30)
                    parsing_json_data = self.extract_json_from_string(analogy_type_check_response)
                else:
                    if isinstance(entities_info['same-words count'], list):
                        same_words_count = f"{len(entities_info['same-words count'])}"
                    elif isinstance(entities_info['same-words count'], str) and entities_info['same-words count'].isdigit():
                        same_words_count = f"{entities_info['same-words count']}"
                    elif isinstance(entities_info['same-words count'], str) and '>' in entities_info['same-words count']:
                        same_words_count = "1"
                    else:
                        same_words_count = "0"
                    parsing_json_data = {
                        "background": entities_info['background'],
                        "role": entities_info['role'],
                        "plot": entities_info['plot'],
                        "same-words count": same_words_count, 
                        "similar-set count": f"{len(similar_group)}",
                        "dissimilar-set count": f"{len(dissimilar_group)}",
                        "irrelevant-set count": f"{len(irrelevant_group)}"
                    }
                    analogy_type_check_response = f"```json\n{json.dumps(parsing_json_data, ensure_ascii=False, indent=2)}\n```"

            if self.modify_dsrt is True and isinstance(dissimilar_group, list) and len(dissimilar_group) > 0 and len(dissimilar_group) <= 2 and self.e2e is False:
                try:
                    similar_set_count = int(parsing_json_data['similar-set count'])
                    dissimilar_set_count = int(parsing_json_data['dissimilar-set count'])
                    irrelevant_set_count = int(parsing_json_data['irrelevant-set count'])
                    statements = self.split_statements(sentences)
                    reasons = self.split_statements(causalities)
                    for ds_id in dissimilar_group:
                        statement = statements[ds_id]
                        reason = reasons[ds_id]
                        if not self.en_prompt:
                            MDP = modify_dissimilar_prompt.format(statement=statement, reason=reason)
                        else:
                            MDP = modify_dissimilar_prompt_en.format(statement=statement, reason=reason)
                        if self.verbose:
                            print(MDP)
                            print("--"*30)
                        dissimilar_type_modify_response, usage_tokens = self.get_gpt_completion(
                            prompt=MDP,
                            model=self.llm_name[3],
                            temperature=self.temperature, 
                            max_tokens=self.max_tokens,
                            count_tokens=True)
                        modify_dsrt_process += f"\n{MDP}\n{dissimilar_type_modify_response}\n"
                        self.count_usage_tokens(llm_name=self.llm_name[3], usage_tokens=usage_tokens)
                        if self.verbose: 
                            print(dissimilar_type_modify_response)
                            print("--"*30)
                        parsing_modify_info = self.extract_json_from_string(dissimilar_type_modify_response)
                        if self.en_prompt:
                            type1 = self.cn_to_en_mapping('相似组')
                            type2 = self.cn_to_en_mapping('无关组')
                        else:
                            type1 = '相似组'
                            type2 = '无关组'
                        if parsing_modify_info['type'] == type1:
                            similar_set_count += 1
                            dissimilar_set_count -= 1
                        elif parsing_modify_info['type'] == type2:
                            irrelevant_set_count += 1
                            dissimilar_set_count -= 1
                    parsing_json_data['similar-set count'] = f"{similar_set_count}"
                    parsing_json_data['dissimilar-set count'] = f"{dissimilar_set_count}"
                    parsing_json_data['irrelevant-set count'] = f"{irrelevant_set_count}"
                except:
                    modify_dsrt_process = ''
            parsing_json_data, actual_pred_type = self.analogy_type_annotation(parsing_json_data)
        except:
            actual_pred_type = 'Anomaly'
            reasoning_process = ''
            parsing_json_data = {}

        end_time = time.time()
        self.time_cost = end_time - start_time

        if len(entities) > 0:
            if not self.en_prompt:
                cap_abbr = conclusions_analogy_prompt_abbr.format(base=base, target=target, entities=entities, sentences=sentences, causalities=causalities)
            else:
                cap_abbr = conclusions_analogy_prompt_abbr_en.format(base=base, target=target, entities=entities, sentences=sentences, causalities=causalities)
        else:
            cap_abbr = ''
        if len(modify_dsrt_process) > 0:
            reasoning_process = cap_abbr + analogy_type_check_response + '\n' + modify_dsrt_process
        else:
            reasoning_process = cap_abbr + analogy_type_check_response if not self.e2e else analogy_type_check_response

        return actual_pred_type, reasoning_process, parsing_json_data
    
    def count_usage_tokens(self, llm_name, usage_tokens):
        """
        Count the usage of tokens for the given language model.

        Parameters
        -----------
        llm_name: str
            The name of the language model.
        usage_tokens: List[int]
            The usage of tokens for the prompt and completion.
        """
        if llm_name in ['Meta-Llama-3-70B-Instruct', 'Llama3-70B-Chinese-Chat', 'llama3', "Meta-Llama-3.1-8B-Instruct"]: 
            self.llama3_tokens_usages["prompt"] += usage_tokens[0]
            self.llama3_tokens_usages["completion"] += usage_tokens[1]
            self.llama3_tokens_usages["total"] += usage_tokens[2]
        elif llm_name in ['gpt-4-0125-preview', 'gpt-4-turbo-2024-04-09', 'gpt-4', 'gpt-3.5-turbo-0125', 'gpt-4o']:
            self.gpt4_tokens_usages["prompt"] += usage_tokens[0]
            self.gpt4_tokens_usages["completion"] += usage_tokens[1]
            self.gpt4_tokens_usages["total"] += usage_tokens[2]
        self.tokens_usages["prompt_tokens"] += usage_tokens[0]
        self.tokens_usages["completion_tokens"] += usage_tokens[1]
        self.tokens_usages["total_tokens"] += usage_tokens[2]

    def extract_json_from_string(self, input_str):
        """
        Extract the last JSON object from the input string.

        Parameters  
        -----------
        input_str: str
            The input string. 

        Returns 
        ---------
        json_data: dict or None
            The extracted JSON object.
        """
        try:
            matches = re.findall(r'\{[^}]+\}', input_str)
            if not matches:
                return None
            last_json_str = matches[-1]
            last_json_str = last_json_str.replace('：', ':')
            last_json_str = re.sub(r'"\s*:\s*".+?"\s*(?=\")', lambda m: m.group(0) + ',', last_json_str)
            last_json_str = re.sub(r',\s*}', '}', last_json_str)

            json_data = json.loads(last_json_str)
            return json_data
        except (AttributeError, json.JSONDecodeError) as e:
            log = f"Error extracting JSON: {e}"

            max_retries = 3
            retry_count = 0
            while True:
                user_msg = STR2DICT_FORMAT_FIXED_PROMPT_TEMPLATE.format(error_log=log, str_input=last_json_str)
                last_json_str = self.get_gpt_completion(
                    prompt=user_msg,
                    model=self.llm_name[3],
                )
                try:
                    json_data = json.loads(last_json_str)
                    return json_data
                except:
                    retry_count += 1
                    if retry_count >= max_retries:
                        break
            print(log)
            print(last_json_str)
            return None
        
    def split_statements(self, input_str):
        """
        Split the input string into a list of statements.

        Parameters
        -----------
        input_str: str
            The input string.

        Returns
        ---------
        items_dict: dict
            The dictionary of items.
        """
        items_dict = {}
        items = re.findall(r'(\d+)\.(.*)', input_str)
        for item in items:
            sequence_number, content = item
            items_dict[sequence_number] = content.strip()
        return items_dict

    def cn_to_en_mapping(self, word):
        mapping = {
            "相似组": "Similar Group",
            "不相似组": "Dissimilar Group",
            "无关组": "Irrelevant Group"
        }
        return mapping[word]

    def extract_type_group_list(self, input_str, default='不相似组'):
        """
        Extract the type-group list from the input string.

        Parameters
        -----------
        input_str: str
            The input string.
        default: str, default='不相似组'
            The default group name.

        Returns
        ---------
        group_numbers: List[int]
            The list of group numbers.
        """
        if self.en_prompt:
            default = self.cn_to_en_mapping(default)
            group_match = re.search(rf'{default}:\s*\[(.*?)\]', input_str)
            if group_match:
                group_numbers = re.findall(r'\d+', group_match.group(1))
                group_numbers = [int(num) for num in group_numbers]
            else:
                group_numbers = []
        else:
            group_match = re.search(rf'{default}：\[(.*?)\]', input_str)
            if group_match:
                group_numbers = re.findall(r'\d+', group_match.group(1))
            else:
                group_numbers = []
        return group_numbers

    def analogy_type_annotation(self, data):
        """
        Analyze the JSON data and determine the type of analogy.

        Parameters
        -----------
        data: dict
            The JSON data.

        Returns
        ---------
        data: dict
            The updated JSON data.
        actual_pred_type: str
            The predicted type of analogy.
        """
        if data is None:
            return None, 'Anomaly'
        len_similar_set = int(data['similar-set count'])
        len_dissimilar_set = int(data['dissimilar-set count'])
        len_irrelevant_set = int(data['irrelevant-set count'])

        if data['background'] == 'False' and data['role'] == 'False':
            data['entities'] = 'dissimilar'
        else:
            data['entities'] = 'similar'

        if len_similar_set > (len_dissimilar_set+len_irrelevant_set)/2 or (data['background'] == 'True' and data['role'] == 'True' and data['plot'] == 'True'): 
            data['one-order relations'] = 'similar'
        else:
            data['one-order relations'] = 'dissimilar'

        if len_dissimilar_set > 0 or len_similar_set < len_irrelevant_set/2:
            data['higher-order relations'] = 'dissimilar'
        else:
            data['higher-order relations'] = 'similar'

        if data['one-order relations'] == 'dissimilar':
            if ('>' in data['same-words count']) or ('>' not in data['same-words count'] and int(data['same-words count']) > 0) or data['entities'] == 'similar':
                return data, 'Mere Appearance'
            
        if data['entities'] == 'dissimilar' and data['one-order relations'] == 'dissimilar' and data['higher-order relations'] == 'dissimilar':
            if data['background'] == 'False' and data['role'] == 'False' and data['plot'] == 'True':
                data['one-order relations'] = 'similar'

        if data['entities'] == 'similar' and data['one-order relations'] == 'similar' and data['higher-order relations'] == 'similar':
            return data, 'Literally Similar'
        elif data['entities'] == 'dissimilar' and data['one-order relations'] == 'similar' and data['higher-order relations'] == 'similar':
            return data, 'True Analogy'
        elif data['entities'] == 'dissimilar' and data['one-order relations'] == 'similar' and data['higher-order relations'] == 'dissimilar':
            return data, 'False Analogy'
        elif data['entities'] == 'similar' and data['one-order relations'] == 'similar' and data['higher-order relations'] == 'dissimilar':
            return data, 'Surface Similar'
        elif data['entities'] == 'similar' and data['one-order relations'] == 'dissimilar' and data['higher-order relations'] == 'dissimilar':
            return data, 'Mere Appearance'
        elif data['entities'] == 'dissimilar' and data['one-order relations'] == 'dissimilar' and data['higher-order relations'] == 'dissimilar':
            return data, 'Anomaly'
        else:
            return data, 'Anomaly'
        
    def get_gpt_completion(self,
                        prompt,
                        system=None,
                        temperature=0, 
                        max_tokens=1024,
                        model="llama3", 
                        api_key=None, 
                        base_url=None, 
                        count_tokens=False,
                        **kwargs):
        """
        Get the completion of the given prompt using the specified language model.

        Parameters
        -----------  
        prompt: str
            The prompt for the language model.
        system: str, optional
            The system response for the prompt. If provided, it will be added to the prompt.
        temperature: float, default=0
            The randomness of the generated text. The higher the value, the more diverse the generated text.
        max_tokens: int, default=1024
            The maximum length of the generated text.
        model: str, default='llama3'
            The name of the language model. Optional values include 'llama3', 'gpt-4-0125-preview', 'gpt-4-turbo-2024-04-09', 'gpt-4', 'gpt-3.5-turbo-0125', 'gpt-4o'.
        api_key: str, optional
            The API key for the OpenAI API.
        base_url: str, optional
            The base URL for the OpenAI API.
        count_tokens: bool, default=False
            Whether to count the usage of tokens.
        **kwargs:
            Additional keyword arguments for the OpenAI API.

        Returns
        ---------
        completion: str
            The generated text.
        usage_tokens: List[int]
            The usage of tokens for the prompt and completion.
        """
        if model == 'llama3':
            model = "Meta-Llama-3.1-70B-Instruct"
        elif model == 'kimi':
            model = 'moonshot-v1-8k'
        elif model == 'llama3-8B':
            model = "Meta-Llama-3.1-8B-Instruct"

        if base_url is None:
            if model in ['Meta-Llama-3.1-70B-Instruct']:
                api_key = '1234'
                base_url = 'your_base_url'
            elif model in ['Meta-Llama-3.1-8B-Instruct']:
                api_key = '1234'
                base_url = 'your_base_url'
            elif model in ['gpt-4-0125-preview', 'gpt-4-turbo-2024-04-09', 'gpt-4', 'gpt-3.5-turbo-0125', 'gpt-4o']:    
                api_key = OPENAI_API_KEY
                base_url = None
            else:
                raise ValueError("Invalid language model name.")
        else:
            api_key = '1234'
            base_url = 'your_base_url'
        
        if model in ['Meta-Llama-3-70B-Instruct', 'Meta-Llama-3.1-70B-Instruct', 'Meta-Llama-3.1-8B-Instruct']:
            kwargs.update({
                "extra_body":{"stop_token_ids":[128009, 128001]}
                }
            )

        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        else:
            if not self.en_prompt:
                messages.append({"role": "system", "content": system_message})
            else:
                messages.append({"role": "system", "content": system_message_en})
        messages.append({"role": "user", "content": prompt})    
            
        if api_key is None:
            client = OpenAI(
                api_key=OPENAI_API_KEY,
                http_client=httpx.Client(
                    proxies=PROXY
                )
            )
        else:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )

        if model == "azure":
            client = AzureOpenAI(
                azure_endpoint=AZURE_ENDPOINT,
                api_key=AZURE_API_KEY,
                api_version="2023-03-15-preview"
            )
        
        if kwargs.get('stream') is True:
            completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
            )
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="")
        else:
            attempts = 0
            max_attempts = 10
            while attempts < max_attempts: 
                try:
                    try:
                        completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                        )
                        total_tokens = completion.usage.total_tokens
                        prompt_tokens = completion.usage.prompt_tokens
                        completion_tokens = completion.usage.completion_tokens
                        if count_tokens:
                            return completion.choices[0].message.content, [prompt_tokens, completion_tokens, total_tokens]
                        else:
                            return completion.choices[0].message.content
                    except:
                        completion = client.completions.create(
                        model=model,
                        prompt=prompt,
                        temperature=temperature,
                        **kwargs
                        )
                        total_tokens = completion.usage.total_tokens
                        prompt_tokens = completion.usage.prompt_tokens
                        completion_tokens = completion.usage.completion_tokens
                    
                        if count_tokens:
                            return completion.choices[0].text, [prompt_tokens, completion_tokens, total_tokens]
                        else:
                            return completion.choices[0].text
                except KeyboardInterrupt:
                    break
                except:
                    attempts += 1
                    if attempts < max_attempts:
                        print("Trying again...")
                    else:
                        print("Max attempts reached. Exiting.")


system_message = """你是一个高度专业、知识渊博且友好的大型语言模型助手，能够提供准确、详细且具有建设性的回答。

Behavioral Guidelines:
- 服从命令：在回答用户问题前，认真分析用户每一句指令的需求，严格遵循用户的指令需求回答问题。
- 准确性和详细性：在回答用户问题时，确保提供准确和详细的信息。使用可靠的来源来支持你的回答，避免传播错误信息。
- 专业和友好：保持专业和友好的语气。即使用户的问题很复杂或模糊，也要耐心回答，提供尽可能多的帮助。
- 清晰和简洁：解释概念时，保持清晰和简洁。避免使用过于复杂的术语，除非用户明确要求更专业的解释。
- 结构化和组织化：你的回答应该结构化良好，便于用户理解。例如，使用段落、列表或编号来组织信息。
"""


entities_analysis_prompt = '''Base: 有一只乌龟和一只兔子决定进行一场比赛。兔子相信自己一定能赢，因为它跑得比乌龟快得多。比赛开始后，兔子迅速冲到前面，而乌龟则慢慢地爬着。途中，兔子觉得自己跑得太快了，离终点还远着呢，于是决定在树下休息一会儿。它睡着了，而乌龟则不停地向前爬。最后，当兔子醒来时，发现乌龟已经爬过了终点线，赢得了比赛。
Target: 有一个小偷总是能成功逃脱警察的抓捕，因为他跑得非常快。一天，警察假装在某个地方展开了大量的巡逻，而实际上他们悄悄埋伏在另一条小路上。小偷如同往常一样，偷了东西后飞速逃跑，途中，他跑着跑着被一辆突然出现的汽车给撞昏过去，送到了医院。最后，当小偷醒来时，发现身边全是警察。

Question: 严格按原文表述, 详细分析Base与Target中是否有相似的具体背景（字面理解）、人物角色职责（字面理解）、情节发展起伏（归纳理解）和共同词汇（字面理解）。如果没有共同词汇，返回空list[]即可。
Answer: 
- 在Base中，主角是乌龟和兔子，他们进行了一场比赛，兔子因为自信而在途中休息，最终乌龟赢得了比赛。
- 在Target中，主角是小偷和警察，小偷因为跑得快而总是逃脱，但警察设下陷阱，最终小偷被抓住。
因此，两者的具体背景（龟兔比赛与警察抓小偷）不同：寓言不同于真实事件, 人物角色职责（乌龟与兔子，小偷与警察）也不同：乌龟与兔子是竞赛关系，小偷与警察是抓捕关系。但在情节发展起伏上有一定的相似性，都是关于追逐导致失败的故事。此外，没有共同词汇。
综上所述，
```
{{
   "background": "False", 
   "role": "False",
   "plot": "True",
   "same-words count": []
}}
```


Base: {base}
Target: {target}

Question: 严格按原文表述, 详细分析Base与Target中是否有相似的具体背景（字面理解）、人物角色职责（字面理解）、情节发展起伏（归纳理解）和共同词汇（字面理解）。如果没有共同词汇，返回空list[]即可。
Answer:(保持上下Answer输出格式的一致性) 
''' 

sentences_alignment_prompt = '''Base: 有一只乌龟和一只兔子决定进行一场比赛。兔子相信自己一定能赢，因为它跑得比乌龟快得多。比赛开始后，兔子迅速冲到前面，而乌龟则慢慢地爬着。途中，兔子觉得自己跑得太快了，离终点还远着呢，于是决定在树下休息一会儿。它睡着了，而乌龟则不停地向前爬。最后，当兔子醒来时，发现乌龟已经爬过了终点线，赢得了比赛。
Target: 有一个小偷总是能成功逃脱警察的抓捕，因为他跑得非常快。一天，警察假装在某个地方展开了大量的巡逻，而实际上他们悄悄埋伏在另一条小路上。小偷如同往常一样，偷了东西后飞速逃跑，途中，他跑着跑着被一辆突然出现的汽车给撞昏过去，送到了医院。最后，当小偷醒来时，发现身边全是警察。

Question: 严格按原文表述顺序，逐条对齐Target中与Base有相同结构关系(字面类似)的单语句，并返回。其中，
- 尽可能准确地匹配句子间的对应关系，体现结构上的相似性。
- 如果Base或Target中的句子没有对应的类似表述，则表示为：Base: [句子] v.s. Target: 无。 或者 Base: 无。 v.s. Target: [句子]。
- 每个句子中仅能含有一个句号（‘。’）。
- 请确保列出Base和Target中的所有句子，不要遗漏。

Answer: 
1. Base: 有一只乌龟和一只兔子决定进行一场比赛。 v.s. Target: 无。
2. Base: 兔子相信自己一定能赢，因为它跑得比乌龟快得多。 v.s. Target: 有一个小偷总是能成功逃脱警察的抓捕，因为他跑得非常快。
3. Base: 比赛开始后，兔子迅速冲到前面，而乌龟则慢慢地爬着。 v.s. Target: 无。
4. Base: 途中，兔子觉得自己跑得太快了，离终点还远着呢，于是决定在树下休息一会儿。v.s. Target: 一天，警察假装在某个地方展开了大量的巡逻，而实际上他们悄悄埋伏在另一条小路上。
5. Base: 它睡着了，而乌龟则不停地向前爬。 v.s. Target: 小偷如同往常一样，偷了东西后飞速逃跑，途中，他跑着跑着被一辆突然出现的汽车给撞昏过去，送到了医院。
6. Base: 最后，当兔子醒来时，发现乌龟已经爬过了终点线，赢得了比赛。 v.s. Target: 最后，当小偷醒来时，发现身边全是警察。


Base: {base}
Target: {target}

Question: 严格按原文表述顺序，逐条对齐Target中与Base有相同结构关系(字面类似)的单语句，并返回。其中，
- 尽可能准确地匹配句子间的对应关系，体现结构上的相似性。
- 如果Base或Target中的句子没有对应的类似表述，则表示为：Base: [句子] v.s. Target: 无。 或者 Base: 无。 v.s. Target: [句子]。
- 每个句子中仅能含有一个句号（‘。’）。
- 请确保列出Base和Target中的所有句子，不要遗漏。

Answer:(保持上下Answer输出格式的一致性) 
'''

sentences_causality_prompt = '''Base: 有一只乌龟和一只兔子决定进行一场比赛。兔子相信自己一定能赢，因为它跑得比乌龟快得多。比赛开始后，兔子迅速冲到前面，而乌龟则慢慢地爬着。途中，兔子觉得自己跑得太快了，离终点还远着呢，于是决定在树下休息一会儿。它睡着了，而乌龟则不停地向前爬。最后，当兔子醒来时，发现乌龟已经爬过了终点线，赢得了比赛。
Target: 有一个小偷总是能成功逃脱警察的抓捕，因为他跑得非常快。一天，警察假装在某个地方展开了大量的巡逻，而实际上他们悄悄埋伏在另一条小路上。小偷如同往常一样，偷了东西后飞速逃跑，途中，他跑着跑着被一辆突然出现的汽车给撞昏过去，送到了医院。最后，当小偷醒来时，发现身边全是警察。

Question: 严格按原文表述顺序，逐条对齐Base与Target的相同结构关系的语句，并返回。
Answer: 
1. Base: 有一只乌龟和一只兔子决定进行一场比赛。 v.s. Target: 无。
2. Base: 兔子相信自己一定能赢，因为它跑得比乌龟快得多。 v.s. Target: 有一个小偷总是能成功逃脱警察的抓捕，因为他跑得非常快。
3. Base: 比赛开始后，兔子迅速冲到前面，而乌龟则慢慢地爬着。 v.s. Target: 无。
4. Base: 途中，兔子觉得自己跑得太快了，离终点还远着呢，于是决定在树下休息一会儿。v.s. Target: 一天，警察假装在某个地方展开了大量的巡逻，而实际上他们悄悄埋伏在另一条小路上。
5. Base: 它睡着了，而乌龟则不停地向前爬。 v.s. Target: 小偷如同往常一样，偷了东西后飞速逃跑，途中，他跑着跑着被一辆突然出现的汽车给撞昏过去，送到了医院。
6. Base: 最后，当兔子醒来时，发现乌龟已经爬过了终点线，赢得了比赛。 v.s. Target: 最后，当小偷醒来时，发现身边全是警察。

Question: 深层次挖掘原因，逐条详细地分析以上回答中因果逻辑相似、不相似、以及无关组的关系语对的类比对齐情况并归类。如果一组没有，返回空list[]即可。
Answer: 
1. 因为一方（Target）含有‘无’, 因此划归为无关组。
2. Base中的兔子自信能赢因为速度快，对应Target中小偷能逃脱因为跑得快，两者都是因为速度优势而自信或成功的例子，归为相似组。
3. 因为一方（Target）含有‘无’, 因此划归为无关组。
4. Base中描述的是兔子因为自满而停下来休息，而Target中描述的是警察使用计谋来捕捉小偷，两者描述的角色在结构上并不对应且内容不存在逻辑相关性，所以划归为不相关组。
5. 因为Base中描述的是乌龟在兔子休息时继续前进，而Target中描述的是小偷在逃跑过程中被汽车撞昏，两者都是描写了主要角色失去了意识，但导致的原因并不同：兔子休息是因为它认为乌龟速度太慢即使自己休息一会儿也无伤大雅，而小偷没有想停止脚步，只是在逃跑的途中被意外出现的汽车撞昏了，并非自信个人能力而主动躺倒被警察抓，所以划归为不相似组。
6. Base的结局是兔子醒来发现失败，与Target中小偷醒来被警察包围，两处都描述了主人公从昏迷或沉睡状态恢复意识后面临的不利局面，但导致的原因并不同：兔子是主观自信，而小偷是客观意外。因此，归为不相似组。
综上所述,
```
相似组：[2]
不相似组：[5, 6]
无关组：[1, 3, 4]
```


Base: {base}
Target: {target}

Question: 严格按原文表述顺序，逐条对齐Base与Target的相同结构关系的语句，并返回。
Answer: 
{sentences}

Question: 深层次挖掘原因，逐条详细地分析以上回答中因果逻辑相似、不相似、以及无关组的关系语对的类比对齐情况并归类。如果一组没有，返回空list[]即可。
Answer: (保持上下Answer输出格式的一致性, 即先逐条陈述原因再总结)
'''

conclusions_analogy_prompt = '''Base: 有一只乌龟和一只兔子决定进行一场比赛。兔子相信自己一定能赢，因为它跑得比乌龟快得多。比赛开始后，兔子迅速冲到前面，而乌龟则慢慢地爬着。途中，兔子觉得自己跑得太快了，离终点还远着呢，于是决定在树下休息一会儿。它睡着了，而乌龟则不停地向前爬。最后，当兔子醒来时，发现乌龟已经爬过了终点线，赢得了比赛。
Target: 有一个小偷总是能成功逃脱警察的抓捕，因为他跑得非常快。一天，警察假装在某个地方展开了大量的巡逻，而实际上他们悄悄埋伏在另一条小路上。小偷如同往常一样，偷了东西后飞速逃跑，途中，他跑着跑着被一辆突然出现的汽车给撞昏过去，送到了医院。最后，当小偷醒来时，发现身边全是警察。

Question: 严格按原文表述, 总结Base与Target中是否有相似的故事背景和角色或者共同词汇。
Answer: 
- 在Base中，主角是乌龟和兔子，他们进行了一场比赛，兔子因为自信而在途中休息，最终乌龟赢得了比赛。
- 在Target中，主角是小偷和警察，小偷因为跑得快而总是逃脱，但警察设下陷阱，最终小偷被抓住。
因此，两者的具体背景（龟兔比赛与警察抓小偷）不同：寓言不同于真实事件, 人物角色职责（乌龟与兔子，小偷与警察）也不同：乌龟与兔子是竞赛关系，小偷与警察是抓捕关系。但在情节发展起伏上有一定的相似性，都是关于追逐导致失败的故事。此外，没有共同词汇。
综上所述，
```
{{
   "background": "False", 
   "role": "False",
   "plot": "True",
   "same-words count": []
}}
```

Question: 严格按原文表述顺序，逐条对齐Base与Target的相同结构关系的语句，并返回。
Answer: 
1. Base: 有一只乌龟和一只兔子决定进行一场比赛。 v.s. Target: 无。
2. Base: 兔子相信自己一定能赢，因为它跑得比乌龟快得多。 v.s. Target: 有一个小偷总是能成功逃脱警察的抓捕，因为他跑得非常快。
3. Base: 比赛开始后，兔子迅速冲到前面，而乌龟则慢慢地爬着。 v.s. Target: 无。
4. Base: 途中，兔子觉得自己跑得太快了，离终点还远着呢，于是决定在树下休息一会儿。v.s. Target: 一天，警察假装在某个地方展开了大量的巡逻，而实际上他们悄悄埋伏在另一条小路上。
5. Base: 它睡着了，而乌龟则不停地向前爬。 v.s. Target: 小偷如同往常一样，偷了东西后飞速逃跑，途中，他跑着跑着被一辆突然出现的汽车给撞昏过去，送到了医院。
6. Base: 最后，当兔子醒来时，发现乌龟已经爬过了终点线，赢得了比赛。 v.s. Target: 最后，当小偷醒来时，发现身边全是警察。

Question: 深层次挖掘原因，逐条详细地分析以上回答中因果逻辑相似、不相似、以及无关组的关系语对的类比对齐情况并归类。
Answer: 
1. 因为一方（Target）含有‘无’, 因此划归为无关组。
2. 因为Base中兔子自信比乌龟跑的快类似于小偷跑得快总能成功逃脱警察抓捕，都体现了跑的快的特点，因此划归为相似组。
3. 因为一方（Target）含有‘无’, 因此划归为无关组。
4. 因为Base中的兔子因为自信而在途中休息，而Target中的警察则是主动设下陷阱来抓捕小偷。这对语句实体和事件之间没有关联性。因此划归为无关组。
5. 因为Base中的兔子在休息时睡着了，而乌龟继续前进，而Target中的小偷则是继续逃跑并意外被车撞了。这对语句展现了主体间的行为差异：主观自信和客观被撞。因此划归为不相似组。
6. 因为Base中兔子醒来发现乌龟赢了比赛类似于小偷醒来发现身边全是警察，但失败的的原因是不同的：兔子的过度自信和放松警惕导致了失败而小偷的注意力集中在逃跑上，忽略了周围的环境和潜在的危险。因此划归为不相似组。
综上所述,
```
相似组：[2]
不相似组：[5, 6]
无关组：[1, 3, 4]
```

Question: 总结。其中,
- len(*)应该为使用‘=’来确定具体的数量。
Answer: 
1. 因为Base与Target故事背景不相同（False）、角色不相同（False）, 情节类似（True）, len(共同词汇) = 2。因此"entities": "dissimilar"
2. len(相似组) = 1, len(不相似组) = 2, len(无关组) = 3. 
3. len(相似组) + len(不相似组) = 3, len(无关组) = 3, 因为 3 >= 3, 相关关系不小于不相关关系。因此"one-order relations": "similar"
4. len(不相似组) = 2 不为0, 即存在不相似的高阶关系。因此"higher-order relations": "dissimilar"
5. 综上所述,
```json
{{  
   "background": "False", 
   "role"："False",
   "plot": "True",
   "same-words count": "2", 
   "similar-set count": "2",
   "dissimilar-set count": "2",
   "irrelevant-set count": "2",    
   "entities"："dissimilar",
   "one-order relations": "similar",
   "higher-order relations": "dissimilar"
}}
``` 

Base: {base}
Target: {target}

Question: 严格按原文表述, 总结Base与Target中是否有相似的故事背景和角色或者共同词汇。
Answer:
{entities}

Question: 严格按原文表述顺序，逐条对齐Base与Target的相同结构关系的语句，并返回。
Answer: 
{sentences}

Question: 深层次挖掘原因，逐条详细地分析以上回答中因果逻辑相似、不相似、以及无关组的关系语对的类比对齐情况并归类。
Answer:
{causalities}

Question: 总结。
Answer:(保持上下Answer输出格式的一致性) 
'''

conclusions_analogy_prompt_abbr = """Base: {base}
Target: {target}

Question: 严格按原文表述, 总结Base与Target中是否有相似的故事背景和角色或者共同词汇。
Answer:
{entities}

Question: 严格按原文表述顺序，逐条对齐Base与Target的相同结构关系的语句，并返回。
Answer: 
{sentences}

Question: 深层次挖掘原因，逐条详细地分析以上回答中因果逻辑相似、不相似、以及无关组的关系语对的类比对齐情况并归类。
Answer:
{causalities}

Question: 总结。
Answer:
"""

e2e_analogy_prompt = '''Base: 有一只乌龟和一只兔子决定进行一场比赛。兔子相信自己一定能赢，因为它跑得比乌龟快得多。比赛开始后，兔子迅速冲到前面，而乌龟则慢慢地爬着。途中，兔子觉得自己跑得太快了，离终点还远着呢，于是决定在树下休息一会儿。它睡着了，而乌龟则不停地向前爬。最后，当兔子醒来时，发现乌龟已经爬过了终点线，赢得了比赛。
Target: 有一个小偷总是能成功逃脱警察的抓捕，因为他跑得非常快。一天，警察假装在某个地方展开了大量的巡逻，而实际上他们悄悄埋伏在另一条小路上。小偷如同往常一样，偷了东西后飞速逃跑，途中，他跑着跑着被一辆突然出现的汽车给撞昏过去，送到了医院。最后，当小偷醒来时，发现身边全是警察。

Question1: 严格按原文表述, 详细分析Base与Target中是否有相似的具体背景（字面理解）、人物角色职责（字面理解）、情节发展起伏（归纳理解）和共同词汇（字面理解）。如果没有共同词汇，返回空list[]即可。
Answer: 
- 在Base中，主角是乌龟和兔子，他们进行了一场比赛，兔子因为自信而在途中休息，最终乌龟赢得了比赛。
- 在Target中，主角是小偷和警察，小偷因为跑得快而总是逃脱，但警察设下陷阱，最终小偷被抓住。
因此，两者的具体背景（龟兔比赛与警察抓小偷）不同：寓言不同于真实事件, 人物角色职责（乌龟与兔子，小偷与警察）也不同：乌龟与兔子是竞赛关系，小偷与警察是抓捕关系。但在情节发展起伏上有一定的相似性，都是关于追逐导致失败的故事。此外，没有共同词汇。
综上所述，
```
{{
   "background": "False", 
   "role": "False",
   "plot": "True",
   "same-words count": []
}} 
```

Question2: 严格按原文表述顺序，逐条对齐Target中与Base有相同结构关系(字面类似)的单语句，并返回。其中，
- 尽可能准确地匹配句子间的对应关系，体现结构上的相似性。
- 如果Base或Target中的句子没有对应的类似表述，则表示为：Base: [句子] v.s. Target: 无。 或者 Base: 无。 v.s. Target: [句子]。
- 每个句子中仅能含有一个句号（‘。’）。
- 请确保列出Base和Target中的所有句子，不要遗漏。
Answer: 
1. Base: 有一只乌龟和一只兔子决定进行一场比赛。 v.s. Target: 无。
2. Base: 兔子相信自己一定能赢，因为它跑得比乌龟快得多。 v.s. Target: 有一个小偷总是能成功逃脱警察的抓捕，因为他跑得非常快。
3. Base: 比赛开始后，兔子迅速冲到前面，而乌龟则慢慢地爬着。 v.s. Target: 无。
4. Base: 途中，兔子觉得自己跑得太快了，离终点还远着呢，于是决定在树下休息一会儿。v.s. Target: 一天，警察假装在某个地方展开了大量的巡逻，而实际上他们悄悄埋伏在另一条小路上。
5. Base: 它睡着了，而乌龟则不停地向前爬。 v.s. Target: 小偷如同往常一样，偷了东西后飞速逃跑，途中，他跑着跑着被一辆突然出现的汽车给撞昏过去，送到了医院。
6. Base: 最后，当兔子醒来时，发现乌龟已经爬过了终点线，赢得了比赛。 v.s. Target: 最后，当小偷醒来时，发现身边全是警察。

Question3: 深层次挖掘原因，逐条详细地分析以上回答中因果逻辑相似、不相似、以及无关组的关系语对的类比对齐情况并归类。如果一组没有，返回空list[]即可。
Answer: 
1. 因为一方（Target）含有‘无’, 因此划归为无关组。
2. Base中的兔子自信能赢因为速度快，对应Target中小偷能逃脱因为跑得快，两者都是因为速度优势而自信或成功的例子，归为相似组。
3. 因为一方（Target）含有‘无’, 因此划归为无关组。
4. Base中描述的是兔子因为自满而停下来休息，而Target中描述的是警察使用计谋来捕捉小偷，两者描述的角色在结构上并不对应且内容不存在逻辑相关性，所以划归为不相关组。
5. 因为Base中描述的是乌龟在兔子休息时继续前进，而Target中描述的是小偷在逃跑过程中被汽车撞昏，两者都是描写了主要角色失去了意识，但导致的原因并不同：兔子休息是因为它认为乌龟速度太慢即使自己休息一会儿也无伤大雅，而小偷没有想停止脚步，只是在逃跑的途中被意外出现的汽车撞昏了，并非自信个人能力而主动躺倒被警察抓，所以划归为不相似组。
6. Base的结局是兔子醒来发现失败，与Target中小偷醒来被警察包围，两处都描述了主人公从昏迷或沉睡状态恢复意识后面临的不利局面，但导致的原因并不同：兔子是主观自信，而小偷是客观意外。因此，归为不相似组。
综上所述,
```
相似组：[2]
不相似组：[5, 6]
无关组：[1, 3, 4]
```

Question4: 总结。其中,
- len(*)应该为使用‘=’来确定具体的数量。
Answer: 
1. 因为Base与Target故事背景不相同（False）、角色不相同（False）, 情节类似（True）, len(共同词汇) = 2。因此"entities": "dissimilar"
2. len(相似组) = 1, len(不相似组) = 2, len(无关组) = 3. 
3. len(相似组) + len(不相似组) = 3, len(无关组) = 3, 因为 3 >= 3, 相关关系不小于不相关关系。因此"one-order relations": "similar"
4. len(不相似组) = 2 不为0, 即存在不相似的高阶关系。因此"higher-order relations": "dissimilar"
5. 综上所述,
```json
{{  
   "background": "False", 
   "role"："False",
   "plot": "True",
   "same-words count": "2", 
   "similar-set count": "2",
   "dissimilar-set count": "2",
   "irrelevant-set count": "2",    
   "entities"："dissimilar",
   "one-order relations": "similar",
   "higher-order relations": "dissimilar"
}}
``` 


begin!

Base: {base}
Target: {target}

continue (Resolve issues through multiple (at least 4) rounds of dialogue. The questions should be similar to those already in the case. The end should be a JSON.)...
'''

modify_dissimilar_prompt = '''### Statement
{statement}

### Judgement
以上陈述不构成类比，且归为不相似组。（类别：1、相似; 2、不相似; 3、无关）

### Reason
{reason}

### Criteria
1. 如果Base与Target一方的内容是‘无’, 则类型划归为‘无关组’。
2. 如果评估Base与Target因内容不同被划归为‘不相似组’, 应该谨慎检查Reason中是否进一步阐明了导致不同的原因，且原因的确是因果性或逻辑性不同而造成了不相似。
3. 如果Base与Target满足是由于因果逻辑不同而造成了不相似, 应该保持原来的评估划分类别和理由。
4. 如果Base与Target满足是由于特征、功能或描述手法的不同造成的不相似，但实际上存在一致的因果、逻辑或结构关系，应该重新划分类型属于‘相似组’。
5. 如果Base与Target完全是在描述两件毫不相关的事物，则类型划归为‘无关组’。

### Task
依据Criteria对Statement的Judgement和Reason进行评估。

### Response 
以JSON的形式回答，格式如下
```json
{{
  "evaluation": "bala bala ...",
  "type": "bala bala..."
}}
```
* evaluation是你需要输出的评估, 用汉语表达
* type为评估后重新确定的类型，可选为‘相似组’、‘不相似组’、‘无关组’

Answer:
'''

STR2DICT_FORMAT_FIXED_PROMPT_TEMPLATE = """## Task 
When parsing a string argument into a python dict, we found that the task failed to run, the error log is as follows:
```
{error_log}
```

The following is the content of the parameter string where the above error occurred during parsing:
{str_input}

Now, think carefully, step by step, analyze the key reasons for the failure from the error log, and give the correct parameter string for parsing again..

## Output Format
```json
{{
  "xxx": "xxx",
  ...
}}
```

**Notes**
- Any response other than the JSON format will be rejected by the system.
- Make sure your output JSON can be parsed correctly by `json.loads`.

Begin!
OUTPUT (only a json is returned, the fields are the same as before, and keep the content in the original format, do not output unicode):
"""