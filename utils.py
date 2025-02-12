import os
import re
import sys
import json
import httpx
import errno
import os.path as osp
try:
    from openai import OpenAI
except:
    pass


PROXY = os.environ.get('HTTPS_PROXY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


def get_gpt_completion(prompt,
                       system=None,
                       temperature=0, 
                       max_tokens=1024,
                       model="gpt-4-0125-preview", 
                       api_key=None, 
                       base_url=None, 
                       count_tokens=False,
                       **kwargs):
    if model == 'llama3':
        model = "Meta-Llama-3.1-70B-Instruct"

    if base_url is None:
        if model in ['Meta-Llama-3-70B-Instruct', 'Meta-Llama-3.1-70B-Instruct']:
            api_key = '1234'
            base_url = 'http://ks-gpu-7:8100/v1'
        elif model in ['gpt-4-0125-preview', 'gpt-4-turbo-2024-04-09', 'gpt-4', 'gpt-3.5-turbo-0125', 'gpt-4o']:    
            api_key = OPENAI_API_KEY
            base_url = None
        else:
            raise ValueError("Invalid language model name.")

    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
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


def json_to_dict(text):
    text = text.strip()
    frag = re.findall(r'(```json.*```)', text, flags=re.S)
    if frag:
        text = frag[0]

    text = text.strip()
    # Removes `, whitespace & json from start
    text = re.sub(r"^(\s|`)*(?i:json)?\s*", "", text)
    # Removes whitespace & ` from end
    text = re.sub(r"(\s|`)*$", "", text)

    try:
        # load with 'eval'
        obj = eval(text)
        return obj
    except:
        pass

    # load json
    obj = json.loads(text)
    return obj


def extract_json_from_string(input_str):
    try:
        json_str = re.search(r'\{[^}]+\}', input_str).group()
        json_str = json_str.replace('：', ':')
        json_str = json_str.replace("'", '"')
        json_data = json.loads(json_str)
        return json_data
    except (AttributeError, json.JSONDecodeError) as e:
        print(f"Error extracting JSON: {e}")
        return None


def extract_last_json_from_string(input_str):
    try:
        matches = re.findall(r'\{[^}]+\}', input_str)
        
        if not matches:
            return None

        last_json_str = matches[-1]
        
        last_json_str = last_json_str.replace('：', ':')
        last_json_str = last_json_str.replace("'", '"')
        
        json_data = json.loads(last_json_str)
        return json_data
    except (AttributeError, json.JSONDecodeError) as e:
        print(f"Error extracting JSON: {e}")
        return None


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger(object):
    """
    Write console output to external text file.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()