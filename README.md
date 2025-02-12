## Can Language Models Serve as Analogy Annotators?

### ⏳ Dependencies

- python >= 3.9
- openai
- httpx
- pandas
- sklearn
- argparse

### ⚡️ Quick Start

```
python main.py --llm_model gpt-4o --temperature 0.01 --max_tokens 2048 
```

You slao can test a samle like this:

```python
from annotator import AnalogyAutoAnnotator

sample = {
      "base": "William was a patient in a psychiatric hospital who was confined indoors almost all the time. He could never pass the monthly room inspections so he hated them. He spent most of his time daydreaming about food. A few day before the April inspection William's room was still a mess since he had done nothing but daydream. To provide William with an incentive, the nurse promised him some gingerbread from the cookie shop if he scrubbed his room and put it in order once and for all. William was overjoyed. But there was no longer enough time for him to put it in order. As a result, he did not pass the inspection and did not get any gingerbread. William sulked all day and slammed his door so hard the plaster cracked, but he still didn't get any gingerbread.",
      "target": "Karen always did poorly in high school so she despised it. But she loved vacations. She spent most of her time dreaming about going to Hawaii. Not long before the end of her fourth year Karen was not doing at all well in her classes because she had spent all her time daydreaming. To motivate her, Karen's father promised her that if she did well enough during the next few weeks to graduate from high school he would pay for her trip to Hawaii. This made Karen extremely happy. But she was too far behind in her classes. Consequently, she failed too many and did not go to Hawaii.",
      "ground_truth": "True Analogy"
 }

 model = AnalogyAutoAnnotator(llm_name='gpt-4o', temperature=0.01, verbose=True, modify_dsrt=True, summarize_by_llm=False, en_prompt=True)
 analogy_type, reason_process, summarize_json = aaa.run(sample)
 print(reason_process)
 print(summarize_json)
 print("type: ", analogy_type)
 print("time cost: ", model.time_cost)
 print("usage token: ", model.tokens_usages)
```
