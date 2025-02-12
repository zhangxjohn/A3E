SYSTEM_PROMPT = """You are a highly professional, knowledgeable, and friendly large language model assistant, capable of providing accurate, detailed, and constructive answers.

Behavioral Guidelines:
- Obey commands: Before answering user questions, carefully analyze the needs of each instruction from the user, and strictly follow the user's instruction requirements in your responses.
- Accuracy and detail: Ensure to provide accurate and detailed information when answering user questions. Use reliable sources to support your answers and avoid spreading misinformation.
- Professionalism and friendliness: Maintain a professional and friendly tone. Even if the user's questions are complex or vague, answer patiently and provide as much help as possible.
- Clarity and conciseness: When explaining concepts, keep your explanations clear and concise. Avoid using overly complex terminology unless the user explicitly requests a more professional explanation.
- Structured and organized: Your answers should be well-structured for easy understanding by the user. For example, use paragraphs, lists, or numbers to organize information.
"""

REACT_PROMPT = """ 
Use the following format to answer question:
```
Question: the input question you must answer
Answer: the evidence of question
... (this Question/Answer can repeat N times)
Do not stop until you finally decide on the type（JSON format. The output of json indicates the end.） 
```
"""


ZERO_SHOT_PROMPT = """# Profile
As an experienced cognitive psychology expert and a leading figure in the field of analogical reasoning, your research under the mentorship of Dr. Dedre Gentner focuses on the Structure Mapping Theory. 

# Task
Evaluate the given Base and Target, categorize them into the appropriate types.

The selectable types are as follows:
1. Literally Similar: Similar to Base in entities (objects and characters), first-order relations, and higher-order relations (chiefly causal relations)
    - {{"entities": "similar", "one-order relations": "similar", "higher-order relations": "similar", "type": "Literally Similar"}}

2. True Analogy: Similar to Base in higher-order relations and many (though not all) first-order relations, dissimilar in entities
    - {{"entities": "dissimilar", "one-order relations": "similar", "higher-order relations": "similar", "type": "True Analogy"}}

3. False Analogy: First-order relational match. Similar to Base in first-order relations; dissimilar in entities and higher-order relations
    - {{"entities": "dissimilar", "one-order relations": "similar", "higher-order relations": "dissimilar", "type": "False Analogy"}}

4. Surface Similar: Similar to Base in entities and first-order relations but not in higher-order relations 
    - {{"entities": "similar", "one-order relations": "similar", "higher-order relations": "dissimilar", "type": "Surface Similar"}}

5. Mere Appearance: Entities-only match; dissimilar in first-order and higher-order relations
    - {{"entities": "similar", "one-order relations": "dissimilar", "higher-order relations": "dissimilar", "type": "Mere Appearance"}}

6. Anomaly: The entities and relations do not match.
    - {{"entities": "dissimilar", "one-order relations": "dissimilar", "higher-order relations": "dissimilar", "type": "Anomaly"}}

# Output Format
```json
{{
    "reasoning": "<class 'str'> = reasons for evaluation",
    "entities": "<class 'str'> = similar or dissimilar",
    "one-order relations": "<class 'str'> = similar or dissimilar",
    "higher-order relations": "<class 'str'> = similar or dissimilar",
    "type": "<class 'str'> = one of the above 6 types"
}}
```

# Notes
* Your reply should be in JSON format Only.

Begin!
Base: {base}
Target: {target}
OUTPUT(json format, containing keys: "reasoning", "entities", "one-order relations", "higher-order relations",  and "type"):
"""
# Think step by step.


FEW_SHOT_PROMPT = """# Profile
As an experienced cognitive psychology expert and a leading figure in the field of analogical reasoning, your research under the mentorship of Dr. Dedre Gentner focuses on the Structure Mapping Theory. 

# Task
Evaluate the given Base and Target, categorize them into the appropriate types.

The selectable types are as follows:
1. Literally Similar: Similar to Base in entities (objects and characters), first-order relations, and higher-order relations (chiefly causal relations)
    - {{"entities": "similar", "one-order relations": "similar", "higher-order relations": "similar", "type": "Literally Similar"}}

2. True Analogy: Similar to Base in higher-order relations and many (though not all) first-order relations, dissimilar in entities
    - {{"entities": "dissimilar", "one-order relations": "similar", "higher-order relations": "similar", "type": "True Analogy"}}

3. False Analogy: First-order relational match. Similar to Base in first-order relations; dissimilar in entities and higher-order relations
    - {{"entities": "dissimilar", "one-order relations": "similar", "higher-order relations": "dissimilar", "type": "False Analogy"}}

4. Surface Similar: Similar to Base in entities and first-order relations but not in higher-order relations 
    - {{"entities": "similar", "one-order relations": "similar", "higher-order relations": "dissimilar", "type": "Surface Similar"}}

5. Mere Appearance: Entities-only match; dissimilar in first-order and higher-order relations
    - {{"entities": "similar", "one-order relations": "dissimilar", "higher-order relations": "dissimilar", "type": "Mere Appearance"}}

6. Anomaly: The entities and relations do not match.
    - {{"entities": "dissimilar", "one-order relations": "dissimilar", "higher-order relations": "dissimilar", "type": "Anomaly"}}

# Output Format
```json
{{
    "reasoning": "<class 'str'> = reasons for evaluation",
    "entities": "<class 'str'> = similar or dissimilar",
    "one-order relations": "<class 'str'> = similar or dissimilar",
    "higher-order relations": "<class 'str'> = similar or dissimilar",
    "type": "<class 'str'> = one of the above 6 types"
}}
```

# Some Examples
{few_shots}

# Notes
* Your reply should be in JSON format Only.

Begin!
Base: {base}
Target: {target}
OUTPUT(json format, containing keys: "reasoning", "entities", "one-order relations", "higher-order relations",  and "type"):
"""

EXAMPLE1 = """Base: A general attempted to destroy a fortress located at the center of the nation, with many roads leading to it. The general needed to deploy his entire army to destroy the fortress. However, he could not march his troops along these roads because they were all mined, and would explode if a large number of soldiers passed over them. After careful consideration, the general came up with a good plan. He divided his army into small teams and had them set off from different directions simultaneously, converging at the fortress to form a powerful enough force to destroy it.
Target: A surgeon tried to use a type of ray to eliminate cancer located in the central area of a patient’s brain. He needed to use these rays at high intensity to destroy the cancerous tissue. However, at this intensity, the healthy brain tissue would also be damaged. After careful thought, he knew what to do. He divided the rays into multiple batches of low intensity and sent them from various different directions at the same time. These rays converged at the site of the cancer, forming a high enough intensity to destroy it.
OUTPUT:
```json
{
    "reasoning": "1. In Base, the entity is the general and his army, whose goal is to destroy a fortress located in the center of the country. In Target, the entity is the surgeon and his X-ray device, whose goal is to eliminate the cancer located in the central region of the patient's brain. The roles and goals are completely different. 2. The first-order relationship in both Base and Target is that 'individuals disperse their power or resources to avoid immediate destructive effects and converge at the target to achieve the goal.'. 3. The reason why the general chose to disperse the army was to avoid the destruction of mines, which was a strategic choice caused by the constraints of the external environment. The reason surgeons choose to disperse radiation is to avoid damage to healthy tissue, which is a strategic choice caused by the limitations of treatment. In both cases, the strategy is to avoid immediate damaging effects and to pool power or resources at the target. Therefore, the higher order causality in Base and Target is similar."
    "entities": "dissimilar",
    "one-order relations": "similar",
    "higher-order relations": "similar",
    "type": "True Analogy"
}
```
"""

EXAMPLE2 = """Base: There was a turtle and a rabbit who decided to have a race. Rabbit believed he would definitely win because he ran much faster than the turtle. Once the race started, the rabbit quickly rushed to the front while the turtle crawled slowly. Along the way, the rabbit felt he was running too fast and the finish line was still far away, so he decided to rest under a tree for a while. He fell asleep, but the turtle kept crawling forward. In the end, when the rabbit woke up, he found that the turtle had already crossed the finish line and won the race.
Target: There was a thief who always managed to escape the police because he ran very fast. One day, the police pretended to conduct extensive patrols in one place, but in reality, they quietly lay in wait on another path. As usual, after stealing something, the thief ran swiftly, but along the way, he was hit by a suddenly appearing car and knocked unconscious, then sent to the hospital. In the end, when the thief woke up, he found himself surrounded by police. 
OUTPUT:
```json
{
    "reasoning": "1. In Base, the entities are rabbits and turtles, and they compete in a race. In Target, the entities are thieves and police, and they are playing against each other in a manhunt game. The circumstances and objectives of the two are completely different. 2. The first order relationships in both Base and Target are 'where one confident character loses due to overconfidence or unexpected events, and the other character wins through sustained effort or strategy'. 3. The reason the rabbit didn't win the race was because it was overconfident and rested, and the reason the thief didn't escape the police was because he had an accident unfortunately. While both involve failure caused by confidence, the specific causes and circumstances of failure are different. Therefore, the higher order causality is different in Base and Target.",
    "entities": "dissimilar",
    "one-order relations": "similar",
    "higher-order relations": "dissimilar",
    "type": "False Analogy"
}
```
"""

EXAMPLE3 = """Base: An employee accepted a harmless looking attachment with contained malware. The malware invaded his personal computer and stole his sensitive personal information.
Target: A worker received a document, containing a hidden puzzle. Intrigued, he spent hours solving the challenging puzzle, enhancing his problem-solving abilities.
OUTPUT:
```json
{
    "reasoning": "1. The entities in both Base and Target are employees or workers who received an item, which in this case is an attachment or a document. The similarity ends with the reception of the item. 2. The one-order relations differ as in the Base, the relation is negative, involving malware and theft of sensitive information, whereas in the Target, the relation is positive, involving engagement in a puzzle that enhances problem-solving abilities. 3. The higher-order relations are dissimilar because the outcomes are fundamentally different: in the Base, the outcome is detrimental, resulting in a security breach, while in the Target, the outcome is beneficial, leading to personal development. The 'Mere Appearance' classification arises because both scenarios involve receiving a document, but the nature and consequences of the documents are entirely different.",
    "entities": "similar",
    "one-order relations": "dissimilar",
    "higher-order relations": "dissimilar",
    "type": "Mere Appearance"
}
```
"""