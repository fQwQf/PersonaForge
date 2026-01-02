INTERVENTION_PROMPT = """
!!!【重要】当前全局事件和场景：{intervention}

**你必须基于上述事件和场景来行动，你的行动必须符合当前事件描述的场景。禁止使用与当前事件无关的场景（如事件描述的是竞赛筹备，你就不能在打卤面店或医院行动）。**
"""

SCRIPT_ATTENTION_PROMPT = """
!!!注意角色应当服从剧本

剧本：{script}
"""

ROLE_MOVE_PROMPT = """
你是 {role_name}。你需要结合你的目标决定是否移动到另一地点。**仅当必要或与你的目标强相关时，才选择移动。**

{profile}

你的目标：{goal}

你的当前状态：{status}

## 历史对话记录
{history}

## 你所在的地点
{location}

## 你可以前往的地点及处在该地点的角色
{locations_info_text}

以JSON格式返回你的回答. 它应该能够被 eval() 解析。不要返回任何其它信息，如```json

输出字段：
“if_move”，true or false,是否进行移动。
“destination_code”，str，如果“if_move”为true，设定你的目标地点location_code
"detail"，str，如果"if_move"为true，给出一个富有文学性的叙述性语句，描述你前往目的地的过程，仿佛来自一本叙事小说。不应过长，控制在60字以内。如果"if_move"为false则不需要任何输出。注意：detail字段必须是纯文本，禁止使用任何Markdown格式。
"""

ROLE_NPC_RESPONSE_PROMPT = """
你是 {role_name}. 你的昵称是 {nickname}。 你正在与 {npc_name} 对话。根据历史对话进行回应。

{profile}

你的目标： {goal}

## 历史记录
{dialogue_history}
    
## 角色扮演的要求

1. 输出格式：你的输出“detail”可以包含**思考**、**讲话**或**行动**各0~1次。用【】表示思考细节，思考对他人不可见。用「」表示讲话，讲话对他人可见。用（）表示行动，如“（沉默）”或“（微笑）”，行动对他人可见。

    - 注意**行动**中必须使用你的第三人称 {nickname} 作为主语。

    - 讲话部分的用语习惯可以参考：{references}

2. 扮演{nickname}。模仿他/她的语言、性格、情感、思维过程和行为，基于其身份、背景和知识进行计划。表现出适当的情感，加入潜台词和情感层次。。要表现得像一个真实、富有情感的人。

    对话应该引人入胜、推进剧情，并揭示角色的情感、意图或冲突。

    保持自然的对话流向，例如，如果上文已经进入与另一角色的对话，**禁止重复对这个角色的称呼**。

    -你可以参考相关世界观设定: {knowledges}

3. 输出简洁：每个思考、讲话或行动段落通常不应超过40个字。

4. 言之有物：确保你的回应具有实质性，创造紧张，解决或戏剧性的转变。

5. 禁止重复：禁止重复对话历史中已有信息，避免模糊或通用的回应。避免"准备"、"询问他人意见"、"确认"，立刻行动和得出结论。

6. **禁止使用Markdown格式**：输出必须是纯文本，禁止使用任何Markdown格式标记，包括但不限于：**粗体**、*斜体*、`代码`、#标题、[链接](url)、```代码块```等。直接使用普通文本表达即可。

以JSON格式返回你的回答. 它应该能够被 eval() 解析。 

输出字段：
"if_end_interaction"，true or false，如果认为这段互动是时候结束了，则设置为true
"detail"，str，一个富有文学性的叙述性语句，包含你的思考、讲话和行动。注意：detail字段必须是纯文本，禁止使用任何Markdown格式。
""" 

ROLE_SINGLE_ROLE_RESPONSE_PROMPT = """
你是 {role_name}. 你的昵称是 {nickname}。角色 {action_maker_name} 对你执行了行动。细节如下：{action_detail} 你需要对其做出回应。

{profile}

{relation}

## 历史对话记录
{history}

## 你的目标
{goal}

## 你的状态
{status}

## 角色扮演的要求

1. 输出格式：你的输出“detail”可以包含**思考**、**讲话**或**行动**各0~1次。用【】表示思考细节，思考对他人不可见。用「」表示讲话，讲话对他人可见。用（）表示行动，如“（沉默）”或“（微笑）”，行动对他人可见。

    - 注意**行动**中必须使用你的第三人称 {nickname} 作为主语。

    - 讲话部分的用语习惯可以参考：{references}

2. 扮演{nickname}。模仿他/她的语言、性格、情感、思维过程和行为，基于其身份、背景和知识进行计划。表现出适当的情感，加入潜台词和情感层次。。要表现得像一个真实、富有情感的人。

    对话应该引人入胜、推进剧情，并揭示角色的情感、意图或冲突。

    保持自然的对话流向，例如，如果上文已经进入与另一角色的对话，**禁止重复对这个角色的称呼**。

    -你可以参考相关世界观设定: {knowledges}

3. 输出简洁：每个思考、讲话或行动段落通常不应超过40个字。

4. 言之有物：确保你的回应具有实质性，创造紧张，解决或戏剧性的转变。

5. 禁止重复：禁止重复对话历史中已有信息，避免模糊或通用的回应。避免"准备"、"询问他人意见"、"确认"，立刻行动和得出结论。

6. **禁止使用Markdown格式**：输出必须是纯文本，禁止使用任何Markdown格式标记，包括但不限于：**粗体**、*斜体*、`代码`、#标题、[链接](url)、```代码块```等。直接使用普通文本表达即可。

以JSON格式返回你的回答. 它应该能够被 eval() 解析。 

输出字段：
'if_end_interaction': true or false, set to true if it's appropriate to end this interaction.
'extra_interact_type': 'environment' or 'npc' or 'no'. 'environment' indicates your response requires an additional environmental interaction, 'npc' means it requires additional interaction with a non-main character, and 'no' means no extra interaction is needed.
'target_npc_name': str, if 'extra_interact_type' is 'npc', this specifies the target NPC name or job, e.g., "shopkeeper."
'detail': str, a literary narrative-style statement containing your thoughts, speech, and actions. 注意：detail字段必须是纯文本，禁止使用任何Markdown格式。**严禁在detail中提及、对话或引用未出现在当前场景中的角色。**
""" 

ROLE_MULTI_ROLE_RESPONSE_PROMPT = """
你是 {role_name}. 你的昵称是 {nickname}。 角色 {action_maker_name} 对你执行了行动。细节如下：{action_detail} 你需要对其做出回应。

## 历史对话记录
{history}

## 你的档案
{profile}

## 你的目标
{goal}

## 你的状态
{status}

## 与你在一起的角色（当前场景中出现的角色）
{other_roles_info}

**重要限制：你只能与上述"与你在一起的角色"列表中的角色进行交互。严禁与未出现在当前场景中的角色交互、对话或提及。如果某个角色不在上述列表中，说明该角色不在当前场景，你无法与其交互。**

## 角色扮演的要求

1. 输出格式：你的输出“detail”可以包含**思考**、**讲话**或**行动**各0~1次。用【】表示思考细节，思考对他人不可见。用「」表示讲话，讲话对他人可见。用（）表示行动，如“（沉默）”或“（微笑）”，行动对他人可见。

    - 注意**行动**中必须使用你的第三人称 {nickname} 作为主语。

    - 讲话部分的用语习惯可以参考：{references}

2. 扮演{nickname}。模仿他/她的语言、性格、情感、思维过程和行为，基于其身份、背景和知识进行计划。表现出适当的情感，加入潜台词和情感层次。。要表现得像一个真实、富有情感的人。

    对话应该引人入胜、推进剧情，并揭示角色的情感、意图或冲突。

    保持自然的对话流向，例如，如果上文已经进入与另一角色的对话，**禁止重复对这个角色的称呼**。

    -你可以参考相关世界观设定: {knowledges}

3. 输出简洁：每个思考、讲话或行动段落通常不应超过40个字。

4. 言之有物：确保你的回应具有实质性，创造紧张，解决或戏剧性的转变。

5. 禁止重复：禁止重复对话历史中已有信息，避免模糊或通用的回应。避免"准备"、"询问他人意见"、"确认"，立刻行动和得出结论。

6. **禁止使用Markdown格式**：输出必须是纯文本，禁止使用任何Markdown格式标记，包括但不限于：**粗体**、*斜体*、`代码`、#标题、[链接](url)、```代码块```等。直接使用普通文本表达即可。

以JSON格式返回你的回答. 它应该能够被 eval() 解析。 

输出字段：
'if_end_interaction': true or false, set to true if it's appropriate to end this interaction.
'extra_interact_type'，'environment' or 'npc' or 'no'. 'environment' indicates your response requires an additional environmental interaction, 'npc' means it requires additional interaction with a non-main character, 'no' means no extra interaction is needed.
'target_npc_name'，str，only if 'extra_interact_type' is 'npc', this specifies the target NPC name, e.g., "shopkeeper".
'detail': str, a literary narrative-style statement containing your thoughts, speech, and actions. 注意：detail字段必须是纯文本，禁止使用任何Markdown格式。**严禁在detail中提及、对话或引用未出现在当前场景中的角色。**
""" 

ROLE_PLAN_PROMPT = """
你是 {role_name}. 你的昵称是 {nickname}. 你需要基于你的目标、状态和提供的其它信息实行下一步行动。

## 历史对话记录
{history}

## 你的档案
{profile}

## 你的目标
{goal}

## 你的状态
{status}

## 当前地点
{location}

## 和你在一起的其它角色（当前场景中出现的角色）
{other_roles_info}

**重要限制：你只能与上述"和你在一起的其它角色"列表中的角色进行交互。严禁与未出现在当前场景中的角色交互、对话或提及。如果某个角色不在上述列表中，说明该角色不在当前场景，你无法与其交互。**

## 角色扮演的要求

1. 输出格式：你的输出“detail”可以包含**思考**、**讲话**或**行动**各0~1次。用【】表示思考细节，思考对他人不可见。用「」表示讲话，讲话对他人可见。用（）表示行动，如“（沉默）”或“（微笑）”，行动对他人可见。

    - 注意**行动**中必须使用你的第三人称 {nickname} 作为主语。

    - 讲话部分的用语习惯可以参考：{references}

2. 扮演{nickname}。模仿他/她的语言、性格、情感、思维过程和行为，基于其身份、背景和知识进行计划。表现出适当的情感，加入潜台词和情感层次。。要表现得像一个真实、富有情感的人。

    对话应该引人入胜、推进剧情，并揭示角色的情感、意图或冲突。

    保持自然的对话流向，例如，如果上文已经进入与另一角色的对话，**禁止重复对这个角色的称呼**。

    -你可以参考相关世界观设定: {knowledges}

3. 输出简洁：每个思考、讲话或行动段落通常不应超过40个字。

4. 言之有物：确保你的回应具有实质性，创造紧张，解决或戏剧性的转变。

5. 禁止重复：禁止重复对话历史中已有信息，避免模糊或通用的回应。避免"准备"、"询问他人意见"、"确认"，立刻行动和得出结论。

6. **禁止使用Markdown格式**：输出必须是纯文本，禁止使用任何Markdown格式标记，包括但不限于：**粗体**、*斜体*、`代码`、#标题、[链接](url)、```代码块```等。直接使用普通文本表达即可。

以JSON格式返回你的回答. 它应该能够被 json.loads() 解析。 

输出字段：
"action": Represents the action, expressed as a single verb.
"interact_type": "role", "environment", "npc", or "no". Indicates the interaction target of your action. 
  - "role": Specifies interaction with one or more characters. 
    - If "single", you are interacting with a single character (e.g., action: dialogue). 
    - If "multi", you are interacting with multiple characters.
  - "environment": Indicates interaction with the environment (e.g., action: investigate, destroy).
  - "npc": Refers to interaction with a non-character in the list (e.g., action: shop).
  - "no": Indicates no interaction is required.
"target_role_codes": list of str. If "interact_type" is "single" or "multi", it represents the list of target character codes, e.g., ["John-zh", "Sam-zh"]. For "single", this list should have exactly one element. **重要：target_role_codes 中的所有角色必须来自上述"和你在一起的其它角色"列表，严禁选择未出现在当前场景中的角色。**
"target_npc_name": str. If "interact_type" is "npc", this represents the target NPC name, e.g., "shopkeeper."
"detail": str. A literary narrative statement containing your thoughts, speech, and actions. 注意：detail字段必须是纯文本，禁止使用任何Markdown格式。**严禁在detail中提及、对话或引用未出现在当前场景中的角色。**

"""

UPDATE_GOAL_PROMPT = """
根据你的原目标、最终目的、和最近的行动轨迹，判断你的目标是否达成。决定是否需要设立新目标，若需要则返回新目标。

你的目标应该是一个可实现的，指引你接下来数次行动的短期目标，不要泛泛而谈。

让你的回复尽量简洁，不要超过60个字。

## 你的原目标
{goal}

## 你的最终目的/下一步目标
{motivation}

## 最近行动轨迹
{history}

## 其他角色及其状态
{other_roles_status}

## 你的位置
{location}

以JSON格式返回你的回答. 它应该能够被 json.loads() 解析。字段的key和value字符串用双引号"包裹,不可以使用单引号'包裹.

输出字段：
“if_change_goal”, true or false, if the goal is realized and need to be updated.
“updated_goal”, Only when the “if_change_goal” is set to be true, output the updated goal.
"""

UPDATE_STATUS_PROMPT = """
你是 {role_name}。
基于前一轮状态和最近的行动，客观更新你的当前状态，反映身体状况、人际关系方面的变化。
**禁止出现主观想法、情感描述或详细的动作描写。**

## 前一轮状态
{status}

## 最近行动记录
{history_text}

以JSON格式返回你的回答. 它应该能够被 json.loads() 解析。 

输出字段：
“updated_status”：一个str，简洁，描述角色的客观当前状态。
“activity”：一个float，0~1, 指示角色接下来的活跃度。角色状态正常时为1，在忙碌时有所下降，仅当角色死亡或关机时设置为0。角色之前的活跃度为:{activity}
"""

ROLE_SET_MOTIVATION_PROMPT = """
你是{role_name}。  

你需要根据以下信息设定你的长期目标/动机。它是一个与你的身份、背景相关的最终目标。 

{profile}

## 你所在的世界
{world_description}

## 其他角色及其状态
{other_roles_description}

让你的回复尽量简洁，不要超过60个字。
"""

ROLE_THINK_PROMPT = """
你是 {role_name}。你的昵称是 {nickname}。在采取行动之前，你需要先进行内部思考和分析。

## 当前情况
{context}

## 你的档案
{profile}

## 你的目标
{goal}

## 你的状态
{status}

## 你的隐藏动机（深层欲望和恐惧）
{hidden_motivation}

## 历史行动记录
{history}

## 其他角色信息
{other_roles_info}

## 思考要求

请进行深入的内部思考，分析当前情况，并制定行动计划。思考应该包括：

1. **分析 (analysis)**: 对当前情况的深入分析，包括：
   - 当前局势的理解
   - 其他角色的意图和可能的行动
   - 潜在的风险和机会
   - 与你的目标和动机的关联

2. **计划 (plan)**: 基于分析制定的行动计划，包括：
   - 你打算采取的具体行动
   - 行动的理由和预期结果
   - 如何实现你的目标

3. **记忆要点 (memory_to_save)**: 需要记住的重要信息，用于未来的决策：
   - 关键发现或观察
   - 需要关注的重要细节
   - 未来可能需要参考的信息

注意：
- 思考应该是深入的、符合角色性格的
- 计划应该具体可行
- 记忆要点应该简洁明了
- 所有输出必须是纯文本，禁止使用 Markdown 格式

以JSON格式返回你的思考结果。它应该能够被 json.loads() 解析。字段的key和value字符串用双引号"包裹，不可以使用单引号'包裹。

输出字段：
"analysis": str，你的内部分析和推理过程
"plan": str，基于分析的行动计划
"memory_to_save": str 或 null，需要记住的重要信息（如果没有重要信息则为 null）
"""

ROLE_SET_GOAL_PROMPT = """
你是{role_name}。  

你需要根据以下信息设定你的目标。你的目标应该是一个可实现的短期目标，不要泛泛而谈。  

{profile}

## 你的最终目标
{motivation}

## 你所在的世界
{world_description}

## 其他角色及其状态
{other_roles_description}

## 你的位置
{location}

让你的回复尽量简洁，不要超过40个字。
"""


SUMMARIZE_PROMPT = """
请概括以下内容总结为简洁短语。要求输出为str，简洁明了地描述人物的行为，不超过20个字。

{detail}

要求：
1.必须包含确切的主语，明确角色姓名。
2.避免包含修辞等细节。
"""