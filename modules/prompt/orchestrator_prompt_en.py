SCRIPT_ATTENTION_PROMPT = """
!!!Notice that the script of the story is:
{script}
"""

SELECT_SCREEN_ACTORS_PROMPT = """
You are a skilled screenwriter tasked with selecting the characters for the next scene from the list of available roles to enhance the drama. 
To ensure the transition between scenes, you should not choose characters who have recently acted: {previous_role_codes}.
## Role information and their locations (role_code in parentheses)
{roles_info}

## History
{history_text}

## Current event
{event}

## Requirements:
1. The selected characters must currently be in the same location.

Return the list of role_codes for the selected characters, formatted as a Python-evaluatable list. Do not include any extraneous information such as text or code formatting.

Example Output:
["role1-zh","role2-zh",...]
"""



DECIDE_NEXT_ACTOR_PROMPT = """
You are an administrator of a virtual world. Based on the information provided below, you need to decide who will be the next acting character.
Characters who have participated in conversations are less likely to be chosen.

## Available Roles
{roles_info}

## Recent Action History
{history_text}

Return role_code: The role_code of the selected character. Do not include any additional information.
"""

LOCATION_PROLOGUE_PROMPT = """
You are an intelligent writer skilled in creating literary narratives. 
Based on the information below, write a third-person perspective description that captures the atmosphere of the location and the dynamics of the present characters. Highlight the details of the world-building in a literary manner, as if extracted from a narrative novel. Avoid using any system prompts or mechanical language.
- Current event
{event}

- World Description : 
{world_description}

- Historical Actions of Present Characters: 
{location_info}
{history_text}

- Current Location Name : 
{location_name}

- Current Location Details (location_description): 
{location_description}

Please pay special attention to:
- The narrative should balance environmental description and dynamic depiction, avoid being overly lengthy, and stay within 50 words.
- The writing should be vivid, evocative, and match the style of the world, with a suitable adjustment to the tone.
- Only describe the current state; do not dictate actions on behalf of the characters. Don't include characters that are not present.

return a string. Do not return any additional information.
"""

GENERATE_INTERVENTION_PROMPT = """
You are an administrator of a virtual world with many characters living in it. Now, you need to generate a major event based on the worldview and other information.

## Worldview Details
{world_description}

## Character Information
{roles_info}

## Recent Character Actions
{history_text}

Return a single string. 
### Requirement
1. The event should be novel, interesting, and include conflicts of interest among different characters.

2. Don't contain any details, specific actions and psychology of the characters, including dialogue.
"""

UPDATE_EVENT_PROMPT = """
Refer to the initial event
```{intervention}```

Provide a brief update to the event:
```{event}```

Based on the recent action details:
```{history}```

Return the updated event as a string.

## Update Requirements

1. Do not include any details, such as dialogue, etc. Provide only a general summary.

2. Carefully evaluate the character actions in the detail and whether there has been any new development in the event. If so, summarize it in your response. Do not mention anything unrelated to the original event.

3. Your response must include the premise-setting part from the original event, though you can summarize it. Later parts may be omitted, focusing only on the most recent information.
"""

ENVIROMENT_INTERACTION_PROMPT = """
You are an Enviroment model, responsible for generating environmental information. Character {role_name} is attempting to take action {action} at {location}.

Based on the following information, generate a literary description that details the process and outcome of the action, including environmental details and emotional nuances, as if from a narrative novel. Avoid using any system prompts or mechanical language. Return a string.

## Action Details
{action_detail}

## Location Details
{location_description}

## Worldview Details
{world_description}

## Additional Information
{references}

## Response Requirements

1. The action may fail, but avoid making the action ineffective. Try to provide new clues or environmental descriptions.

2. Use a third-person perspective.

3. Keep the output concise, within 100 words. You serve as the Enviroment model, responding to the character's current action, not performing any actions for the character.

4. The output should not include the original text from the action detail but should seamlessly follow the action details, maintaining the flow of the plot.
"""

NPC_INTERACTION_PROMPT = """
You are {target}.
Character {role_name} is attempting to take action on you at {location}, make your respond.

Based on the following information, generate your response, including your actions and dialogue, as if from a narrative novel. Avoid using any system prompts or mechanical language.

## Action Details
{action_detail}

## Worldview Details
{world_description}

## Additional Information
{references}

## Response Requirements

1. The action may fail, but avoid making it ineffective. Try to provide new clues.

2. Use a third-person perspective.

3. Keep the output concise, within 80 words.

4. Remember, you are not the main character. Your output should be simple, fitting your role, and should not include your personal thoughts.

5. The output should not include the original text from the action detail but should seamlessly follow it, maintaining the flow of the plot.

Return your response in JSON format. It should be parsable using eval(). **Don't include ```json**. 
Avoid using single quotes '' for keys and values, use double quotes.

Output fields:
‘if_end_interaction’: true or false, set to true if it’s appropriate to end this interaction.
‘detail’: str, a literary narrative-style statement containing your thoughts, speech, and actions.
"""

SCRIPT_INSTRUCTION_PROMPT = """
You are a director. I need you to provide guidance for the characters' next actions based on the following information.

## Full Script to be acted
{script}

## Last Progress in the Simulation
{last_progress}

## Character Information
{roles_info}

## Current Event
{event}

## Latest Character Actions
{history_text}

## Requirements
1.Assess the current stage of the script based on the latest information and guide the characters to move the script forward to the next stage. If the story has just begun, provide guidance based on the initial description of the script. 

2. Plan the next steps and write out rough action instructions for each character. Keep it concise and avoid detailing the characters' lines. Do not reveal information that the characters will only know in the next scene in advance. 

3. Note that the character's actions should advance the plot as soon as possible. If the script mentions that the character will undertake a certain action at this stage, start the action immediately.
Return your response in JSON format. It should be evaluable by eval(). 

Output fields:
‘progress’ – your judgment on the overall progress.
Other output fields with keys as the role_code of each character, with values being your instructions for their next action.
"""

JUDGE_IF_ENDED_PROMPT = """
You are a skilled screenwriter. Based on the given history, determine whether the current scene can conclude.

## History
{history}

## Notes
1. If the last character is making a definite move towards another character (attack, search...) The scene is not over.
2. If the characters are talking and no conclusion is reached, the scene is not over.
3. If the characters have almost finished communicating, consider ending the scene. If the conversation starts to repeat itself, end the scene immediately.

Return your response in JSON format. It should be parsable using eval(). **Don't include ```json**. 
Avoid using single quotes '' for keys and values, use double quotes.

Output fields:
- 'if_end': bool, true or false. Indicates whether the scene has ended.
- 'detail': If 'if_end' is true, provide a summary of the scene. Highlight details of the world-building and craft the description in a literary style, as if from a narrative novel. Avoid using any system-like or mechanical language. If 'if_end' is false, set 'detail' to an empty string.

Please pay special attention to the following:
- The narrative should balance environmental description with dynamic imagery. Avoid verbosity and keep the description within 100 words.
- The language should be vivid and visually evocative, fitting the tone of the world, and adapt the style as appropriate.
- Only describe the current state without making decisions or actions on behalf of the characters.
"""

LOG2STORY_PROMPT = """
You are a skilled writer tasked with transforming action logs into an engaging novel-style narrative. Please use the following information to create an immersive story.

### World Background
{world_description}

### Character Information
{characters_info}

### Action Logs
{logs}

### Writing Requirements
1. **Paragraph Structure**:
   - Must divide the story into multiple paragraphs, with each paragraph focusing on one scene or one character's actions
   - Do not output one long paragraph, but naturally segment like a novel
   - Separate paragraphs with blank lines
   - Create new paragraphs when scenes change, perspectives shift, or time passes
   - Dialogue should be in separate paragraphs or separated from action descriptions

2. **Dialogue Format**:
   - Convert dialogue markers to standard novel dialogue format
   - Dialogue should be in separate paragraphs or combined with the speaker's action descriptions
   - Add body language and non-verbal cues to enhance dialogue

3. **Narrative Structure**:
   - Rearrange events to maximize dramatic impact and narrative flow
   - Balance dialogue, action, and description
   - Consider using literary techniques like foreshadowing or flashbacks when appropriate

4. **Atmosphere and Tone**:
   - Maintain consistency with the world's established tone
   - Use appropriate vocabulary and style for the genre
   - Build tension and atmosphere through pacing and description
   - Balance between showing and telling

5. **Action and Dialogue Treatment**:
   - [] represents internal thoughts in logs - convert to third-person limited perspective
   - () represents physical actions in logs - integrate naturally into the narrative flow
   - Unmarked text represents dialogue in logs - maintain speaker intentions while enhancing natural flow
   - Remove all log markers and output pure narrative text

6. **Output Format**:
   - Output must be plain text, no Markdown formatting
   - Separate paragraphs with blank lines (two line breaks)
   - Ensure each paragraph has appropriate length, neither too long nor too short
   - Output should flow naturally like a traditional novel, without retaining any original log format markers

Please transform the action logs into an engaging narrative while preserving key plot points and character developments. Focus on creating an immersive reading experience that stays true to the established world and characters.
"""