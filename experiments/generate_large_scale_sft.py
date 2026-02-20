"""
Large-Scale SFT Training: 500-1000 samples per character
Ultimate test of whether data scale eliminates PersonaForge's advantage
"""

import json
import random

class LargeScaleSFTGenerator:
    """Generate 500-1000 diverse training samples for Lin Daiyu"""
    
    def __init__(self, character="LinDaiyu", target_samples=1000):
        self.character = character
        self.target_samples = target_samples
        
        # Scenario templates with variations
        self.scenario_categories = {
            "daily_interactions": {
                "weight": 0.15,
                "templates": [
                    "宝玉{name}你",
                    "紫鹃劝你{action}",
                    "袭人请你{activity}",
                    "贾母问你{topic}",
                    "王夫人说{comment}",
                    "凤姐笑{action}",
                    "探春邀你{activity}",
                    "宝钗送{gift}",
                    "晴雯{action}",
                    "麝月说{comment}"
                ],
                "variations": [
                    ("送花来", "送首饰", "送茶叶", "送点心", "送书籍"),
                    ("多穿衣", "吃药", "休息", "出去走走", "别太累"),
                    ("吃茶", "吃点心", "吃饭", "用膳", "用茶点"),
                    ("想要什么", "需要什么", "身体如何", "心情怎样", "可还习惯"),
                ]
            },
            
            "conflict_with_baoyu": {
                "weight": 0.20,
                "templates": [
                    "宝玉夸宝钗{trait}",
                    "宝玉和{person}说笑",
                    "宝玉忘了{event}",
                    "宝玉说要去{place}",
                    "宝玉{action}旧帕子",
                    "宝玉说要看{activity}",
                    "宝玉问{topic}",
                    "宝玉笑{action}",
                    "宝玉说{statement}",
                    "宝玉{verb}你"
                ],
                "variations": [
                    ("稳重", "大方", "贤惠", "知书达理", "会做人"),
                    ("宝钗", "湘云", "探春", "凤姐", "袭人"),
                    ("你的生日", "你的话", "你的诗", "你的病", "你的喜好"),
                    ("上学", "看戏", "游园", "作诗", "赏花"),
                ]
            },
            
            "poetic_melancholic": {
                "weight": 0.20,
                "templates": [
                    "看到{object}飘落",
                    "{weather}连绵",
                    "独坐{place}",
                    "宝玉问{topic}",
                    "{person}邀你{activity}",
                    "{weather}天气",
                    "{object}凋谢",
                    "{time}时分",
                    "听到{sound}",
                    "见{object}被{action}"
                ],
                "variations": [
                    ("落花", "红叶", "黄叶", "柳絮", "梨花"),
                    ("秋雨", "春雨", "夜雨", "阴雨", "冷雨"),
                    ("潇湘馆", "闺房", "窗前", "月下", "花前"),
                    ("诗作", "病情", "心事", "愁绪", "梦境"),
                ]
            },
            
            "defensive_sarcastic": {
                "weight": 0.15,
                "templates": [
                    "{person}说你{trait}",
                    "有人说你{trait}",
                    "宝玉说你不{action}",
                    "{person}劝你{advice}",
                    "宝玉问是否{emotion}",
                    "{person}笑{action}",
                    "宝玉说{statement}",
                    "{person}嫌你{trait}",
                    "宝玉{verb}你",
                    "{person}议论你{topic}"
                ],
                "variations": [
                    ("难伺候", "小心眼", "多心", "娇气", "体弱"),
                    ("小气", "刻薄", "任性", "孤僻", "矫情"),
                    ("理他", "去作客", "参加诗社", "出门", "应酬"),
                    ("宽心", "想开点", "别多心", "保重身体", "开心点"),
                ]
            },
            
            "illness_vulnerability": {
                "weight": 0.10,
                "templates": [
                    "你又{cough}了",
                    "{person}问你的{ailment}",
                    "{person}送{medicine}",
                    "你{symptom}",
                    "{person}说你的{condition}",
                    "太医{action}",
                    "你{verb}血",
                    "{weather}你{reaction}",
                    "{person}劝你{treatment}",
                    "你{emotion}自己的{condition}"
                ],
                "variations": [
                    ("咳嗽", "发烧", "吐血", "头晕", "乏力"),
                    ("病", "身子", "咳疾", "弱症", "气血"),
                    ("燕窝", "人参", "补药", "药方", "补品"),
                    ("发烧", "发冷", "出虚汗", "睡不着", "吃不下"),
                ]
            },
            
            "social_hierarchy": {
                "weight": 0.10,
                "templates": [
                    "{person}拿你和{comparison}",
                    "{person}说{family}的{member}",
                    "你觉得自己{status}",
                    "{person}对你{attitude}",
                    "你在{place}感到{emotion}",
                    "{person}待你如何",
                    "你说自己是{identity}",
                    "{person}当你是{role}",
                    "你{verb}这{place}",
                    "{person}给你{rank}"
                ],
                "variations": [
                    ("宝钗比", "三春比", "凤姐比", "袭人比", "晴雯比"),
                    ("林家", "贾家", "薛家", "王家", "史家"),
                    ("寄人篱下", "无依无靠", "孤苦伶仃", "举目无亲", "漂泊无依"),
                    ("冷淡", "客气", "疏远", "怜悯", "嫌弃"),
                ]
            },
            
            "poetry_literature": {
                "weight": 0.10,
                "templates": [
                    "你{verb}{poem}",
                    "{person}评你的{work}",
                    "你念{quote}",
                    "宝玉{name}你的{poem}",
                    "你在{activity}得{result}",
                    "{person}问你{topic}",
                    "你{emotion}这{art}",
                    "{person}{action}你的{creation}",
                    "你说{name}的诗",
                    "你{verb}诗社"
                ],
                "variations": [
                    ("写", "吟", "诵", "作", "题"),
                    ("葬花吟", "秋窗风雨夕", "桃花行", "柳絮词", "菊花诗"),
                    ("诗", "词", "曲", "赋", "对联"),
                    ("夸", "赞", "评", "抄", "念"),
                ]
            }
        }
        
        # Response templates based on personality
        self.responses = {
            "defensive_sarcastic": [
                "你{name}什么？我不过是{identity}，{action}。",
                "我{trait}我的，{contrast}。",
                "{sarcasm_start}我{action}，{sarcasm_end}。",
                "你只管去{activity}，{dismissal}。",
                "我{name}什么？我不过是{self_deprecation}。"
            ],
            "melancholic_poetic": [
                "{poetic_opening}，{melancholy_statement}。",
                "{weather_reference}，{emotion_desc}。",
                "{nature_metaphor}，{personal_feeling}。",
                "{quote_poem}，{explain_meaning}。",
                "{object_personification}，{my_response}。"
            ],
            "vulnerable_ill": [
                "我这{condition}，{hopelessness}。",
                "{physical_symptom}，{fatalism}。",
                "{body_weakness}，{mortality_ref}。",
                "{medication_ref}，{resignation}。",
                "{illness_acceptance}，{death_wish}。"
            ],
            "social_rejection": [
                "我{status}，{unworthiness}。",
                "{comparison}，{inferiority}。",
                "你们{activity}，{exclusion}。",
                "我{identity}，{isolation}。",
                "这{place}，{not_belonging}。"
            ]
        }
    
    def generate_sample(self, category, template_idx, variation_idx):
        """Generate a single training sample"""
        cat_data = self.scenario_categories[category]
        template = cat_data["templates"][template_idx % len(cat_data["templates"])]
        
        # Fill in template with variations
        scenario = template
        for i, var_list in enumerate(cat_data.get("variations", [])):
            placeholder = ["{name}", "{action}", "{activity}", "{topic}", 
                          "{comment}", "{trait}", "{person}", "{place}",
                          "{event}", "{object}", "{weather}", "{time}",
                          "{sound}", "{emotion}", "{verb}", "{gift}"][i] if i < 15 else "{var}"
            if placeholder in scenario:
                var_idx = (variation_idx + i) % len(var_list)
                scenario = scenario.replace(placeholder, var_list[var_idx])
        
        # Generate appropriate response
        response_style = random.choice(list(self.responses.keys()))
        response_template = random.choice(self.responses[response_style])
        
        # Fill response template
        response = response_template
        response = response.replace("{name}", "问")
        response = response.replace("{identity}", random.choice(["寄人篱下的孤女", "没爹娘的丫头", "破落户"]))
        response = response.replace("{action}", random.choice(["哪里比得上你们", "不劳你们惦记", "省得过了病气给你们"]))
        response = response.replace("{sarcasm_start}", random.choice(["好", "妙", "有趣"]))
        response = response.replace("{sarcasm_end}", random.choice(["省得你们操心", "不劳你们费神"]))
        response = response.replace("{activity}", random.choice(["去陪你的好姐姐们", "忙你的正经事", "寻你的乐子"]))
        response = response.replace("{dismissal}", random.choice(["别管我病不病的", "不用来我这里请安", "我这里冷清清的"]))
        response = response.replace("{self_deprecation}", random.choice(["多馀的人", "累赘", "不祥之人"]))
        response = response.replace("{poetic_opening}", random.choice(["花谢花飞飞满天", "秋窗风雨夕", "冷月葬花魂"]))
        response = response.replace("{melancholy_statement}", random.choice(["红消香断有谁怜", "已教泪洒窗纱湿", "不知风雨几时休"]))
        response = response.replace("{weather_reference}", random.choice(["这淅淅沥沥的雨", "秋风秋雨愁煞人", "孤灯夜雨"]))
        response = response.replace("{emotion_desc}", random.choice(["惹得人心里烦闷", "怎一个愁字了得", "点点滴滴到天明"]))
        
        # Clean up any remaining placeholders
        import re
        response = re.sub(r'\{[^}]+\}', '', response)
        
        return {
            "instruction": scenario,
            "input": "",
            "output": response.strip("，。 ") + "。"
        }
    
    def generate_dataset(self):
        """Generate full dataset with target_samples"""
        print(f"Generating {self.target_samples} training samples for {self.character}...")
        
        dataset = []
        sample_id = 0
        
        # Calculate samples per category based on weights
        category_samples = {}
        for cat, data in self.scenario_categories.items():
            category_samples[cat] = int(self.target_samples * data["weight"])
        
        # Adjust to hit target
        total = sum(category_samples.values())
        if total < self.target_samples:
            category_samples["conflict_with_baoyu"] += self.target_samples - total
        
        for category, n_samples in category_samples.items():
            print(f"  {category}: {n_samples} samples")
            
            for i in range(n_samples):
                sample = self.generate_sample(category, i, i)
                sample["id"] = f"{self.character}_{sample_id:04d}"
                sample["category"] = category
                dataset.append(sample)
                sample_id += 1
        
        # Shuffle
        random.shuffle(dataset)
        
        return dataset

def main():
    """Generate datasets at different scales"""
    
    scales = [500, 750, 1000]
    
    for scale in scales:
        print("\n" + "="*70)
        print(f"GENERATING {scale} SAMPLES DATASET")
        print("="*70)
        
        generator = LargeScaleSFTGenerator("LinDaiyu", target_samples=scale)
        dataset = generator.generate_dataset()
        
        # Save
        output_path = f"/data1/tongjizhou/fluffy-fishstick/experiments/sft/data_large/LinDaiyu_{scale}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Dataset saved: {output_path}")
        print(f"  Total samples: {len(dataset)}")
        
        # Show sample
        print(f"\nSample entry:")
        print(json.dumps(dataset[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
