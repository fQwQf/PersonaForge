"""
Self Identity API Routes - 简化版身份创建API

提供身份建议、目标建议和用户agent创建功能
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import os
import json

from sw_utils import get_models, load_json_file

router = APIRouter(prefix="/api/self-identity", tags=["self-identity"])
security = HTTPBearer()


# ===== Pydantic Models =====

class SuggestIdentityRequest(BaseModel):
    scroll_id: int
    mbti: str
    big_five: Dict[str, float]


class SuggestGoalRequest(BaseModel):
    scroll_id: int
    mbti: str
    big_five: Dict[str, float]
    identity: str


class CreateUserAgentRequest(BaseModel):
    scroll_id: int
    mbti: str
    big_five: Dict[str, float]
    identity: str
    goal: str
    nickname: Optional[str] = None


class SuggestionResponse(BaseModel):
    suggestions: List[str]


# ===== Helper Functions =====

def get_scroll_info(scroll_id: int) -> Dict[str, Any]:
    """获取卷轴/世界信息"""
    # 从数据库获取scroll信息
    from database import db
    scroll = db.get_scroll(scroll_id)
    if not scroll:
        raise HTTPException(status_code=404, detail=f"Scroll not found: {scroll_id}")
    return scroll


def get_world_description(scroll: Dict[str, Any]) -> str:
    """从scroll获取世界描述，尝试多个来源"""
    description_parts = []
    
    # 添加scroll名称作为上下文
    scroll_name = scroll.get("name", "")
    if scroll_name:
        description_parts.append(f"世界名称: {scroll_name}")
    
    # 尝试从world_file获取
    world_file = scroll.get("world_file", "")
    if world_file and os.path.exists(world_file):
        world_data = load_json_file(world_file)
        if world_data.get("description"):
            description_parts.append(world_data.get("description"))
        if world_data.get("era"):
            description_parts.append(f"时代背景: {world_data.get('era')}")
        if world_data.get("setting"):
            description_parts.append(f"世界设定: {world_data.get('setting')}")
    
    # 尝试从orchestrator.json获取
    scroll_dir = scroll.get("scroll_dir", "")
    if scroll_dir:
        orch_path = os.path.join(scroll_dir, "orchestrator.json")
        if os.path.exists(orch_path):
            orch_data = load_json_file(orch_path)
            if orch_data.get("description") and orch_data.get("description") not in description_parts:
                description_parts.append(orch_data.get("description"))
            if orch_data.get("event"):
                description_parts.append(f"当前剧情: {orch_data.get('event')}")
    
    # 尝试从world_info.json获取
    if scroll_dir:
        world_info_path = os.path.join(scroll_dir, "world_info.json")
        if os.path.exists(world_info_path):
            world_info = load_json_file(world_info_path)
            if world_info.get("description") and world_info.get("description") not in description_parts:
                description_parts.append(world_info.get("description"))
    
    # 使用scroll自带的description作为备选
    if not description_parts and scroll.get("description"):
        description_parts.append(scroll.get("description"))
    
    # 如果还是没有，使用默认
    if not description_parts:
        return "一个神秘的世界"
    
    return "\n".join(description_parts)


def generate_identity_suggestions(mbti: str, big_five: Dict[str, float], world_description: str, language: str = "zh") -> List[str]:
    """
    基于人格和世界设定生成身份建议
    """
    llm = get_models("gemini-2.5-flash")
    
    # 构建人格描述
    traits = []
    if big_five.get("openness", 0.5) > 0.6:
        traits.append("creative and curious" if language == "en" else "富有创造力和好奇心")
    if big_five.get("extraversion", 0.5) > 0.6:
        traits.append("outgoing" if language == "en" else "外向活泼")
    elif big_five.get("extraversion", 0.5) < 0.4:
        traits.append("introverted" if language == "en" else "内敛安静")
    if big_five.get("conscientiousness", 0.5) > 0.6:
        traits.append("organized" if language == "en" else "认真负责")
    
    traits_text = "、".join(traits) if traits else "平和"
    
    if language == "zh":
        prompt = f"""根据以下信息，为用户推荐4个适合的身份/角色类型：

用户MBTI类型: {mbti}
用户性格特点: {traits_text}

世界背景:
{world_description[:500] if len(world_description) > 500 else world_description}

要求:
1. 身份应该与用户的性格特点相匹配
2. 身份应该适合这个世界的背景设定
3. 每个身份用2-4个字描述
4. 返回JSON格式: {{"suggestions": ["身份1", "身份2", "身份3", "身份4"]}}

只返回JSON，不要其他内容。"""
    else:
        prompt = f"""Based on the following information, recommend 4 suitable identities/roles for the user:

User MBTI Type: {mbti}
User Personality: {traits_text}

World Background:
{world_description[:500] if len(world_description) > 500 else world_description}

Requirements:
1. Identities should match user's personality
2. Identities should fit the world setting
3. Each identity should be 2-4 words
4. Return JSON format: {{"suggestions": ["identity1", "identity2", "identity3", "identity4"]}}

Return only JSON, no other content."""
    
    try:
        response = llm.chat(prompt)
        # 尝试解析JSON
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result.get("suggestions", ["旅行者", "学者", "探险家", "观察者"])
    except Exception as e:
        print(f"Error generating identity suggestions: {e}")
    
    # 默认建议
    if language == "zh":
        return ["旅行者", "学者", "探险家", "观察者"]
    else:
        return ["Traveler", "Scholar", "Explorer", "Observer"]


def generate_goal_suggestions(mbti: str, big_five: Dict[str, float], identity: str, world_description: str, language: str = "zh") -> List[str]:
    """
    基于人格、身份和世界设定生成目标建议
    """
    llm = get_models("gemini-2.5-flash")
    
    if language == "zh":
        prompt = f"""根据以下信息，为用户推荐4个合适的目标：

用户MBTI类型: {mbti}
用户身份: {identity}

世界背景:
{world_description[:500] if len(world_description) > 500 else world_description}

要求:
1. 目标应该与用户的身份和性格匹配
2. 目标应该在这个世界中可以实现
3. 每个目标用简短的一句话描述（10-20字）
4. 返回JSON格式: {{"suggestions": ["目标1", "目标2", "目标3", "目标4"]}}

只返回JSON，不要其他内容。"""
    else:
        prompt = f"""Based on the following information, recommend 4 suitable goals for the user:

User MBTI Type: {mbti}
User Identity: {identity}

World Background:
{world_description[:500] if len(world_description) > 500 else world_description}

Requirements:
1. Goals should match user's identity and personality
2. Goals should be achievable in this world
3. Each goal should be a brief sentence (5-15 words)
4. Return JSON format: {{"suggestions": ["goal1", "goal2", "goal3", "goal4"]}}

Return only JSON, no other content."""
    
    try:
        response = llm.chat(prompt)
        # 尝试解析JSON
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result.get("suggestions", ["探索这个世界", "结识有趣的人", "寻找未知的秘密", "体验新的冒险"])
    except Exception as e:
        print(f"Error generating goal suggestions: {e}")
    
    # 默认建议
    if language == "zh":
        return ["探索这个世界", "结识有趣的人", "寻找未知的秘密", "体验新的冒险"]
    else:
        return ["Explore this world", "Meet interesting people", "Discover hidden secrets", "Experience new adventures"]


# ===== API Endpoints =====

@router.post("/suggest-identity", response_model=SuggestionResponse)
async def suggest_identity(request: SuggestIdentityRequest):
    """
    基于MBTI和Big Five生成身份建议
    """
    try:
        scroll = get_scroll_info(request.scroll_id)
        world_description = get_world_description(scroll)
        language = scroll.get("language", "zh")
        
        suggestions = generate_identity_suggestions(
            mbti=request.mbti,
            big_five=request.big_five,
            world_description=world_description,
            language=language
        )
        
        return SuggestionResponse(suggestions=suggestions)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in suggest_identity: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/suggest-goal", response_model=SuggestionResponse)
async def suggest_goal(request: SuggestGoalRequest):
    """
    基于人格、身份和世界设定生成目标建议
    """
    try:
        scroll = get_scroll_info(request.scroll_id)
        world_description = get_world_description(scroll)
        language = scroll.get("language", "zh")
        
        suggestions = generate_goal_suggestions(
            mbti=request.mbti,
            big_five=request.big_five,
            identity=request.identity,
            world_description=world_description,
            language=language
        )
        
        return SuggestionResponse(suggestions=suggestions)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in suggest_goal: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create")
async def create_user_agent(
    request: CreateUserAgentRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    创建用户agent（使用人格+身份+目标）
    
    Returns:
        用户agent信息，包括生成的role_code
    """
    try:
        # 验证token并获取用户信息
        from database import db
        token = credentials.credentials
        user = db.verify_token(token)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user_id = user["id"]
        username = user.get("username", f"user_{user_id}")
        
        # 生成唯一的role_code
        import uuid
        role_code = f"user_{user_id}_{uuid.uuid4().hex[:8]}"
        
        # 获取scroll信息
        scroll = get_scroll_info(request.scroll_id)
        world_file = scroll.get("world_file", "")
        
        # 保存用户agent信息到数据库或文件
        user_agent_info = {
            "user_id": user_id,
            "role_code": role_code,
            "scroll_id": request.scroll_id,
            "mbti": request.mbti,
            "big_five": request.big_five,
            "identity": request.identity,
            "goal": request.goal,
            "nickname": request.nickname or username
        }
        
        # 保存到用户数据
        user_agents_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "user_agents")
        os.makedirs(user_agents_dir, exist_ok=True)
        
        user_agent_file = os.path.join(user_agents_dir, f"{role_code}.json")
        with open(user_agent_file, 'w', encoding='utf-8') as f:
            json.dump(user_agent_info, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "role_code": role_code,
            "agent_info": user_agent_info,
            "message": "用户agent创建成功"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in create_user_agent: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user-agent/{role_code}")
async def get_user_agent(
    role_code: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    获取用户agent信息
    """
    try:
        user_agents_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "user_agents")
        user_agent_file = os.path.join(user_agents_dir, f"{role_code}.json")
        
        if not os.path.exists(user_agent_file):
            raise HTTPException(status_code=404, detail="User agent not found")
        
        with open(user_agent_file, 'r', encoding='utf-8') as f:
            user_agent_info = json.load(f)
        
        return {
            "success": True,
            "agent_info": user_agent_info
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_user_agent: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
