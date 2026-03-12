import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from nanobot.bus.queue import MessageBus
from nanobot.agent.loop import AgentLoop
from nanobot.providers.litellm_provider import LiteLLMProvider
from nanobot.config.loader import load_config


def make_provider(config) -> Optional[LiteLLMProvider]:
    """
    根据 nanobot 配置创建 LLM provider。
    """
    provider_cfg = config.get_provider()
    model = config.agents.defaults.model

    if not (provider_cfg and provider_cfg.api_key) and not model.startswith("bedrock/"):
        print("错误：未检测到可用的 API key。")
        print("请先执行 nanobot onboard，或检查 ~/.nanobot/config.json")
        return None

    return LiteLLMProvider(
        api_key=provider_cfg.api_key if provider_cfg else None,
        api_base=config.get_api_base(),
        default_model=model,
        extra_headers=provider_cfg.extra_headers if provider_cfg else None,
        provider_name=config.get_provider_name(),
    )


def build_agent(config, provider: LiteLLMProvider) -> AgentLoop:
    """
    创建 AgentLoop。
    """
    bus = MessageBus()

    brave_api_key = None
    restrict_to_workspace = False
    exec_config = None

    try:
        exec_config = config.tools.exec
    except Exception:
        pass

    try:
        restrict_to_workspace = config.tools.restrict_to_workspace
    except Exception:
        pass

    try:
        brave_api_key = config.tools.web.search.api_key or None
    except Exception:
        pass

    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_iterations=20,
        brave_api_key=brave_api_key,
        exec_config=exec_config,
        restrict_to_workspace=restrict_to_workspace,
    )
    return agent


def load_job_file(input_path: str) -> Dict[str, Any]:
    """
    从 JSON 文件读取任务配置。
    JSON 示例：
    {
      "task": "我希望做一个学术风格的，偏清新风格，适用于学术交流的场景。",
      "content": "nanobot 是一个超轻量级的个人 AI 助手......",
      "images": [
        "./images/demo1.png",
        "./images/demo2.jpg"
      ]
    }
    """
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")

    text = path.read_text(encoding="utf-8")
    data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError("输入 JSON 必须是对象格式。")

    task = data.get("task", "")
    content = data.get("content", "")
    images = data.get("images", [])

    if not task or not isinstance(task, str):
        raise ValueError("JSON 中必须提供字符串类型的 task。")

    if not content or not isinstance(content, str):
        raise ValueError("JSON 中必须提供字符串类型的 content。")

    if images is None:
        images = []

    if not isinstance(images, list):
        raise ValueError("JSON 中的 images 必须是数组，例如 [] 或 ['a.png']。")

    normalized_images: List[str] = []
    for img in images:
        if not isinstance(img, str):
            raise ValueError("images 数组中的每一项都必须是字符串路径。")
        normalized_images.append(str(Path(img).expanduser()))

    return {
        "task": task.strip(),
        "content": content.strip(),
        "images": normalized_images,
    }


def build_image_instruction(images: List[str]) -> str:
    """
    构造图片相关 prompt。
    """
    if not images:
        return """
图片输入情况：
- 当前未提供图片。
- 请不要强行预留空白图片区。
- 若为了美观需要视觉元素，请优先使用 CSS 基础图形，必要时使用简洁 inline SVG。
""".strip()

    image_lines = "\n".join([f"- {img}" for img in images])

    return f"""
图片输入情况：
- 当前提供了 {len(images)} 张图片。
- 图片路径如下：
{image_lines}

图片处理要求：
1. 若提供了图片，请将图片插入或排布到画板中合适的位置，使整体版式更美观。
2. 如果只有单张图片，请将其作为主视觉图、说明图或辅助图，放在与内容逻辑相符的位置。
3. 如果有多张图片，请设计为并列展示、图组展示、步骤图展示或主次搭配展示，但必须保证布局整洁。
4. 图片不得遮挡核心文字，不得超出固定画布范围。
5. 图片区域要与整体风格一致，避免显得突兀。
6. HTML 中请直接使用普通 <img> 标签引用这些图片路径，不要把图片转成 base64。
7. 图片应放在 data-region 标记的独立区域中，例如 data-region="image_panel" 或 data-region="gallery"。
8. 若图片较多，请合理筛选展示方式，避免画面拥挤。
""".strip()


def compose_prompt(user_task: str, main_content: str, images: List[str]) -> str:
    """
    组合最终 prompt。
    """
    image_instruction = build_image_instruction(images)

    return f"""
你是一个经验丰富的全栈代码专家，擅长编写“PPT 友好型”的可编辑 HTML。

当前用户对页面的额外需求是：
{user_task}

你的任务：
请根据下面的主题内容，生成一个“PPT 友好型”的单页 HTML，用于后续转换为可编辑 PPTX。
请注意：你需要先对主题内容进行提炼和总结，不要把原始内容全文照搬进 HTML。

核心目标：
- 页面美观、正式，适合汇报/答辩
- 转成 PPTX 后，文字尽量保持可编辑
- 图形元素尽量可编辑、可拆分、可替换
- 优先使用 CSS 实现基础图形，减少纯图片化风险

硬性要求：
1. 固定 16:9 画布，尺寸 1600x900（或 1280x720）。
2. 单页静态布局，不要响应式设计。
3. 所有文字必须是普通 HTML 真实文本节点，使用 h1、h2、div、p、ul、li 等标签承载。
4. 列表必须使用原生 ul/li，不要手工模拟项目符号。
5. 使用清晰的固定分区和 absolute 定位，方便映射为 PPT 文本框。
6. 每个主要内容块请用独立 div，并加 data-region 属性，例如 data-region="title"、data-region="left_main"。
7. 优先保证“转换后文字可编辑”，其次再考虑视觉效果。
8. 页面布局必须受控，不能出现内容超出画布范围的情况。
9. 所有核心文字必须保留在 HTML 文本节点中，不能放进 SVG。
10. HTML 所有元素必须全部限制在固定画布中，请根据画布大小合理排布。

图形与插图策略：
11. 可以适度加入插图和视觉装饰，使页面更美观，但必须遵循“优先 CSS 化”的原则。
12. 基础图形请尽量用 CSS 表达，例如：
   - 色块
   - 分隔线
   - 圆点
   - 箭头
   - 标签
   - 卡片
   - 边框高亮
   - 简单几何形状
13. 复杂图形允许使用 inline SVG，但仅限必要场景，例如：
   - 流程示意图中的不规则连接
   - 简单示意性图标
   - 结构关系图
   - 抽象小插图
14. 即使使用 inline SVG，也应保持简洁，避免超复杂路径、过多节点、过度装饰。
15. 整体策略必须是：优先 CSS 化，复杂图形再用 inline SVG，以便后续编辑和转换。
16. 不要把整块内容做成单张图片，不要使用 base64 图片承载主要内容。
17. SVG 仅用于辅助插图，不用于承载核心信息文字。
18. 请注意排版格式，不要有文字重叠，图文重叠，图重叠的现象。

样式限制：
18. 不要使用 canvas。
19. 不要使用复杂滤镜、mask、clip-path、backdrop-filter。
20. 尽量不要使用 ::before / ::after；如必须使用，也仅限极简单装饰，不能承载关键信息。
21. 不要做成网页宣传页，不要做成炫技式设计稿。
22. 风格要求简洁、专业、现代，并结合用户提出的风格倾向进行设计。

{image_instruction}

输出要求：
23. 输出完整 HTML，可直接保存为 .html 文件打开。
24. 不要输出 markdown 代码块，不要解释，只输出纯 HTML。

请按以上要求生成。
主题内容如下：
{main_content}
""".strip()


def save_html(content: str, output_path: str) -> Path:
    """
    将 HTML 内容保存到指定路径。
    """
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


async def run_once(
    user_task: str,
    main_content: str,
    images: List[str],
    session_key: str = "cli:auto",
    channel: str = "cli",
    chat_id: str = "auto",
) -> str:
    """
    单次执行入口。
    """
    print("正在加载 nanobot 配置...")
    config = load_config()

    print("正在初始化 LLM provider...")
    provider = make_provider(config)
    if provider is None:
        raise RuntimeError("provider 初始化失败，请检查配置。")

    print("正在初始化 AgentLoop...")
    agent = build_agent(config, provider)

    final_prompt = compose_prompt(
        user_task=user_task,
        main_content=main_content,
        images=images,
    )

    print("\n" + "=" * 72)
    print("开始执行任务")
    print("=" * 72)

    response = await agent.process_direct(
        content=final_prompt,
        session_key=session_key,
        channel=channel,
        chat_id=chat_id,
    )

    return response


def parse_args():
    parser = argparse.ArgumentParser(
        description="用 Python 直接调用 nanobot 生成可编辑 HTML"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 JSON 配置文件路径，例如 ./job.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output.html",
        help="生成的 HTML 保存路径，例如 ./result/demo.html",
    )
    parser.add_argument(
        "--session",
        type=str,
        default="cli:auto",
        help="会话标识，用于维持上下文，默认 cli:auto",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="cli",
        help="渠道标识，默认 cli",
    )
    parser.add_argument(
        "--chat-id",
        type=str,
        default="auto",
        help="聊天 ID，默认 auto",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    try:
        job = load_job_file(args.input)

        result = await run_once(
            user_task=job["task"],
            main_content=job["content"],
            images=job["images"],
            session_key=args.session,
            channel=args.channel,
            chat_id=args.chat_id,
        )

        print("\n" + "=" * 72)
        print("nanobot 响应")
        print("=" * 72)
        print(result)
        print("=" * 72)

        saved_path = save_html(result, args.output)
        print(f"\nHTML 已保存到：{saved_path}")

    except KeyboardInterrupt:
        print("\n已中断。")
        sys.exit(130)
    except Exception as e:
        print(f"\n执行失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())