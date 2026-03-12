import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional

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


def compose_prompt(user_task: str, main_content: str) -> str:
    """
    组合最终 prompt。
    
    user_task:
        用户对页面风格、场景、设计倾向的要求
        例如：我希望做一个学术风格的，偏清新风格，适用于学术交流的场景。
    
    main_content:
        需要被总结并写入 HTML 的主题内容
    """
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

图形与插图策略：
10. 可以适度加入插图和视觉装饰，使页面更美观，但必须遵循“优先 CSS 化”的原则。
11. 基础图形请尽量用 CSS 表达，例如：
   - 色块
   - 分隔线
   - 圆点
   - 箭头
   - 标签
   - 卡片
   - 边框高亮
   - 简单几何形状
12. 复杂图形允许使用 inline SVG，但仅限必要场景，例如：
   - 流程示意图中的不规则连接
   - 简单示意性图标
   - 结构关系图
   - 抽象小插图
13. 即使使用 inline SVG，也应保持简洁，避免超复杂路径、过多节点、过度装饰。
14. 整体策略必须是：优先 CSS 化，复杂图形再用 inline SVG，以便后续编辑和转换。
15. 不要把整块内容做成单张图片，不要使用 base64 图片承载主要内容。
16. SVG 仅用于辅助插图，不用于承载核心信息文字。
17. HTML所有元素要全部限制在固定画布中，请你合理根据画布大小排布各种元素。

样式限制：
17. 不要使用 canvas。
18. 不要使用复杂滤镜、mask、clip-path、backdrop-filter。
19. 尽量不要使用 ::before / ::after；如必须使用，也仅限极简单装饰，不能承载关键信息。
20. 不要做成网页宣传页，不要做成炫技式设计稿。
21. 风格要求简洁、专业、现代，并结合用户提出的风格倾向进行设计。

输出要求：
22. 输出完整 HTML，可直接保存为 .html 文件打开。
23. 不要输出 markdown 代码块，不要解释，只输出纯 HTML。

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

    final_prompt = compose_prompt(user_task=user_task, main_content=main_content)

    print("\n" + "=" * 72)
    print("开始执行任务")
    # print("=" * 72)
    # print(final_prompt)
    # print("=" * 72)

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
        "--task",
        type=str,
        required=True,
        help="用户对页面风格/场景的要求，例如：学术风格、偏清新、适合学术交流",
    )
    parser.add_argument(
        "--content",
        type=str,
        required=True,
        help="主题内容，供 nanobot 总结后生成 HTML",
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
        result = await run_once(
            user_task=args.task,
            main_content=args.content,
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