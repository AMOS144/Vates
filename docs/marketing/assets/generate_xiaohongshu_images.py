"""生成小红书发布使用的 8 张竖版技术信息图。

运行命令：
uv run --with pillow python docs/marketing/assets/generate_xiaohongshu_images.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


WIDTH = 1200
HEIGHT = 1600
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "xiaohongshu"

FONT_CN = "/System/Library/Fonts/Hiragino Sans GB.ttc"
FONT_BOLD = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_MONO = "/System/Library/Fonts/SFNSMono.ttf"

BG_TOP = (8, 13, 28)
BG_BOTTOM = (3, 7, 17)
CARD = (15, 25, 46)
CARD_LIGHT = (21, 35, 62)
TEXT = (241, 247, 255)
MUTED = (145, 165, 191)
CYAN = (40, 218, 255)
PURPLE = (148, 104, 255)
GREEN = (78, 232, 158)
ORANGE = (255, 178, 76)
RED = (255, 103, 126)
GRID = (30, 47, 75)


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    """载入中英文系统字体。"""
    path = FONT_MONO if mono else FONT_BOLD if bold else FONT_CN
    return ImageFont.truetype(path, size=size)


def background(accent: tuple[int, int, int]) -> Image.Image:
    """生成带渐变、网格和光晕的竖版背景。"""
    image = Image.new("RGB", (WIDTH, HEIGHT), BG_BOTTOM)
    draw = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        ratio = y / (HEIGHT - 1)
        color = tuple(
            round(BG_TOP[i] * (1 - ratio) + BG_BOTTOM[i] * ratio)
            for i in range(3)
        )
        draw.line((0, y, WIDTH, y), fill=color)
    for x in range(0, WIDTH, 75):
        draw.line((x, 0, x, HEIGHT), fill=GRID, width=1)
    for y in range(0, HEIGHT, 75):
        draw.line((0, y, WIDTH, y), fill=GRID, width=1)
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.ellipse((250, 300, 1050, 1100), fill=(*accent, 90))
    layer = layer.filter(ImageFilter.GaussianBlur(220))
    image.paste(layer, (0, 0), layer)
    return image


def card(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int] = CARD,
    outline: tuple[int, int, int] = GRID,
    radius: int = 34,
    width: int = 2,
) -> None:
    """绘制带柔和阴影的圆角卡片。"""
    x1, y1, x2, y2 = box
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        (x1 + 8, y1 + 15, x2 + 8, y2 + 15),
        radius=radius,
        fill=(0, 0, 0, 130),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(18))
    image.paste(shadow, (0, 0), shadow)
    ImageDraw.Draw(image).rounded_rectangle(
        box,
        radius=radius,
        fill=fill,
        outline=outline,
        width=width,
    )


def page_header(
    image: Image.Image,
    page: int,
    title: str,
    subtitle: str,
    accent: tuple[int, int, int],
) -> None:
    """绘制小红书页面统一页眉。"""
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((70, 70, 220, 126), radius=28, fill=(15, 48, 72), outline=accent, width=2)
    draw.text((145, 98), "VATES", font=font(24, bold=True), fill=accent, anchor="mm")
    draw.text((1130, 98), f"{page:02d} / 08", font=font(22, mono=True), fill=MUTED, anchor="rm")
    draw.multiline_text((70, 190), title, font=font(56, bold=True), fill=TEXT, spacing=12)
    draw.multiline_text((72, 340), subtitle, font=font(27), fill=MUTED, spacing=10)


def page_footer(image: Image.Image, text: str = "项目实测 · 结果因设备、模型与配置而异") -> None:
    """绘制统一脚注。"""
    draw = ImageDraw.Draw(image)
    draw.line((70, 1502, 1130, 1502), fill=GRID, width=2)
    draw.text((70, 1545), text, font=font(18), fill=MUTED, anchor="lm")
    draw.text((1130, 1545), "AMOS144/Vates", font=font(18, mono=True), fill=MUTED, anchor="rm")


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int] = CYAN,
    width: int = 6,
) -> None:
    """绘制横向或纵向箭头。"""
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 > x1 else -1
        draw.polygon(
            ((x2, y2), (x2 - 22 * direction, y2 - 14), (x2 - 22 * direction, y2 + 14)),
            fill=color,
        )
    else:
        direction = 1 if y2 > y1 else -1
        draw.polygon(
            ((x2, y2), (x2 - 14, y2 - 22 * direction), (x2 + 14, y2 - 22 * direction)),
            fill=color,
        )


def chip(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    width: int,
) -> None:
    """绘制胶囊标签。"""
    x, y = center
    draw.rounded_rectangle((x - width // 2, y - 28, x + width // 2, y + 28), radius=28, fill=(*color,))
    draw.text((x, y), text, font=font(20, bold=True), fill=BG_BOTTOM, anchor="mm")


def cover() -> None:
    """第 1 页：封面。"""
    image = background(CYAN)
    draw = ImageDraw.Draw(image)
    page_header(image, 1, "Mac 跑 80B\n大模型", "41GB 权重，不再等于 41GB 运行内存", CYAN)

    draw.text((70, 520), "41GB", font=font(150, bold=True), fill=PURPLE)
    draw.text((75, 682), "Qwen3-Next-80B-A3B 4-bit", font=font(25), fill=MUTED)
    arrow(draw, (315, 790), (885, 790), CYAN, width=8)
    chip(draw, (600, 790), "SSD STREAMING", CYAN, 320)
    draw.text((70, 900), "≈ 8GB", font=font(166, bold=True), fill=GREEN)
    draw.text((78, 1085), "运行内存峰值", font=font(34, bold=True), fill=TEXT)

    card(image, (70, 1190, 1130, 1405), fill=(12, 29, 47), outline=ORANGE)
    draw.text((120, 1260), "13–15", font=font(80, bold=True), fill=ORANGE)
    draw.text((430, 1282), "tok/s", font=font(34, bold=True), fill=TEXT)
    draw.text((120, 1360), "Apple Silicon · MLX · Vates 开源", font=font(25), fill=MUTED)
    page_footer(image)
    image.save(OUTPUT / "01-cover.png", optimize=True)


def moe_routing() -> None:
    """第 2 页：MoE 路由。"""
    image = background(PURPLE)
    draw = ImageDraw.Draw(image)
    page_header(image, 2, "512 个专家\n不需要同时上场", "每个 token、每个 MoE 层只激活其中 10 个", PURPLE)

    card(image, (70, 470, 1130, 1260), fill=(12, 21, 40), outline=PURPLE)
    start_x, start_y = 124, 555
    gap_x, gap_y = 66, 72
    active = {0, 7, 19, 34, 58, 73, 91, 112, 127, 143}
    for row in range(16):
        for col in range(16):
            idx = row * 16 + col
            x = start_x + col * gap_x
            y = start_y + row * gap_y // 2
            color = GREEN if idx in active else (45, 57, 82)
            radius = 13 if idx in active else 8
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    draw.text((100, 1195), "图中 256 个点为缩略示意", font=font(18), fill=MUTED)
    draw.text((1080, 1195), "10 / 512", font=font(36, bold=True, mono=True), fill=GREEN, anchor="rm")
    card(image, (70, 1300, 1130, 1445), fill=CARD_LIGHT, outline=GREEN)
    draw.text((600, 1352), "当前不用的专家，为什么必须占着高速内存？", font=font(29, bold=True), fill=TEXT, anchor="mm")
    draw.text((600, 1404), "Vates 从这个问题开始", font=font(21), fill=MUTED, anchor="mm")
    page_footer(image)
    image.save(OUTPUT / "02-moe-routing.png", optimize=True)


def ssd_expert_pool() -> None:
    """第 3 页：SSD 与双源专家池。"""
    image = background(CYAN)
    draw = ImageDraw.Draw(image)
    page_header(image, 3, "专家住在 SSD\n需要谁，就加载谁", "真实区负责当前计算，侧区保留预取与高频专家", CYAN)

    card(image, (70, 485, 1130, 800), fill=(10, 27, 44), outline=CYAN)
    draw.text((145, 570), "SSD", font=font(50, bold=True), fill=CYAN)
    draw.text((145, 645), "per-expert blob", font=font(27, mono=True), fill=TEXT)
    draw.text((145, 710), "一个专家 = 一段连续字节 = 一次 pread", font=font(24), fill=MUTED)
    for i in range(7):
        x = 755 + (i % 4) * 82
        y = 560 + (i // 4) * 92
        draw.rounded_rectangle((x, y, x + 58, y + 58), radius=12, fill=(25, 66, 88), outline=CYAN, width=2)
        draw.text((x + 29, y + 29), f"E{i}", font=font(16, mono=True), fill=TEXT, anchor="mm")

    arrow(draw, (600, 820), (600, 930), GREEN)
    draw.text((630, 875), "按需读取 / 预测预取", font=font(21), fill=GREEN)

    card(image, (70, 955, 550, 1335), fill=CARD_LIGHT, outline=GREEN)
    draw.text((110, 1020), "真实区", font=font(38, bold=True), fill=GREEN)
    draw.text((110, 1085), "当前路由确定要用", font=font(23), fill=MUTED)
    for row in range(3):
        for col in range(5):
            x = 112 + col * 78
            y = 1160 + row * 65
            draw.rounded_rectangle((x, y, x + 55, y + 42), radius=9, fill=(27, 83, 70))

    card(image, (650, 955, 1130, 1335), fill=CARD_LIGHT, outline=PURPLE)
    draw.text((690, 1020), "LFU 侧区", font=font(38, bold=True), fill=PURPLE)
    draw.text((690, 1085), "预取结果 + 近期高频", font=font(23), fill=MUTED)
    for row in range(3):
        for col in range(5):
            x = 692 + col * 78
            y = 1160 + row * 65
            alpha = 90 + (row * 5 + col) * 8
            draw.rounded_rectangle((x, y, x + 55, y + 42), radius=9, fill=(74, 54, min(150, alpha)))

    page_footer(image)
    image.save(OUTPUT / "03-ssd-expert-pool.png", optimize=True)


def prefetch() -> None:
    """第 4 页：跨层预测预取。"""
    image = background(PURPLE)
    draw = ImageDraw.Draw(image)
    page_header(image, 4, "预取不是盲猜\n提前运行未来层的 gate", "预测要宽，读取要克制；把 I/O 藏进中间层计算", PURPLE)

    steps = [
        ("当前层 x", "attention 后的较新隐藏状态", CYAN),
        ("未来层 gate", "早层 ahead=1 · 后层 ahead=3", PURPLE),
        ("24 个候选", "MTP 多 token 按 max 聚合", PURPLE),
        ("过滤缺口", "剔除常驻专家 · budget 截断", ORANGE),
        ("C++ pread", "GPU 回调触发 · 主线程零同步", CYAN),
        ("侧区命中", "没到位仍由 demand 读取", GREEN),
    ]
    y = 470
    for index, (title, detail, color) in enumerate(steps, start=1):
        card(image, (120, y, 1080, y + 132), fill=CARD_LIGHT, outline=color, radius=25)
        draw.ellipse((155, y + 38, 215, y + 98), fill=color)
        draw.text((185, y + 68), str(index), font=font(20, bold=True), fill=BG_BOTTOM, anchor="mm")
        draw.text((255, y + 40), title, font=font(29, bold=True), fill=TEXT)
        draw.text((255, y + 88), detail, font=font(21), fill=MUTED)
        if index < len(steps):
            arrow(draw, (600, y + 134), (600, y + 166), color, width=4)
        y += 165

    page_footer(image)
    image.save(OUTPUT / "04-cross-layer-prefetch.png", optimize=True)


def memory_and_speed() -> None:
    """第 5 页：内存与速度。"""
    image = background(GREEN)
    draw = ImageDraw.Draw(image)
    page_header(image, 5, "低内存，也要跑得动", "完整权重、运行峰值和交互速度是三件不同的事", GREEN)

    card(image, (70, 470, 1130, 835), fill=CARD, outline=GREEN)
    draw.text((120, 540), "运行内存峰值", font=font(30, bold=True), fill=MUTED)
    draw.text((120, 635), "≈ 8GB", font=font(122, bold=True), fill=GREEN)
    draw.text((120, 785), "4-bit 完整权重约 41GB", font=font(27), fill=TEXT)

    card(image, (70, 890, 1130, 1260), fill=CARD, outline=ORANGE)
    draw.text((120, 960), "生产路径生成速度", font=font(30, bold=True), fill=MUTED)
    draw.text((120, 1055), "13–15", font=font(122, bold=True), fill=ORANGE)
    draw.text((665, 1108), "tok/s", font=font(42, bold=True), fill=TEXT)
    draw.text((120, 1210), "Textual TUI · 实时显示速度与内存", font=font(27), fill=TEXT)

    card(image, (180, 1320, 1020, 1435), fill=(10, 30, 43), outline=CYAN)
    draw.text((600, 1377), "SSD 提供容量 · C++ 隐藏延迟 · MLX 执行计算", font=font(25, bold=True), fill=CYAN, anchor="mm")
    page_footer(image)
    image.save(OUTPUT / "05-memory-and-speed.png", optimize=True)


def long_context_kv() -> None:
    """第 6 页：长上下文 KV。"""
    image = background(GREEN)
    draw = ImageDraw.Draw(image)
    page_header(image, 6, "128k 上下文\n也要控制内存", "专家权重压下去后，KV cache 会成为新的增长项", GREEN)

    card(image, (70, 500, 1130, 1190), fill=CARD, outline=GREEN)
    draw.text((120, 565), "KV CACHE", font=font(25, bold=True, mono=True), fill=MUTED)
    draw.text((120, 690), "3.0 GiB", font=font(72, bold=True), fill=(107, 126, 154))
    arrow(draw, (430, 745), (650, 745), GREEN, width=8)
    draw.text((1080, 690), "0.68 GiB", font=font(72, bold=True), fill=GREEN, anchor="ra")
    draw.text((600, 850), "IsoQuant K4/V3 + SO(4)", font=font(34, bold=True), fill=TEXT, anchor="mm")
    draw.text((600, 920), "仅作用于 12 个全注意力层", font=font(25), fill=MUTED, anchor="mm")

    draw.rounded_rectangle((120, 1010, 1080, 1058), radius=24, fill=(35, 49, 73))
    draw.rounded_rectangle((120, 1010, 338, 1058), radius=24, fill=GREEN)
    draw.text((120, 1108), "约 77% KV 占用下降", font=font(27, bold=True), fill=GREEN)

    card(image, (140, 1265, 1060, 1425), fill=CARD_LIGHT, outline=PURPLE)
    draw.text((600, 1320), "线性注意力层的递归状态保持不变", font=font(27, bold=True), fill=TEXT, anchor="mm")
    draw.text((600, 1372), "长对话不只取决于模型权重", font=font(21), fill=MUTED, anchor="mm")
    page_footer(image)
    image.save(OUTPUT / "06-long-context-kv.png", optimize=True)


def correctness_and_no_go() -> None:
    """第 7 页：正确性与失败实验。"""
    image = background(ORANGE)
    draw = ImageDraw.Draw(image)
    page_header(image, 7, "最怕的不是崩溃\n而是悄悄读错专家", "每次提速都要回答：输出为什么没有被改坏？", ORANGE)

    card(image, (70, 470, 1130, 890), fill=CARD, outline=GREEN)
    draw.text((120, 530), "CORRECTNESS", font=font(22, bold=True, mono=True), fill=GREEN)
    checks = [
        "容量不变性：换池容量，greedy token 逐字节一致",
        "DUAL_VERIFY / STG_VERIFY：池槽字节真值校验",
        "专家并集实测：K=3、top-k=10 → 最大 30",
    ]
    for index, text in enumerate(checks):
        y = 620 + index * 90
        draw.ellipse((120, y - 20, 160, y + 20), fill=GREEN)
        draw.line((129, y, 137, y + 8), fill=BG_BOTTOM, width=5)
        draw.line((137, y + 8, 152, y - 9), fill=BG_BOTTOM, width=5)
        draw.text((190, y), text, font=font(24), fill=TEXT, anchor="lm")

    card(image, (70, 950, 1130, 1385), fill=CARD, outline=RED)
    draw.text((120, 1010), "NO-GO 也公开", font=font(31, bold=True), fill=RED)
    failures = [
        ("完整树形验证", "slots=32 时约为基线 0.46×"),
        ("滑动窗口专家池", "重载量超过 SSD 带宽"),
        ("事件门控异步", "机制成立，但端到端零收益"),
    ]
    for index, (title, detail) in enumerate(failures):
        y = 1100 + index * 90
        draw.text((120, y), "×", font=font(34, bold=True), fill=RED, anchor="lm")
        draw.text((185, y), title, font=font(25, bold=True), fill=TEXT, anchor="lm")
        draw.text((1080, y), detail, font=font(21), fill=MUTED, anchor="rm")

    page_footer(image)
    image.save(OUTPUT / "07-correctness-and-no-go.png", optimize=True)


def open_source() -> None:
    """第 8 页：开源行动。"""
    image = background(CYAN)
    draw = ImageDraw.Draw(image)
    page_header(image, 8, "Vates 已开源", "Apple Silicon 上的流式 MoE 推理实验", CYAN)

    draw.text((600, 540), "VATES", font=font(120, bold=True), fill=TEXT, anchor="mm")
    draw.text((600, 650), "STREAMING MoE · MLX · MTP", font=font(27, mono=True), fill=CYAN, anchor="mm")

    card(image, (100, 760, 1100, 930), fill=(245, 248, 252), outline=CYAN)
    draw.text((150, 810), "GitHub 搜索", font=font(22, bold=True), fill=(65, 78, 98))
    draw.text((150, 875), "AMOS144/Vates", font=font(43, bold=True, mono=True), fill=(8, 15, 28))
    draw.ellipse((1008, 820, 1058, 870), outline=(8, 15, 28), width=5)
    draw.line((1047, 860, 1075, 888), fill=(8, 15, 28), width=5)

    features = [
        ("Apache-2.0", GREEN),
        ("60 个测试文件", PURPLE),
        ("完整消融报告", ORANGE),
        ("Textual TUI", CYAN),
    ]
    positions = [(300, 1040), (900, 1040), (300, 1150), (900, 1150)]
    for (text, color), position in zip(features, positions):
        chip(draw, position, text, color, 330)

    card(image, (140, 1270, 1060, 1420), fill=CARD_LIGHT, outline=CYAN)
    draw.text((600, 1322), "看演示 · 跑起来 · 提 Issue · 点一个 Star", font=font(29, bold=True), fill=TEXT, anchor="mm")
    draw.text((600, 1375), "欢迎一起测试不同芯片、SSD 与上下文", font=font(22), fill=MUTED, anchor="mm")

    page_footer(image)
    image.save(OUTPUT / "08-open-source.png", optimize=True)


def main() -> None:
    """生成全部小红书图片。"""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generators = [
        cover,
        moe_routing,
        ssd_expert_pool,
        prefetch,
        memory_and_speed,
        long_context_kv,
        correctness_and_no_go,
        open_source,
    ]
    for generate in generators:
        generate()
    for path in sorted(OUTPUT.glob("*.png")):
        print(f"generated {path.name}")


if __name__ == "__main__":
    main()
