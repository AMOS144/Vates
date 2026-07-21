"""生成知乎文章使用的三张技术信息图。"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


WIDTH = 1600
HEIGHT = 900
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "zhihu"

FONT_CN = "/System/Library/Fonts/Hiragino Sans GB.ttc"
FONT_BOLD = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_MONO = "/System/Library/Fonts/SFNSMono.ttf"

BG_TOP = (8, 13, 27)
BG_BOTTOM = (3, 7, 18)
CARD = (15, 24, 45)
CARD_LIGHT = (21, 34, 61)
TEXT = (240, 246, 255)
MUTED = (144, 163, 190)
CYAN = (41, 218, 255)
PURPLE = (148, 104, 255)
GREEN = (79, 231, 158)
ORANGE = (255, 176, 77)
RED = (255, 105, 125)
GRID = (31, 48, 76)


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    """载入统一字体。"""
    path = FONT_MONO if mono else FONT_BOLD if bold else FONT_CN
    return ImageFont.truetype(path, size=size)


def background() -> Image.Image:
    """生成带渐变与网格的深色背景。"""
    image = Image.new("RGB", (WIDTH, HEIGHT), BG_BOTTOM)
    draw = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        ratio = y / (HEIGHT - 1)
        color = tuple(
            round(BG_TOP[i] * (1 - ratio) + BG_BOTTOM[i] * ratio)
            for i in range(3)
        )
        draw.line((0, y, WIDTH, y), fill=color)
    for x in range(0, WIDTH, 80):
        draw.line((x, 0, x, HEIGHT), fill=GRID, width=1)
    for y in range(0, HEIGHT, 80):
        draw.line((0, y, WIDTH, y), fill=GRID, width=1)
    return image


def glow(image: Image.Image, center: tuple[int, int], color: tuple[int, int, int], radius: int) -> None:
    """叠加柔和光晕。"""
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, 105))
    layer = layer.filter(ImageFilter.GaussianBlur(radius // 2))
    image.paste(layer, (0, 0), layer)


def rounded_card(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int] = CARD,
    outline: tuple[int, int, int] = GRID,
    radius: int = 28,
    width: int = 2,
) -> None:
    """绘制带阴影的圆角卡片。"""
    x1, y1, x2, y2 = box
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rounded_rectangle(
        (x1 + 8, y1 + 14, x2 + 8, y2 + 14),
        radius=radius,
        fill=(0, 0, 0, 130),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    image.paste(shadow, (0, 0), shadow)
    ImageDraw.Draw(image).rounded_rectangle(
        box,
        radius=radius,
        fill=fill,
        outline=outline,
        width=width,
    )


def header(
    image: Image.Image,
    title: str,
    subtitle: str,
    index: str,
) -> None:
    """绘制统一页眉。"""
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((70, 58, 218, 108), radius=25, fill=(17, 52, 78), outline=CYAN, width=2)
    draw.text((144, 83), "VATES", font=font(24, bold=True), fill=CYAN, anchor="mm")
    draw.text((1460, 82), index, font=font(24, mono=True), fill=MUTED, anchor="rm")
    draw.text((70, 148), title, font=font(54, bold=True), fill=TEXT)
    draw.text((72, 224), subtitle, font=font(25), fill=MUTED)


def footer(image: Image.Image, text: str = "项目实测 · 结果因设备、模型与配置而异") -> None:
    """绘制统一脚注。"""
    draw = ImageDraw.Draw(image)
    draw.line((70, 842, 1530, 842), fill=GRID, width=2)
    draw.text((70, 862), text, font=font(18), fill=MUTED, anchor="lm")
    draw.text((1530, 862), "github.com/AMOS144/Vates", font=font(18, mono=True), fill=MUTED, anchor="rm")


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int] = CYAN,
    width: int = 5,
) -> None:
    """绘制直线箭头。"""
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2 - 18, y2), fill=color, width=width)
    draw.polygon(((x2, y2), (x2 - 22, y2 - 12), (x2 - 22, y2 + 12)), fill=color)


def metric(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    value: str,
    label: str,
    color: tuple[int, int, int],
    note: str = "",
) -> None:
    """绘制居中的指标。"""
    x, y = center
    draw.text((x, y), value, font=font(72, bold=True), fill=color, anchor="mm")
    draw.text((x, y + 70), label, font=font(24, bold=True), fill=TEXT, anchor="mm")
    if note:
        draw.text((x, y + 110), note, font=font(19), fill=MUTED, anchor="mm")


def draw_ssd(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    """绘制简洁 SSD 图标。"""
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=24, fill=(12, 31, 49), outline=CYAN, width=3)
    draw.ellipse((x1 + 34, y1 + 34, x2 - 34, y2 - 34), outline=(62, 110, 142), width=4)
    draw.ellipse((x1 + 68, y1 + 68, x2 - 68, y2 - 68), outline=CYAN, width=4)
    draw.ellipse((x1 + 92, y1 + 92, x2 - 92, y2 - 92), fill=CYAN)
    draw.rounded_rectangle((x2 - 58, y2 - 36, x2 - 20, y2 - 20), radius=7, fill=GREEN)


def memory_wall() -> None:
    """生成首屏内存结果卡。"""
    image = background()
    glow(image, (800, 510), CYAN, 330)
    header(
        image,
        "41GB 的 80B MoE，8GB 峰值跑起来了",
        "不把全部专家塞进内存，让 512 个专家留在 SSD",
        "01 / MEMORY WALL",
    )
    draw = ImageDraw.Draw(image)

    rounded_card(image, (70, 330, 460, 740), outline=(84, 61, 106))
    rounded_card(image, (585, 330, 1015, 740), fill=(11, 27, 43), outline=CYAN)
    rounded_card(image, (1140, 330, 1530, 535), outline=(52, 93, 88))
    rounded_card(image, (1140, 560, 1530, 740), outline=(52, 93, 88))

    metric(draw, (265, 455), "≈ 41GB", "4-bit 完整权重", PURPLE, "Qwen3-Next-80B-A3B")
    draw_ssd(draw, (680, 380, 920, 620))
    draw.text((800, 666), "SSD STREAMING", font=font(28, bold=True), fill=CYAN, anchor="mm")
    draw.text((800, 702), "专家按需读取 · 跨层预测预取", font=font(19), fill=MUTED, anchor="mm")
    metric(draw, (1335, 402), "≈ 8GB", "运行内存峰值", GREEN)
    metric(draw, (1335, 620), "13–15", "tok/s 生成速度", ORANGE)

    arrow(draw, (475, 535), (565, 535), PURPLE)
    arrow(draw, (1035, 535), (1120, 535), GREEN)
    footer(image)
    image.save(OUTPUT / "01-vates-memory-wall.png", optimize=True)


def process_node(
    image: Image.Image,
    box: tuple[int, int, int, int],
    step: str,
    title: str,
    detail: str,
    color: tuple[int, int, int],
) -> None:
    """绘制预取流程节点。"""
    rounded_card(image, box, fill=CARD_LIGHT, outline=color, radius=22)
    draw = ImageDraw.Draw(image)
    x1, y1, x2, _ = box
    draw.rounded_rectangle((x1 + 18, y1 + 18, x1 + 66, y1 + 56), radius=18, fill=color)
    draw.text((x1 + 42, y1 + 37), step, font=font(17, bold=True), fill=BG_BOTTOM, anchor="mm")
    draw.text(((x1 + x2) // 2, y1 + 84), title, font=font(23, bold=True), fill=TEXT, anchor="mm")
    draw.multiline_text(
        ((x1 + x2) // 2, y1 + 125),
        detail,
        font=font(17),
        fill=MUTED,
        anchor="ma",
        align="center",
        spacing=8,
    )


def cross_layer_prefetch() -> None:
    """生成跨层预取流程图。"""
    image = background()
    glow(image, (900, 520), PURPLE, 360)
    header(
        image,
        "预取不是盲猜：提前运行未来层的 gate",
        "预测要宽，读取要克制；把 SSD 延迟藏进中间层计算",
        "02 / PREFETCH",
    )
    draw = ImageDraw.Draw(image)

    boxes = [
        (55, 345, 275, 555),
        (315, 345, 535, 555),
        (575, 345, 795, 555),
        (835, 345, 1055, 555),
        (1095, 345, 1315, 555),
        (1355, 345, 1575, 555),
    ]
    nodes = [
        ("L", "当前层 x", "attention 后\n较新隐藏状态", CYAN),
        ("1", "未来层 gate", "提前看 L+a 层\n无需额外模型", PURPLE),
        ("2", "24 个候选", "MTP 多 token\n按 max 聚合", PURPLE),
        ("3", "过滤缺口", "剔除已常驻\n受 budget 限制", ORANGE),
        ("4", "C++ pread", "GPU 回调触发\n主线程零同步", CYAN),
        ("5", "侧区命中", "真实区 ∪ 侧区\nmiss 走 demand", GREEN),
    ]
    for box, node in zip(boxes, nodes):
        process_node(image, box, *node)
    for left, right in zip(boxes, boxes[1:]):
        arrow(draw, (left[2] + 8, 450), (right[0] - 8, 450), CYAN, width=4)

    rounded_card(image, (120, 640, 1480, 785), fill=(10, 22, 39), outline=(46, 69, 98), radius=22)
    draw.text((165, 678), "时间窗口", font=font(19, bold=True), fill=MUTED)
    draw.rounded_rectangle((165, 716, 655, 750), radius=17, fill=(25, 65, 92))
    draw.text((410, 733), "早层 ahead = 1 · 保预测召回", font=font(18, bold=True), fill=CYAN, anchor="mm")
    draw.rounded_rectangle((690, 716, 1180, 750), radius=17, fill=(57, 42, 91))
    draw.text((935, 733), "后层 ahead = 3 · 抢 I/O 时序", font=font(18, bold=True), fill=(190, 164, 255), anchor="mm")
    draw.text((1430, 733), "COMPUTE ∥ I/O", font=font(19, bold=True, mono=True), fill=GREEN, anchor="rm")

    footer(image)
    image.save(OUTPUT / "02-vates-cross-layer-prefetch.png", optimize=True)


def comparison_bar(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    ratio: float,
    color: tuple[int, int, int],
) -> None:
    """绘制指标对比条。"""
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=(y2 - y1) // 2, fill=(27, 42, 66))
    draw.rounded_rectangle(
        (x1, y1, x1 + max(20, int((x2 - x1) * ratio)), y2),
        radius=(y2 - y1) // 2,
        fill=color,
    )


def benchmark_card(
    image: Image.Image,
    box: tuple[int, int, int, int],
    tag: str,
    title: str,
    before: str,
    after: str,
    note: str,
    ratio_before: float,
    ratio_after: float,
    color: tuple[int, int, int],
) -> None:
    """绘制单组消融数据卡。"""
    rounded_card(image, box, fill=CARD, outline=color, radius=28)
    draw = ImageDraw.Draw(image)
    x1, y1, x2, _ = box
    draw.rounded_rectangle((x1 + 28, y1 + 28, x1 + 118, y1 + 68), radius=20, fill=color)
    draw.text((x1 + 73, y1 + 48), tag, font=font(16, bold=True), fill=BG_BOTTOM, anchor="mm")
    draw.text((x1 + 30, y1 + 104), title, font=font(28, bold=True), fill=TEXT)
    draw.text((x1 + 30, y1 + 164), before, font=font(31, bold=True), fill=MUTED)
    draw.text((x1 + 192, y1 + 164), "→", font=font(30, bold=True), fill=color)
    draw.text((x1 + 250, y1 + 164), after, font=font(39, bold=True), fill=color)
    comparison_bar(draw, (x1 + 30, y1 + 220, x2 - 30, y1 + 245), ratio_before, (68, 83, 108))
    comparison_bar(draw, (x1 + 30, y1 + 267, x2 - 30, y1 + 292), ratio_after, color)
    draw.multiline_text(
        (x1 + 30, y1 + 334),
        note,
        font=font(18),
        fill=MUTED,
        spacing=8,
    )


def benchmark_results() -> None:
    """生成三组关键实验结果卡。"""
    image = background()
    glow(image, (800, 500), GREEN, 350)
    header(
        image,
        "不是一组总分：三条数据路径，三次独立消融",
        "KV 占用、缓存命中与 C++ 热路径分别测量，收益不可直接相加",
        "03 / ABLATION",
    )

    benchmark_card(
        image,
        (70, 330, 530, 770),
        "KV",
        "128k 上下文 KV",
        "3.0 GiB",
        "0.68 GiB",
        "IsoQuant K4/V3 + SO(4)\n仅作用于 12 个全注意力层",
        1.0,
        0.227,
        GREEN,
    )
    benchmark_card(
        image,
        (570, 330, 1030, 770),
        "LFU",
        "侧区缓存命中率",
        "0.76",
        "0.81",
        "持久 LFU 保留近期高频专家\n避免每步清空侧区",
        0.76,
        0.81,
        CYAN,
    )
    benchmark_card(
        image,
        (1070, 330, 1530, 770),
        "C++",
        "统一池吞吐",
        "13.70",
        "14.80",
        "槽状态、驱逐与 I/O 下沉\n单位：tok/s",
        13.70 / 16,
        14.80 / 16,
        PURPLE,
    )
    footer(image, "独立消融结果 · 测试条件见 benchmarks/reports")
    image.save(OUTPUT / "03-vates-benchmark-results.png", optimize=True)


def main() -> None:
    """生成全部图片。"""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    memory_wall()
    cross_layer_prefetch()
    benchmark_results()
    for path in sorted(OUTPUT.glob("*.png")):
        print(f"generated {path.relative_to(ROOT.parent.parent.parent)}")


if __name__ == "__main__":
    main()
