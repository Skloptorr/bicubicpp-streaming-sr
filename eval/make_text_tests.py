r"""Black-and-white text fixtures: the easiest way to see SR go wrong.

Text is the harshest test for a super-resolution network. Every stroke edge is a
step function with a known correct shape, strokes are thin relative to the
upscale factor, and CJK glyphs pack many high-contrast edges into a few pixels.
Bicubic smears them predictably; a network can either restore them or invent them,
and at 1:1 the difference is immediate.

Three fixtures, each rendered crisply at 2560x1440 and then upscaled to 3840x2160:

    1  cjk      Chinese at 44 down to 11 px, black on white
    2  code     monospace on black, 30 down to 11 px
    3  dense    mixed CJK + Latin table, 20 down to 10 px

Everything is pure greyscale, so any colour in the output is colour the network
invented -- which is exactly the artefact that shows up in mixed foliage.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 2560, 1440
OUT = os.environ.get("SR_OUT", "out/text_tests")

MSYH = os.environ.get("FONT_CJK", "msyh.ttc")
MSYHBD = os.environ.get("FONT_CJK_BOLD", "msyhbd.ttc")
CONSOLA = os.environ.get("FONT_MONO", "consola.ttf")
ARIAL = os.environ.get("FONT_SANS", "arial.ttf")


def font(path, size):
    return ImageFont.truetype(path, size)


def cjk_page():
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    y = 24
    lines = [
        ("超分辨率重建质量测试", MSYHBD, 44),
        ("黑字白底，纯灰阶，无任何彩色信息", MSYH, 30),
        ("倍率 1.5x：2560x1440 -> 3840x2160", MSYH, 22),
        ("", MSYH, 12),
        ("这一行是 26 像素：秋水共长天一色，落霞与孤鹜齐飞。", MSYH, 26),
        ("这一行是 22 像素：渔舟唱晚，响穷彭蠡之滨；雁阵惊寒，声断衡阳之浦。", MSYH, 22),
        ("这一行是 18 像素：遥襟甫畅，逸兴遄飞。爽籁发而清风生，纤歌凝而白云遏。", MSYH, 18),
        ("这一行是 16 像素：睢园绿竹，气凌彭泽之樽；邺水朱华，光照临川之笔。", MSYH, 16),
        ("这一行是 14 像素：四美具，二难并。穷睇眄于中天，极娱游于暇日。", MSYH, 14),
        ("这一行是 13 像素：天高地迥，觉宇宙之无穷；兴尽悲来，识盈虚之有数。", MSYH, 13),
        ("这一行是 12 像素：望长安于日下，目吴会于云间。地势极而南溟深。", MSYH, 12),
        ("这一行是 11 像素：怀帝阍而不见，奉宣室以何年。嗟乎时运不齐命途多舛。", MSYH, 11),
        ("", MSYH, 12),
        ("混合：ABCdef 0123456789 ！？，。、《》「」『』（）【】", MSYH, 15),
        ("混合：The quick brown fox jumps over the lazy dog 0123456789", ARIAL, 15),
        ("细体 20 像素：冯唐易老，李广难封。屈贾谊于长沙，非无圣主。", os.environ.get("FONT_CJK_LIGHT", "msyhl.ttc"), 20),
        ("粗体 20 像素：冯唐易老，李广难封。屈贾谊于长沙，非无圣主。", MSYHBD, 20),
        ("", MSYH, 12),
        ("笔画密集：鬱鬻爨龘齉靐齾爩麤龗灪龖厵爨癵驫麣纞虋讟钃", MSYH, 17),
        ("笔画密集：魑魅魍魉饕餮耄耋囹圄蹉跎魃魈魁鬾鬿魕魖魓魒魐魀", MSYH, 13),
    ]
    for text, f, size in lines:
        if text:
            d.text((28, y), text, font=font(f, size), fill=(0, 0, 0))
        y += int(size * 1.55)
    return img


def code_page():
    img = Image.new("RGB", (W, H), (16, 16, 16))
    d = ImageDraw.Draw(img)
    y = 24
    lines = [
        ("# 等宽字体，白字黑底，纯灰阶", CONSOLA, 28),
        ("", CONSOLA, 14),
        ("def super_resolve(lr, sr_rate=1.5):", CONSOLA, 24),
        ('    """Upscale lr by sr_rate with Bicubic++."""', CONSOLA, 20),
        ("    x = leaky(conv2(leaky(conv1(leaky(conv0(lr))))))", CONSOLA, 18),
        ("    return pixel_shuffle(conv_out(x) + residual, 2 * sr_rate)", CONSOLA, 16),
        ("", CONSOLA, 12),
        ("# 14px -- 这一档开始糊", CONSOLA, 14),
        ("0123456789 ABCDEFGHIJ abcdefghij {}[]()<>=+-*/%", CONSOLA, 14),
        ("if (e.g > 0.0) { weights.rg = float2(d1, d2); } else { weights = 0; }", CONSOLA, 14),
        ("", CONSOLA, 10),
        ("# 12px", CONSOLA, 12),
        ("for (int j = 0; j < 4; ++j) { acc += wy[j] * row; } // 0123456789", CONSOLA, 12),
        ("#define SMAA_MAX_SEARCH_STEPS 8 // AAAAAAAAAAAAAAAAAAAA", CONSOLA, 12),
        ("", CONSOLA, 10),
        ("# 11px -- 最小可读档", CONSOLA, 11),
        ("kernel = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]] * strength", CONSOLA, 11),
        ("mmmmmmmmmm iiiiiiiiii llllllllll 1111111111 0000000000", CONSOLA, 11),
        ("MWMWMWMWMW AWMWMWMWMW | | | | | | | | | | | | | | | |", CONSOLA, 11),
        ("", CONSOLA, 10),
        ("# 竖线密集（最容易被误补）", CONSOLA, 12),
        ("||||||||||||||||||||||||||||||||||||||||||||||||", CONSOLA, 13),
        ("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII", CONSOLA, 13),
    ]
    for text, f, size in lines:
        if text:
            d.text((26, y), text, font=font(f, size), fill=(255, 255, 255))
        y += int(size * 1.5)
    return img


def dense_page():
    """A small-font table: the case where interpolation and invention differ most."""
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((28, 20), "密集小字表格  |  纯灰阶  |  2560x1440", font=font(MSYHBD, 26), fill=(0, 0, 0))
    rows = [
        ("模型", "通道", "PSNR-Y", "中位增益", "HF/GT", "备注"),
        ("bicubic", "-", "33.479", "0.000", "67%", "双三次插值"),
        ("MSE c=32", "32", "34.618", "+0.708", "85%", "相似度损失"),
        ("MSE c=48", "48", "36.848", "+1.006", "91%", "当前发布版"),
        ("感知 loss", "32", "34.245", "+0.572", "78%", "反而更软"),
        ("梯度 loss", "32", "34.502", "+0.666", "76%", "反而更软"),
        ("官方 x3", "32", "34.890", "+1.102", "93%", "论文权重"),
    ]
    x0, y = 28, 74
    cols = [150, 90, 190, 170, 150, 260]
    fs, fh = font(MSYH, 17), font(MSYHBD, 17)
    for j, h in enumerate(rows[0]):
        d.text((x0 + sum(cols[:j]), y), h, font=fh, fill=(0, 0, 0))
    y += 30
    d.line([(x0, y - 4), (x0 + sum(cols), y - 4)], fill=(0, 0, 0), width=1)
    for r in rows[1:]:
        for j, c in enumerate(r):
            d.text((x0 + sum(cols[:j]), y), c, font=fs, fill=(0, 0, 0))
        y += 26
    y += 24
    d.text((28, y), "段落 13px：超分辨率网络在小字号上最容易暴露问题，因为一个字的笔画只有"
                   "两三个像素宽，任何过冲、振铃、错误连通都会直接改变字形。",
           font=font(MSYH, 13), fill=(0, 0, 0))
    y += 30
    d.text((28, y), "Paragraph 11px: at this size each stem is roughly one LR pixel wide, so the "
                   "network has to decide whether a stroke exists at all.",
           font=font(ARIAL, 11), fill=(0, 0, 0))
    y += 26
    d.text((28, y), "粗体 12px：重影、断笔、粘连是三种典型失败形态。",
           font=font(MSYHBD, 12), fill=(0, 0, 0))
    y += 24
    d.text((28, y), "常规 12px：重影、断笔、粘连是三种典型失败形态。",
           font=font(MSYH, 12), fill=(0, 0, 0))
    y += 24
    d.text((28, y), "细体 12px：重影、断笔、粘连是三种典型失败形态。",
           font=font(os.environ.get("FONT_CJK_LIGHT", "msyhl.ttc"), 12), fill=(0, 0, 0))
    # 右下角放一组极小的数字/符号
    y += 40
    d.text((28, y), "0123456789 ！？，。、；：“”‘’（）【】《》 …—－", font=font(MSYH, 12), fill=(0, 0, 0))
    y += 22
    d.text((28, y), "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz", font=font(ARIAL, 12), fill=(0, 0, 0))
    return img


def main():
    os.makedirs(OUT, exist_ok=True)
    pages = [("1_cjk", cjk_page()), ("2_code", code_page()), ("3_dense", dense_page())]
    for name, img in pages:
        p = os.path.join(OUT, "原图_%s_2560x1440.png" % name)
        img.save(p)
        a = np.asarray(img.convert("L"))
        print("  %-10s %dx%d  灰度均值 %.1f  std %.1f  -> %s"
              % (name, img.width, img.height, a.mean(), a.std(), os.path.basename(p)))
    print("wrote 3 fixtures to %s" % OUT)


if __name__ == "__main__":
    main()
