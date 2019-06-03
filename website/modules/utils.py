import json

import randomcolor

COLORS_HEX = ["#CE224D", "#5CBBFF", "#FF9800", "#4CAF50", "#009432", "#CE224D", "#5CBBFF", "#FF9800", "#4CAF50", "#009432", "#CE224D", "#5CBBFF", "#FF9800", "#4CAF50", "#009432", "#CE224D", "#5CBBFF", "#FF9800", "#4CAF50", "#009432", "#CE224D", "#5CBBFF", "#FF9800", "#4CAF50", "#009432"]
COLORS_RGB = [(206, 34, 77), (206, 34, 77), (206, 34, 77), (206, 34, 77)]

def jsonize(x):
    return json.dumps(x, ensure_ascii=False)

def color_str(color):
    return "".join([str(hex(c))[2:].zfill(2) for c in color]).upper()

def color_str_opacity(color, opacity="C0"):
    return color_str(color) + opacity

def get_hex_color():
    for c in COLORS_HEX:
        yield c

def get_pos_neg_color(cls, hex=False):
    if cls == "POSITIVE":
        return (206, 34, 77) if not hex else "#CE224D"
    else:
        return (36, 73, 100) if not hex else "#5CBBFF"

def get_rgb_color(tag):
    for c in COLORS_RGB:
        yield c