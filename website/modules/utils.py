import json

import randomcolor

COLORS = ["#CE224D", "#5CBBFF", "#FF9800", "#4CAF50", "#009432", "#CE224D", "#5CBBFF", "#FF9800", "#4CAF50", "#009432", "#CE224D", "#5CBBFF", "#FF9800", "#4CAF50", "#009432", "#CE224D", "#5CBBFF", "#FF9800", "#4CAF50", "#009432", "#CE224D", "#5CBBFF", "#FF9800", "#4CAF50", "#009432"]

def jsonize(x):
    return json.dumps(x, ensure_ascii=False)

def color_str(color):
    return "".join([str(hex(c))[2:].zfill(2) for c in color]).upper()

def color_str_opacity(color, opacity="C0"):
    return color_str(color) + opacity

def get_color():
    for c in COLORS:
        yield c