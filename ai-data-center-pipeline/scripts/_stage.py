# -*- coding: utf-8 -*-
"""建设状态分类——单一来源。

04_stages.py 与 06_audit.py 都从这里导入，避免两处规则漂移
（曾经一次出现 04 判 19 土建 + 2 前期、06 判 20 土建 + 1 前期的不一致）。

粗分类（4 档，可完全由数据推出）：
    已投运（达产） / 部分投运 / 在建 / 前期
细分类（5 档，需要读状态文本）：
    在建（土建阶段） / 在建（机电安装）
"""
import re

H100 = "H100 equivalents"
ITMW = "IT power (MW)"
OPS = "Buildings operational"

# 施工类关键词（出现即视为已开工）
BUILD_KW = ["construct", "foundation", "steel", "roof", "shell", "chiller",
            "cooling", "generator", "building", "permit", "land clearing",
            "excavat", "concrete", "grading", "earthwork", "pad"]

# 只有场地准备、尚未进入建筑本体的描述
EARLY_RE = re.compile(
    r"^\s*land clearing|land clearing begins|land clearing continues|"
    r"first signs of development|land clearing has begun|land clearing for",
    re.I)

# 机电安装/调试证据（比"土建"更靠后一档）
MEP_KW = ["chiller", "cooling tower", "cooling install", "cooling looks complete",
          "cooling is at an advanced stage", "installing cooling",
          "generator install", "generators are", "backup generator",
          "rooftop unit", "fit-out", "commissioning", "energized",
          "power equipment", "substation"]


def classify(act, plan):
    """4 档粗分类（纯数值优先，文本仅用于区分'在建'与'前期'）。"""
    if len(act) == 0:
        return "无观测"
    a = act[H100].max()
    if a > 0:
        return "已投运（达产）" if a >= 0.999 * plan else "部分投运"
    if act[ITMW].max() > 0 or act[OPS].max() > 0:
        return "在建（已有机电/建筑）"
    last = str(act["Construction status"].iloc[-1]).lower()
    if any(k in last for k in BUILD_KW) and not EARLY_RE.search(last):
        return "在建（土建阶段）"
    return "前期/土地平整"


def classify_fine(act, plan):
    """5 档细分类：把'在建'拆成土建与机电安装两档。"""
    coarse = classify(act, plan)
    if coarse != "在建（土建阶段）":
        return coarse
    text = str(act["Construction status"].iloc[-1]).lower()
    if any(k in text for k in MEP_KW):
        return "在建（机电安装）"
    return "在建（土建阶段）"


def mep_evidence(act):
    """返回命中的机电安装关键词，便于人工复核。"""
    if len(act) == 0:
        return ""
    text = str(act["Construction status"].iloc[-1]).lower()
    return "|".join([k for k in MEP_KW if k in text])
