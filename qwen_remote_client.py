# =========================
# qwen_remote_client.py
# 远程 Qwen 分类模块
# 功能：
# 1. 连接远程 vLLM / OpenAI-compatible 接口
# 2. 自动解析远端模型名
# 3. 把本地图片转成 base64 后发送给远程模型
# 4. 让模型返回 front / back 二分类结果
# 5. 支持直接作为模块导入，也支持单独运行测试
# =========================

# =========================
# 标准库 / 第三方库导入
# =========================
import os
import re
import time
import base64
import argparse
import requests


# ==========================================================
# Qwen 远程分类默认配置
# 说明：
# 这些参数可以在主程序中通过 set_remote_config() 动态覆盖
# ==========================================================

# 远程模型名
# - 如果写 "auto"，程序会自动访问 /v1/models 获取模型名
# - 如果写具体模型名，则直接使用该名称
MODEL_NAME = "auto"

# 是否打印模型原始输出
PRINT_RAW_RESPONSE = True

# 远程 OpenAI-compatible API 根地址
REMOTE_API_BASE = "http://127.0.0.1:8000/v1"

# 聊天补全接口地址
REMOTE_CHAT_URL = f"{REMOTE_API_BASE.rstrip('/')}/chat/completions"

# 模型列表接口地址
REMOTE_MODELS_URL = f"{REMOTE_API_BASE.rstrip('/')}/models"

# API Key
# 如果服务端没有鉴权，也可以保留为 "EMPTY"
REMOTE_API_KEY = "EMPTY"

# HTTP 请求超时时间（秒）
REMOTE_TIMEOUT = 120


# ==========================================================
# 提示词
# 说明：
# 要求模型严格输出 front 或 back
# ==========================================================
PROMPT = """
你在做工业零件正反面二分类任务。

判别规则如下：
1. 主要大表面有凸起的小圆柱形定位销或者有刻字是 front
2. 主要大表面有多个小孔洞并且没有文字的是 back
你的任务：
判断输入图片中的零件是 front 还是 back。

输出要求：
1. 只能输出 front 或 back
2. 不要输出解释
3. 不要输出标点
4. 不要输出其他任何内容
5.每一次判断都忘掉上一次的答案
""".strip()


# ==========================================================
# 动态修改远程接口配置
# 说明：
# 主程序导入本模块后，可以通过这个函数覆盖默认配置
# ==========================================================
def set_remote_config(api_base=None,
                      timeout=None,
                      model_name=None,
                      api_key=None,
                      print_raw_response=None):
    """
    在主程序中动态更新远程配置

    参数：
    - api_base: 远程 API 根地址，例如 http://127.0.0.1:8000/v1
    - timeout: 超时时间（秒）
    - model_name: 模型名；auto 表示自动解析
    - api_key: 鉴权 key
    - print_raw_response: 是否打印模型原始输出
    """
    global REMOTE_API_BASE, REMOTE_CHAT_URL, REMOTE_MODELS_URL
    global REMOTE_TIMEOUT, MODEL_NAME, REMOTE_API_KEY, PRINT_RAW_RESPONSE

    # 更新远程根地址，同时联动更新两个具体接口地址
    if api_base is not None:
        REMOTE_API_BASE = api_base.rstrip("/")
        REMOTE_CHAT_URL = f"{REMOTE_API_BASE}/chat/completions"
        REMOTE_MODELS_URL = f"{REMOTE_API_BASE}/models"

    # 更新超时时间
    if timeout is not None:
        REMOTE_TIMEOUT = timeout

    # 更新模型名
    if model_name is not None:
        MODEL_NAME = model_name

    # 更新 API Key
    if api_key is not None:
        REMOTE_API_KEY = api_key

    # 更新是否打印原始输出
    if print_raw_response is not None:
        PRINT_RAW_RESPONSE = print_raw_response


# ==========================================================
# 将模型输出规范化为 front / back / unknown
# 说明：
# 有些模型可能输出：
# - "front"
# - "back"
# - "The answer is front"
# 所以这里统一归一化
# ==========================================================
def normalize_label(text: str) -> str:
    """
    将模型输出归一化为 front / back / unknown
    """
    # 空字符串直接视为 unknown
    if not text:
        return "unknown"

    # 去首尾空格、统一转小写
    s = text.strip().lower()

    # 合并多个空白字符
    s = re.sub(r"\s+", " ", s)

    # 精确匹配
    if s == "front":
        return "front"
    if s == "back":
        return "back"

    # 模糊包含匹配
    if "front" in s:
        return "front"
    if "back" in s:
        return "back"

    # 其余情况视为 unknown
    return "unknown"


# ==========================================================
# 将本地图片文件转为 base64 字符串
# 说明：
# 远程接口采用 data:image/png;base64,... 的方式传图
# ==========================================================
def image_file_to_base64(image_path: str) -> str:
    """
    读取图片并转为 base64 字符串
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ==========================================================
# 构造远程 HTTP 请求头
# ==========================================================
def get_remote_headers() -> dict:
    """
    构造远程 API 请求头
    """
    headers = {"Content-Type": "application/json"}

    # 如果设置了 API key，则添加 Authorization
    if REMOTE_API_KEY:
        headers["Authorization"] = f"Bearer {REMOTE_API_KEY}"

    return headers


# ==========================================================
# 自动解析远端模型名
# 说明：
# 当 MODEL_NAME = "auto" 时，
# 访问 /v1/models 并取第一个模型的 id 作为实际模型名
# ==========================================================
def resolve_remote_model_name(force_refresh: bool = False) -> str:
    """
    自动解析远端当前暴露的模型名

    参数：
    - force_refresh:
        False: 若当前 MODEL_NAME 不是 auto，则直接返回
        True : 强制重新访问远端 /models 获取模型名
    """
    global MODEL_NAME

    # 如果已经指定了明确模型名，且不要求强制刷新，则直接返回
    if not force_refresh and MODEL_NAME and MODEL_NAME.lower() != "auto":
        return MODEL_NAME

    try:
        # 访问远端模型列表接口
        resp = requests.get(
            REMOTE_MODELS_URL,
            headers=get_remote_headers(),
            timeout=REMOTE_TIMEOUT
        )
        resp.raise_for_status()

        # 解析 JSON
        data = resp.json()
        models = data.get("data", [])

        # 若没有返回模型，报错
        if not models:
            raise RuntimeError("远端 /models 返回为空")

        # 默认取第一个模型 id
        model_id = models[0].get("id", "")
        if not model_id:
            raise RuntimeError("远端 /models 未返回有效模型名")

        # 更新全局模型名
        MODEL_NAME = model_id
        print(f"[REMOTE] 自动解析模型名成功: {MODEL_NAME}")
        return MODEL_NAME

    except Exception as e:
        raise RuntimeError(f"自动解析远端模型名失败: {e}")


# ==========================================================
# 对单张图片进行 front / back 远程分类
# 说明：
# 这里走的是 OpenAI-compatible /v1/chat/completions 接口
# ==========================================================
def classify_one_image(image_path: str):
    """
    对单张图片进行远程 front/back 分类

    参数：
    - image_path: 本地图片路径

    返回：
    - "front"
    - "back"
    - "unknown"
    - 或异常时返回 None
    """
    # 先检查文件是否存在
    if not os.path.exists(image_path):
        print(f"[ERROR] 文件不存在: {image_path}")
        return None

    try:
        # 获取当前实际使用的模型名
        model_name = resolve_remote_model_name()

        # 将图片转为 base64
        image_b64 = image_file_to_base64(image_path)

        # 构造 OpenAI-compatible 请求体
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": PROMPT
                        }
                    ]
                }
            ],
            # 低温度，尽量稳定输出
            "temperature": 0.0,
            "top_p": 0.1,
            # 输出 token 尽量少，因为只需要 front / back
            "max_tokens": 8,
            # 关闭 thinking
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }

        # 发请求并计时
        t0 = time.time()
        resp = requests.post(
            REMOTE_CHAT_URL,
            headers=get_remote_headers(),
            json=payload,
            timeout=REMOTE_TIMEOUT
        )
        resp.raise_for_status()
        response = resp.json()
        t1 = time.time()
        wall_time = t1 - t0

        # 提取模型原始输出
        raw_text = ""
        if "choices" in response and len(response["choices"]) > 0:
            raw_text = response["choices"][0].get("message", {}).get("content", "")

            # 有些接口 content 可能返回 list，需要拼接
            if isinstance(raw_text, list):
                raw_text = " ".join([
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in raw_text
                ])

            raw_text = str(raw_text).strip()

        # 归一化结果
        label = normalize_label(raw_text)

        # 提取 token 使用信息
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # 打印原始输出
        if PRINT_RAW_RESPONSE:
            print(f"[RAW] {os.path.basename(image_path)} -> {raw_text}")

        # 打印最终结果和耗时
        print(f"[RESULT] {os.path.basename(image_path)} -> {label}")
        print("[TIME DETAIL]")
        print(f"  remote_api_base      : {REMOTE_API_BASE}")
        print(f"  remote_model         : {model_name}")
        print(f"  wall_time            : {wall_time:.4f} s")
        print(f"  prompt_tokens        : {prompt_tokens}")
        print(f"  completion_tokens    : {completion_tokens}")
        print(f"  total_tokens         : {total_tokens}")

        return label

    except Exception as e:
        print(f"[ERROR] 远程识别失败: {image_path}")
        print(f"[ERROR] remote_api_base={REMOTE_API_BASE}")
        print(f"[ERROR] {e}")
        return None


# ==========================================================
# 主程序测试入口
# ==========================================================
# 主程序测试入口（PyCharm 直接运行版）
# 说明：
# 不再使用命令行参数，而是在这里直接写测试参数
# 这样可以直接在 PyCharm 里点运行
# ==========================================================
if __name__ == "__main__":
    # =========================
    # 1. 在这里直接写测试参数
    # =========================

    # 待测试图片路径
    test_image = "/home/ma/test.png"

    # 远程 OpenAI-compatible API 地址
    remote_api_base = "http://127.0.0.1:8000/v1"

    # 远程接口超时时间（秒）
    remote_timeout = 120

    # 模型名称
    # - 写 "auto"：自动从 /v1/models 获取
    # - 写具体模型名：直接使用该模型
    remote_model_name = "auto"

    # API Key
    # 如果服务端未开启鉴权，可保持 "EMPTY"
    remote_api_key = "EMPTY"

    # 是否打印模型原始输出
    print_raw = True

    # =========================
    # 2. 用上面这些参数覆盖默认配置
    # =========================
    set_remote_config(
        api_base=remote_api_base,
        timeout=remote_timeout,
        model_name=remote_model_name,
        api_key=remote_api_key,
        print_raw_response=print_raw
    )

    # =========================
    # 3. 打印当前配置
    # =========================
    print("=" * 60)
    print("[TEST] 当前远程配置")
    print(f"REMOTE_API_BASE   = {REMOTE_API_BASE}")
    print(f"REMOTE_CHAT_URL   = {REMOTE_CHAT_URL}")
    print(f"REMOTE_MODELS_URL = {REMOTE_MODELS_URL}")
    print(f"REMOTE_TIMEOUT    = {REMOTE_TIMEOUT}")
    print(f"MODEL_NAME        = {MODEL_NAME}")
    print(f"TEST_IMAGE        = {test_image}")
    print("=" * 60)

    # =========================
    # 4. 检查测试图片是否存在
    # =========================
    if not os.path.exists(test_image):
        raise FileNotFoundError(f"测试图片不存在: {test_image}")

    # =========================
    # 5. 先尝试探测远端模型
    # =========================
    try:
        resolve_remote_model_name(force_refresh=(MODEL_NAME.lower() == "auto"))
    except Exception as e:
        print(f"[WARN] 远端模型探测失败：{e}")

    # =========================
    # 6. 执行远程分类测试
    # =========================
    result = classify_one_image(test_image)

    # =========================
    # 7. 输出最终结果
    # =========================
    print("=" * 60)
    print(f"[FINAL RESULT] {result}")
    print("=" * 60)