import requests

def fetch_all_titles(limit=100):
    """
    使用 Wikipedia API 提取页面标题。
    :param limit: 每次请求的页面数量（最大值 500）。
    :return: 所有页面的标题列表。
    """
    base_url = "https://en.wikipedia.org/w/api.php"
    all_titles = []
    aplcontinue = None  # 用于分页

    while True:
        # 构造 API 请求
        params = {
            "action": "query",
            "list": "allpages",
            "format": "json",
            "aplimit": limit,  # 每次返回的页面数量
        }
        if aplcontinue:
            params["apcontinue"] = aplcontinue

        # 发送请求
        response = requests.get(base_url, params=params)
        data = response.json()

        # 提取标题
        pages = data.get("query", {}).get("allpages", [])
        for page in pages:
            all_titles.append(page["title"])
        print(page["title"])
        # 检查是否有下一页
        aplcontinue = data.get("continue", {}).get("apcontinue")
        if not aplcontinue:
            break

    return all_titles


# 调用函数
if __name__ == "__main__":
    titles = fetch_all_titles(limit=100)
    print(f"共提取到 {len(titles)} 个页面标题：")
    print(titles[:10])  # 打印前 10 个标题