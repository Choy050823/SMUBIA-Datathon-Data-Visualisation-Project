import os
import time
import pandas as pd
from tqdm import tqdm
from anthropic import Anthropic
from google.colab import userdata

# 1. 定义 24 个分类 + other
CATEGORIES = [
    "Corporate and Business Topics",
    "Labor and Employment Issues",
    "Privacy, Security, and Cyber Matters",
    "Legal and Crime Stories",
    "Government Actions and Regulations",
    "Technology and Digital Trends",
    "Environment and Climate Topics",
    "Social Issues and Activism",
    "Healthcare and Medicine",
    "Community and Cultural Events",
    "International Relations and Trade",
    "Education and Learning",
    "Consumer Topics",
    "Infrastructure and Development",
    "Energy and Resources",
    "Political Topics and Protests",
    "Media and Communication",
    "Financial Policies and Taxation",
    "Human Rights and Social Justice",
    "Science, Research, and Innovation",
    "Disaster and Crisis Management",
    "Organized Crime and Trafficking",
    "Sports, Entertainment, and Leisure",
    "Military",
    "other"
]
os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
# 2. 配置 Claude API
client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")  # 请确保设置环境变量
)

def classify_text_with_claude(text):
    """
    使用 Claude API 进行文本分类
    返回最匹配的类别或 'other'
    """
    prompt = (
        "请从以下 24 个类别中选择最合适的一个类别，如果都不合适就选择 'other'。\n"
        "类别列表：\n" + "\n".join(CATEGORIES) + "\n\n"
        f"文本：{text}\n"
        "请仅返回最合适的类别名称，不要添加任何其他解释或标点符号："
    )

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )

        category_text = response.content[0].text.strip()

        # 验证返回的类别是否在预定义列表中
        if category_text in CATEGORIES:
            return category_text
        return "other"

    except Exception as e:
        print(f"分类过程中出现错误: {str(e)}")
        return "other"

# 3. 读取并处理 CSV 文件
def process_csv(input_path, output_path):
    # 读取CSV文件
    df = pd.read_csv(input_path)

    # 计算需要处理的行数
    rows_to_process = len(df) - 400

    # 创建进度条
    with tqdm(total=rows_to_process, desc="正在分类") as pbar:
        for i in range(400, len(df)):
            text_content = df.at[i, "Text"]

            # 获取分类结果
            predicted_category = classify_text_with_claude(text_content)

            # 更新DataFrame
            df.at[i, "Theme"] = predicted_category

            # 更新进度条
            pbar.update(1)

            # 添加延时以符合API限制
            time.sleep(1.2)  # 每分钟50个请求，预留一些余量

    # 保存结果
    df.to_csv(output_path, index=False)
    print(f"分类完成，新文件已保存到 {output_path}")

if __name__ == "__main__":
    input_path = "./Main/Important_Data/input.csv"  # 请根据实际文件路径修改
    output_path = "./Main/Important_Data/output_classified.csv"
    process_csv(input_path, output_path)