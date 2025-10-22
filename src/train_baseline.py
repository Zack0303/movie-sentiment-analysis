# src/train_baseline.py

import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib # 用于保存模型

print("=== 基线模型训练脚本启动 ===")

# --- 1. 定义路径 ---
# os.path.abspath(__file__) 获取当前脚本的绝对路径
# os.path.dirname(...) 获取路径所在的目录
# 两次 dirname 就能从 src/train_baseline.py 回到项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'IMDB Dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'baseline_model.joblib')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')

# 确保 models 文件夹存在
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 2. 加载数据 ---
print(f"加载数据从: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# --- 3. 简单文本预处理 ---
def simple_cleaner(text):
    # 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 转为小写
    text = text.lower()
    return text

print("开始预处理文本...")
df['review_cleaned'] = df['review'].apply(simple_cleaner)

# --- 4. 标签编码 ---
# 将 'positive' 转为 1, 'negative' 转为 0
df['sentiment_encoded'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# --- 5. 划分数据集 ---
print("划分训练集和测试集...")
X = df['review_cleaned']
y = df['sentiment_encoded']

# test_size=0.2 表示 20% 的数据作为测试集
# random_state=42 确保每次划分结果都一样，方便复现
# stratify=y 确保训练集和测试集中，好评/差评的比例与原始数据一致
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

# --- 6. 特征工程 (TF-IDF) ---
print("开始 TF-IDF 特征提取...")
# TfidfVectorizer 会将文本转换为数字矩阵
# max_features=5000 表示我们只关心最常见的 5000 个词
# stop_words='english' 会自动移除英文中的常见停用词 (如 'a', 'the', 'in')
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# .fit_transform() 在训练集上学习词典并转换
X_train_tfidf = vectorizer.fit_transform(X_train)
# .transform() 在测试集上只进行转换，使用训练集学到的词典
X_test_tfidf = vectorizer.transform(X_test)

print("特征提取完成。")

# --- 7. 训练模型 (逻辑回归) ---
print("开始训练逻辑回归模型...")
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train_tfidf, y_train)

print("模型训练完成。")

# --- 8. 评估模型 ---
print("开始评估模型...")
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])

print("\n--- 模型评估结果 ---")
print(f"准确率 (Accuracy): {accuracy:.4f}")
print("分类报告 (Classification Report):")
print(report)
print("---------------------\n")

# --- 9. 保存模型和 Vectorizer ---
print(f"保存模型到: {MODEL_PATH}")
joblib.dump(model, MODEL_PATH)

print(f"保存 TF-IDF Vectorizer 到: {VECTORIZER_PATH}")
joblib.dump(vectorizer, VECTORIZER_PATH)

print("=== 脚本执行完毕 ===")