# -*- coding:utf-8 -*-
"""
SVM垃圾邮件分类实验，自动在线加载Spambase数据集，无需手动下载
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import requests
from io import StringIO

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

def load_spambase_online():
    # 在线读取UCI数据集，不用下载文件
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    resp = requests.get(url)
    data_text = StringIO(resp.text)
    col_names = [f"feat_{i}" for i in range(57)] + ["label"]
    df = pd.read_csv(data_text, names=col_names)
    return df

def main():
    print("正在在线加载spambase数据集……")
    df = load_spambase_online()
    print(f"数据集总样本：{df.shape[0]}，特征数：{df.shape[1]-1}")
    print(f"垃圾邮件占比：{df['label'].mean():.2%}\n")

    X = df.drop("label", axis=1)
    y = df["label"]

    # 划分训练测试集 + SVM必须标准化
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 线性SVM
    print("===== 线性核SVM C=1 =====")
    svm_linear = LinearSVC(C=1, max_iter=20000, random_state=42)
    svm_linear.fit(X_train_scaled, y_train)
    y_pred_linear = svm_linear.predict(X_test_scaled)
    print(f"测试集准确率：{accuracy_score(y_test, y_pred_linear):.4f}")
    print(classification_report(y_test, y_pred_linear))

    # RBF网格搜索调参
    print("\n===== RBF高斯核SVM 网格搜索调参 =====")
    param_grid = {"C":[0.1,1,10], "gamma":[0.001,0.01,0.1]}
    svc_rbf = SVC(kernel="rbf")
    grid = GridSearchCV(svc_rbf, param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    print(f"最优参数：{grid.best_params_}")
    print(f"交叉验证得分：{grid.best_score_:.4f}")
    y_pred_rbf = grid.predict(X_test_scaled)
    print(f"测试集准确率：{accuracy_score(y_test, y_pred_rbf):.4f}")
    print(classification_report(y_test, y_pred_rbf))

    # 混淆矩阵图
    cm = confusion_matrix(y_test, y_pred_rbf)
    plt.figure(figsize=(5,4))
    plt.matshow(cm, fignum=1, cmap="Blues")
    plt.title("RBF‑SVM混淆矩阵(垃圾邮件分类)", pad=18)
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=13)
    plt.xlabel("预测标签(0正常 1垃圾)")
    plt.ylabel("真实标签(0正常 1垃圾)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png",dpi=150)
    plt.show()

    print("\n======结果汇总======")
    print(f"线性SVM准确率：{accuracy_score(y_test,y_pred_linear):.4f}")
    print(f"RBF调参后准确率：{accuracy_score(y_test,y_pred_rbf):.4f}")

if __name__ == "__main__":
    main()
