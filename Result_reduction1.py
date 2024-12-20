def update_pred_result(old_pred_file, val_file, new_pred_file):
    """
    将 pred_result_old.txt 文件中的文件命名替换为 val.txt 文件中的命名，并生成新的 pred_result.txt 文件。

    :param old_pred_file: 原始预测结果文件路径
    :param val_file: val.txt 文件路径，包含新的文件命名
    :param new_pred_file: 新的预测结果文件保存路径
    """
    # 读取 val.txt 中的文件名
    with open(val_file, "r") as val_f:
        val_lines = val_f.readlines()

    # 提取 val.txt 中的文件名（按顺序存储）
    val_filenames = [line.strip().split("\t")[0] for line in val_lines]

    # 读取 pred_result_old.txt 中的内容
    with open(old_pred_file, "r") as old_pred_f:
        old_pred_lines = old_pred_f.readlines()

    # 确保两者数量一致，否则抛出异常
    if len(val_filenames) != len(old_pred_lines):
        raise ValueError("val.txt 文件中的文件数量与 pred_result_old.txt 不一致，无法完成映射。")

    # 替换旧文件名为新文件名
    new_pred_lines = []
    for val_filename, old_pred_line in zip(val_filenames, old_pred_lines):
        _, prediction = old_pred_line.strip().split("\t")  # 获取预测值
        new_pred_lines.append(f"{val_filename}\t{prediction}")

    # 保存为新的 pred_result.txt 文件
    with open(new_pred_file, "w") as new_pred_f:
        new_pred_f.write("\n".join(new_pred_lines))

    print(f"新文件已生成：{new_pred_file}")

# 文件路径配置
old_pred_file = r"D:\0Knowledge\Python\deep learning\curriculum_design\dataset5_pred_result\pred_result_old.txt"
val_file = r"D:\0Knowledge\Python\deep learning\curriculum_design\dataset3\annotations\annotations\val.txt"
new_pred_file = r"D:\0Knowledge\Python\deep learning\curriculum_design\dataset5_pred_result\pred_result.txt"

# 调用函数
update_pred_result(old_pred_file, val_file, new_pred_file)
