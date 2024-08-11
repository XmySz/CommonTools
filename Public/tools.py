import os

from PIL import Image


def convert_jpg_to_png(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            # 构建完整的文件路径
            file_path = os.path.join(directory, filename)

            # 打开图像
            img = Image.open(file_path)

            # 构建新的文件路径，将扩展名替换为 .png
            new_file_path = os.path.splitext(file_path)[0] + '.png'

            # 保存为 PNG 格式
            img.save(new_file_path, 'PNG')

            # 删除原始的 JPG 文件
            os.remove(file_path)

            print(f"Converted {file_path} to {new_file_path}")


def convert_directory_to_binary_images(directory, threshold=128):
    """
    将指定目录中的所有 PNG 图像转换为二值图像（像素值为 0 和 255），并就地保存。

    :param directory: 包含 PNG 图像的目录路径
    :param threshold: 阈值，默认值为 128
    """
    for filename in os.listdir(directory):
        if filename.lower().endswith('.png'):
            # 构建完整的文件路径
            file_path = os.path.join(directory, filename)

            with Image.open(file_path) as img:
                # 转换为灰度图像
                gray_img = img.convert('L')

                # 应用阈值，将像素值低于阈值的像素设置为 0，高于或等于阈值的像素设置为 255
                binary_img = gray_img.point(lambda p: 255 if p >= threshold else 0)

                # 保存二值图像，覆盖原始文件
                binary_img.save(file_path)
                print(f"Converted {file_path} to binary image.")


def RenameCurrentDirOutputExcel(files_path, excel_path):
    """
        重命名当前目录下的所有文件并生成映射表格, 通常用在远程主机中
    :return:
    """
    import os
    import pandas as pd

    files = sorted(os.listdir(files_path))
    new_names = []
    for index, f in enumerate(files, 1):
        os.rename(os.path.join(files_path, f), os.path.join(files_path, f"ag_{index:>03}_0000.nii.gz"))
        new_names.append(f"ag_{index:>03}_0000.nii.gz")

    pd.DataFrame({"Radiology_ID": [os.path.join(files_path, i) for i in files], "New_Name": new_names}).to_excel(
        excel_path, index=False)


