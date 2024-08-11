import os, glob, shutil
import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel as nib


def StandardizeNaming(dir_path, train=True, prefix="breast", start_index=1):
    files = sorted(glob.glob(dir_path + '/*'))
    for index, file in enumerate(files, start_index):
        new_name = f"{prefix}_{index:>03}_0000.nii.gz" if train else f"{prefix}_{index:>03}.nii.gz"
        os.rename(file, os.path.join(dir_path, new_name))


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


def generate_label_table(input_dir, output_csv):
    # 初始化空列表以存储文件名和标签
    data = []

    # 遍历输入目录中的所有文件
    for file_name in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, file_name)):
            # 确定标签
            if '_S_' in file_name:
                label = 1
            elif '_M_' in file_name:
                label = 2
            elif '_N_' in file_name:
                label = 0

            # 添加文件名和标签到数据列表
            data.append([file_name, label])

    # 创建DataFrame
    df = pd.DataFrame({"file_name": [row[0] for row in data], "label": [row[1] for row in data]})

    # 保存DataFrame到CSV文件
    df.to_excel(output_csv, index=False)

    print(f"Label table saved to {output_csv}")


def CheckShapeMatching(imagesTr, labelsTr):
    images = sorted(glob.glob(imagesTr + '/*'))
    labels = sorted(glob.glob(labelsTr + '/*'))

    for i, (image, label) in enumerate(zip(images, labels), 1):
        nii_image = nib.load(image)
        nii_label = nib.load(label)

        img_shape = nii_image.header.get_data_shape()
        label_shape = nii_label.header.get_data_shape()

        print(f"{i:>03}: {image} {img_shape} {label} {label_shape}")

        assert img_shape == label_shape, image


def ExtractFirstVolumeSingleFile(input_path, output_path):
    """
    从形状为(2, x, x, x)的nii.gz格式图像中提取第一个数据，并保存到新文件。

    :param input_path: 输入的 .nii.gz 文件路径
    :param output_path: 输出的 .nii.gz 文件路径
    """
    # 加载nii.gz文件
    nii_img = nib.load(input_path)

    # 提取数据
    data = nii_img.get_fdata()
    print(data.shape)

    # 确保输入数据形状的第一个维度为2
    if data.shape[-1] != 2 and data.shape[-1] != 3 and data.shape[-1] != 4:
        return None
    else:
        # 提取第一个数据集（假设数据第一个维度是我们的目标维度）
        first_vol_data = data[:, :, :, 0]

        # 创建新的Nifti1Image对象
        new_img = nib.Nifti1Image(first_vol_data, affine=nii_img.affine, header=nii_img.header)

        # 保存新图像
        nib.save(new_img, output_path)


def ExtractFirstVolumeDir(input_dir, output_dir):
    """
    从形状为(2, x, x, x)的nii.gz格式图像中提取第一个数据，并保存到新文件。

    :param input_dir: 输入的 .nii.gz 文件路径
    :param output_dir: 输出的 .nii.gz 文件路径
    """
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    files = sorted(glob.glob(input_dir + '/*'))
    for f in files:
        try:
            print(f)
            filename = os.path.basename(f)
            save_path = os.path.join(output_dir, filename)
            ExtractFirstVolumeSingleFile(f, save_path)
        except Exception as e:
            print(e)
            continue


def align_image_and_label_nibabel(src_dir, dst_dir):
    srcs = sorted(os.listdir(src_dir))
    dsts = sorted(os.listdir(dst_dir))

    for src, dst in zip(srcs, dsts):
        print(src)
        ref_img = nib.load(os.path.join(src_dir, src))
        mov_img = nib.load(os.path.join(dst_dir, dst))

        # 获取参考图像的仿射矩阵并应用于移动图像
        affine_matrix = ref_img.affine
        aligned_img = nib.Nifti1Image(mov_img.get_fdata(), affine_matrix)

        new_dir = os.path.join(os.path.dirname(dst_dir), 'new_labels')
        new_path = os.path.join(new_dir, dst)

        nib.save(aligned_img, new_path)


def flip_vertical_nifti(src, dst):
    """
        垂直翻转nii.gz图像
    """
    # 读取nii.gz文件
    image = sitk.ReadImage(src)
    # 获取图像的大小
    size = image.GetSize()
    # 循环遍历每个切片
    for z in range(size[2]):
        # 提取当前切片的像素数据
        slice = image[:, :, z]

        # 将当前切片垂直翻转180度
        flipped_slice = sitk.Flip(slice, [False, True])

        # 将翻转后的切片赋值回原始图像中
        image[:, :, z] = flipped_slice

    # 保存旋转后的图像
    sitk.WriteImage(image, dst)


def flip_vertical_nifti_dir(src_dir, dst_dir):
    """
        垂直翻转nii.gz图像
    """
    if not os.path.exists(dst_dir): os.makedirs(dst_dir)

    files = sorted(glob.glob(src_dir + '/*'))
    for f in files:
        try:
            print(f)
            filename = os.path.basename(f)
            save_path = os.path.join(dst_dir, filename)
            flip_vertical_nifti(f, save_path)
        except Exception as e:
            print(e)
            continue


def remove_problem_file(root_dir, problem_excel):
    df = pd.read_excel(problem_excel, sheet_name='Sheet2')
    problem_files = df['file_name'].values
    for r, d, f in os.walk(root_dir):
        for file in f:
            if file[:-7] in problem_files:
                print(file)
                os.remove(os.path.join(r, file))


def dcm2nii_for_file(dcms_dir, nii_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcms_dir)
    reader.SetFileNames(dicom_names)
    image2 = reader.Execute()
    image_array = sitk.GetArrayFromImage(image2)  # z, y, x
    image3 = sitk.GetImageFromArray(image_array)
    image3.CopyInformation(image2)
    sitk.WriteImage(image3, nii_path)


def dcm2nii_for_dir(dcms_dir, nii_dir):
    if not os.path.exists(nii_dir): os.makedirs(nii_dir)

    for r, d, f in os.walk(dcms_dir):
        if "A1" in r and len(f) > 50:
            print(r)
            dcm2nii_for_file(r, os.path.join(nii_dir, r.split('\\')[-4] + '.nii.gz'))


def keep_singe_side_label(label_dir, side='left'):
    files = sorted(glob.glob(label_dir + '/*'))
    for file in files:
        label = sitk.ReadImage(file)
        labelArr = sitk.GetArrayFromImage(label)
        print(file, labelArr.shape)
        if side == 'left':
            labelArr[:, :, 256:] = 0
        elif side == 'right':
            labelArr[:, :, :256] = 0
        new_label = sitk.GetImageFromArray(labelArr)
        new_label.CopyInformation(label)
        sitk.WriteImage(new_label, file)


if __name__ == "__main__":
    # flip_vertical_nifti_dir(r"F:\Data\AdrenalNoduleClassification\Dataset\Separate\MultiNodule\Labels",
    #                         r"F:\Data\AdrenalNoduleClassification\Dataset\Separate\MultiNodule\Labels_")

    # align_image_and_label_nibabel(r"F:\Data\AdrenalNoduleClassification\Images",
    #                               r"F:\Data\AdrenalNoduleClassification\Labels", )

    # remove_problem_file(r"F:\Data\AdrenalNoduleClassification\Dataset\Separate",
    #                     r"F:\Data\AdrenalNoduleClassification\Materials\Dataset_Info.xlsx")

    # dcm2nii_for_dir(r"F:\Data\AdrenalNoduleClassification\1",
    #                 r"F:\Data\AdrenalNoduleClassification\Problems\Images")

    # keep_singe_side_label(r"F:\Data\AdrenalNoduleClassification\ALL\Single\Left\Labels", side='left')

    CheckShapeMatching(r"F:\Data\AdrenalNoduleClassification\Dataset\Total\Images",
                       r"F:\Data\AdrenalNoduleClassification\Dataset\Total\Labels")
