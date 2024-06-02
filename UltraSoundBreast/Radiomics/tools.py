import numpy as np
import pandas as pd
import glob
import SimpleITK as sitk
import cv2
import pingouin as pg


def LoadExternalValidation(featuresExcelPath):
    """
        用于从表格中加载外部验证集的数据， 并返回数据和标签
    """
    data = pd.read_excel(featuresExcelPath, )

    X = data.drop(["Target", "Patient"], axis=1)
    Y = data["Target"]

    return X, Y


def CheckShapeMatch(imageDir, labelDir):
    """
        检查图像和标签对应的形状是否相同且都为二维, 通常用在特征提取之前, 就地保存
    """
    images = sorted(glob.glob(imageDir + '/*'))
    labels = sorted(glob.glob(labelDir + '/*'))

    for imagePath, labelPath in zip(images, labels):
        img = sitk.ReadImage(imagePath)
        label = sitk.ReadImage(labelPath)
        imgArr = sitk.GetArrayFromImage(img)
        labelArr = sitk.GetArrayFromImage(label)

        print(imagePath, imgArr.shape, labelArr.shape)

        if len(imgArr.shape) == 3:
            newImg = sitk.GetImageFromArray(imgArr[0])
            sitk.WriteImage(newImg, imagePath)
        elif len(labelArr.shape) == 3:
            newImg = sitk.GetImageFromArray(labelArr[0])
            sitk.WriteImage(newImg, labelPath)
        else:
            continue


def ComputeICC(excelPath1, excelPath2, saveExcel):
    """
        计算两个表格的特征的ICC, 两个表格需要具有同样的列名和行数
    :param excelPath1:
    :param excelPath2:
    :param saveExcel:
    :return:
    """
    df1 = pd.read_excel(excelPath1)
    df2 = pd.read_excel(excelPath2)

    df1['Assessment'] = 1  # 添加评估者列
    df2['Assessment'] = 2

    merged_df = pd.concat([df1, df2], axis=0)  # 合并数据框，为了方便计算ICC，我们需要把每个特征的两次评估放在同一列中

    icc_results = {}
    features = df1.columns.tolist()
    features = [i for i in features if i not in ["Assessment", "Patient"]]  # 根据实际需要移除不需要计算ICC的列名

    for feature in features:
        icc = pg.intraclass_corr(data=merged_df, targets='Patient', raters='Assessment', ratings=feature)
        icc_results[feature] = icc.set_index('Type').at['ICC1', 'ICC']

    icc_df = pd.DataFrame.from_dict(icc_results, orient='index', columns=['ICC'])
    icc_df.to_excel(saveExcel, index=False)

    print(f'ICC calculation completed and results are saved to {saveExcel}')


def Equalize_HistogramInRoi(image_path, mask_path, save_path):
    """
        对输入的超声2D PNG格式图像的ROI区域进行直方图均衡化
    :param image_path:
    :param mask_path:
    :param save_path:
    :return:
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise ValueError("无法读取输入图像或mask图像")

    result_image = np.copy(image)

    roi_coords = np.where(mask > 0)
    roi_pixels = image[roi_coords]

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    equalized_roi = clahe.apply(roi_pixels)
    result_image[roi_coords] = equalized_roi.reshape(-1)

    cv2.imwrite(save_path, result_image)


if __name__ == "__main__":
    CheckShapeMatch("/media/sci/P44_Pro/UltraSoundBreastData/数据集/省医训练及内部验证/规范命名/All/Images",
                    "/media/sci/P44_Pro/UltraSoundBreastData/数据集/省医训练及内部验证/规范命名/All/Labels")
