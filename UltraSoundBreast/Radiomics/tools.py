import os

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
    features = [i for i in features if i not in ["Assessment", "ID"]]  # 根据实际需要移除不需要计算ICC的列名

    for feature in features:
        icc = pg.intraclass_corr(data=merged_df, targets='ID', raters='Assessment', ratings=feature)
        icc_results[feature] = icc.set_index('Type').at['ICC3k', 'ICC']

    icc_df = pd.DataFrame.from_dict(icc_results, orient='index', columns=['ICC'])
    icc_df.to_excel(saveExcel, index=False)

    print(f'ICC calculation completed and results are saved to {saveExcel}')


def EqualizeHistogramInRoi(image_path, mask_path, save_path, OnlyROI=True):
    """
        对输入的超声2D PNG格式图像的ROI区域进行直方图均衡化
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    result_image = np.copy(image)

    roi_coords = np.where(mask > 0)
    roi_pixels = image[roi_coords] if OnlyROI else image

    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(3, 3))
    equalized_roi = clahe.apply(roi_pixels)

    if OnlyROI:
        result_image[roi_coords] = equalized_roi.reshape(-1)
        cv2.imwrite(save_path, result_image)
    else:
        cv2.imwrite(save_path, equalized_roi)


def EqualizeHistogramInRoiDir(imageDir, maskDir, saveDir):
    """
        对输入的超声2D PNG/JGP格式图像的ROI区域进行直方图均衡化
    """
    images = sorted(glob.glob(imageDir + '/*'))
    masks = sorted(glob.glob(maskDir + '/*'))

    for imagePath, maskPath in zip(images, masks):
        EqualizeHistogramInRoi(imagePath, maskPath, saveDir + '/' + os.path.basename(imagePath), OnlyROI=False)


def LassoSelectFeatures(excelPath, sheetName="Sheet3", saveExcelPath=None):
    """
        使用Lasso模型筛选特征, 将结果保存到saveExcelPath中
    """
    from sklearn.model_selection import train_test_split, KFold, GridSearchCV
    from sklearn.linear_model import Lasso
    from matplotlib import pyplot as plt

    featuresExcelPath = excelPath

    data = pd.read_excel(featuresExcelPath, sheet_name=sheetName)

    X = data.drop(["Target", "Patient"], axis=1)
    X.columns = X.columns.astype(str)
    Y = data["Target"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    params = {"alpha": np.linspace(1e-5, 10.0, num=500)}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    lasso = Lasso()
    lasso_cv = GridSearchCV(lasso, param_grid=params, cv=kf, n_jobs=8)
    lasso_cv.fit(X, Y)
    print("Best Params {}".format(lasso_cv.best_params_))

    names = data.drop(["Target", "Patient"], axis=1).columns

    lasso1 = Lasso(alpha=lasso_cv.best_params_["alpha"])
    lasso1.fit(X_train, Y_train)

    lasso1_coef = np.abs(lasso1.coef_)
    print("Lasso Coef == 0 : {}".format(len(lasso1_coef[lasso1_coef == 0])))

    print(names[lasso1_coef > 0])

    if saveExcelPath is not None:
        newDF = data[["Patient", "Target"] + names[lasso1_coef > 0].values.tolist()]
        newDF.to_excel(saveExcelPath, index=False)

    plt.bar(names, lasso1_coef)
    plt.xticks(rotation=90)
    plt.grid()
    plt.title("Feature Selection Based on Lasso")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.ylim(0, 0.5)
    plt.show()


def ExtractExcelColumns(excelPath1, excelPath2, saveExcel):
    """
        从两个表格中提取相同的列, 并保存到新的表格中
    """
    df1 = pd.read_excel(excelPath1, sheet_name="Sheet1")
    df2 = pd.read_excel(excelPath2, sheet_name="Sheet1")

    columns2 = df2.columns.tolist()

    newDF = df1[columns2]
    newDF.to_excel(saveExcel, index=False)

    print(f"Extracted columns are saved to {saveExcel}")


if __name__ == "__main__":
    # EqualizeHistogramInRoiDir("/media/sci/P44_Pro/UltraSoundBreastData/数据集/标准化图像/省医/Images",
    #                           "/media/sci/P44_Pro/UltraSoundBreastData/数据集/标准化图像/省医/Labels",
    #                           "/media/sci/P44_Pro/UltraSoundBreastData/数据集/标准化图像/省医/EqualizedImagesAll")

    EqualizeHistogramInRoi("/media/sci/P44_Pro/UltraSoundBreastData/数据集/标准化图像/省医/Images/G11675.png",
                           "/media/sci/P44_Pro/UltraSoundBreastData/数据集/标准化图像/省医/Labels/G11675.png",
                           "/media/sci/P44_Pro/UltraSoundBreastData/数据集/标准化图像/省医/EqualizedImagesAll/G11675.png",
                           OnlyROI=True)