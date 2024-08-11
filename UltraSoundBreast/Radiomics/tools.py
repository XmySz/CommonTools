import os
import shutil
import numpy as np
import pandas as pd
import glob
import SimpleITK as sitk
import cv2
import pingouin as pg
from PIL import Image


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

        assert imgArr.shape[0] == labelArr.shape[0], f"Shape mismatch between {imagePath} and {labelPath}"

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

    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(3, 3))
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
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    images = sorted(glob.glob(imageDir + '/*'))
    masks = sorted(glob.glob(maskDir + '/*'))

    for imagePath, maskPath in zip(images, masks):
        print(imagePath, )
        try:
            EqualizeHistogramInRoi(imagePath, maskPath, saveDir + '/' + os.path.basename(imagePath), OnlyROI=True)
        except Exception as e:
            print("*" * 100, e)


def LassoSelectFeaturesWithImportance(excelPath, sheetName="Sheet1", saveExcelPath=None):
    """
        使用Lasso模型筛选特征, 将结果保存到saveExcelPath中
    """
    from sklearn.model_selection import train_test_split, KFold, GridSearchCV
    from sklearn.linear_model import Lasso
    from matplotlib import pyplot as plt

    featuresExcelPath = excelPath

    data = pd.read_excel(featuresExcelPath, sheet_name=sheetName)

    X = data.drop(["Patient", "Target", ], axis=1)
    X.columns = X.columns.astype(str)
    Y = data["Target"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    params = {"alpha": np.linspace(1e-6, 10.0, num=5000)}
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


def LassoSelectFeaturesWithCoefsGraph(excelPath, sheetName="Sheet3", saveExcelPath=None):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Lasso
    from matplotlib import pyplot as plt
    from operator import itemgetter

    featuresExcelPath = excelPath

    data = pd.read_excel(featuresExcelPath, sheet_name=sheetName)

    X = data.drop(["Patient", "Target", ], axis=1)
    X.columns = X.columns.astype(str)
    Y = data["Target"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    coefs = []
    alphas = np.linspace(1e-6, 10.0, num=5000)

    for a in alphas:
        lasso = Lasso(alpha=a, fit_intercept=False)
        lasso.fit(X, Y)
        coefs.append(lasso.coef_)

    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Lasso coefficients as a function of the regularization')
    plt.axis('tight')
    plt.show()

    lasso1 = Lasso(alpha=1e-1, fit_intercept=False)
    lasso1.fit(X_train, Y_train)
    lasso1_coef = np.abs(lasso1.coef_)
    print("Lasso Coef == 0 : {}".format(len(lasso1_coef[lasso1_coef == 0])))

    names = data.drop(["Target", "Patient"], axis=1).columns
    print(names[lasso1_coef > 0])
    print(lasso1_coef[lasso1_coef > 0])

    # 合并为一个字典
    d = dict(zip(names[lasso1_coef > 0], lasso1_coef[lasso1_coef > 0]))

    # 根据d的values排序
    sorted_dict_values = dict(sorted(d.items(), key=itemgetter(1)))
    for k, v in sorted_dict_values.items():
        print(k, v)

    if saveExcelPath is not None:
        newDF = data[["Patient", "Target"] + names[lasso1_coef > 0].values.tolist()]
        newDF.to_excel(saveExcelPath, index=False)


def ExtractExcelColumns(excelPath1, excelPath2, saveExcel):
    """
        从两个表格中提取相同的列, 并保存到新的表格中, 只需要留下excelPath2中的列
        excelPath1: 第一个表格路径, 用于提取列
        excelPath2: 第二个表格路径, 用于提取列名
    """
    df1 = pd.read_excel(excelPath1, sheet_name="Sheet1")
    df2 = pd.read_excel(excelPath2, sheet_name="Sheet1")

    columns2 = df2.columns.tolist()

    newDF = df1[columns2]
    newDF.to_excel(saveExcel, index=False)

    print(f"Extracted columns are saved to {saveExcel}")


def AdjustBrightnessInRoi(image_path, mask_path, save_path, brightness=50, OnlyROI=False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    result_image = np.copy(image)

    if OnlyROI:
        roi_coords = np.where(mask > 0)
        roi_pixels = image[roi_coords]
        adjusted_roi = cv2.add(roi_pixels, brightness)
        result_image[roi_coords] = adjusted_roi.reshape(-1)
    else:
        result_image = cv2.add(image, brightness)

    cv2.imwrite(save_path, result_image)


def AdjustBrightnessInRoiDir(imageDir, maskDir, saveDir, brightness=50):
    """
        对输入的超声2D PNG/JGP格式图像的ROI区域进行亮度调整
    """
    images = sorted(glob.glob(imageDir + '/*'))
    masks = sorted(glob.glob(maskDir + '/*'))

    for imagePath, maskPath in zip(images, masks):
        print(imagePath, )
        AdjustBrightnessInRoi(imagePath, maskPath, saveDir + '/' + os.path.basename(imagePath), brightness=brightness,
                              OnlyROI=False)


def CheckExcelsColumnsUnion(EXCEL_PATH1, EXCEL_PATH2, saveExcel=False):
    """
        输出两个Excel表格的列名的交集
    """
    df1 = pd.read_excel(EXCEL_PATH1)
    df2 = pd.read_excel(EXCEL_PATH2)

    columns1 = df1.columns.tolist()
    columns2 = df2.columns.tolist()

    # 输出重复的列名
    print(set(columns1) & set(columns2))
    print(len(set(columns1) & set(columns2)))

    if saveExcel is not None:
        newDF = df1[list(set(columns1) & set(columns2))]
        newDF.to_excel(EXCEL_PATH1.replace(".xlsx", "_1.xlsx"), index=False)

        newDF = df2[list(set(columns1) & set(columns2))]
        newDF.to_excel(EXCEL_PATH2.replace(".xlsx", "_1.xlsx"), index=False)


def ExtractExcelRows(excelPath1, saveExcelPath, excelPath2):
    """
        从表格中提取指定的行, 并保存到新的表格中
    """
    df = pd.read_excel(excelPath1)
    rows = pd.read_excel(excelPath2)["Patient"].values
    print(rows, len(rows))
    newDF = df[df["Patient"].isin(rows)]
    newDF.to_excel(saveExcelPath, index=False)


def ExpandEdgeForSingleFile(maskPath, savePath, expandSize=5):
    """
        对单个超声图像的ROI区域进行边缘扩展
    """
    img = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
    expandImg = np.zeros_like(img)
    indices = np.where(img == 255)
    for y, x in zip(indices[0], indices[1]):
        expandImg[max(0, y - expandSize):min(y + expandSize + 1, img.shape[0]),
        max(0, x - expandSize):min(x + expandSize + 1, img.shape[1])] = 255

    cv2.imwrite(savePath, expandImg)


def ExpandEdgeForDir(maskDir, saveDir, expandSize=5):
    """
        对超声图像的ROI区域进行边缘扩展
    """
    masks = sorted(glob.glob(maskDir + '/*'))

    for maskPath in masks:
        print(maskPath)
        fileName = os.path.basename(maskPath)
        ExpandEdgeForSingleFile(maskPath, os.path.join(saveDir, fileName), expandSize=expandSize)


def RemoveDir1FilesInDir2(dir1, dir2, dir3):
    """
        删除dir2中存在于dir1中的文件
    """
    files1 = sorted(glob.glob(dir1 + '/*'))

    for file1 in files1:
        fileName = os.path.basename(file1)
        if os.path.exists(os.path.join(dir2, fileName)):
            shutil.copy(os.path.join(dir2, fileName), os.path.join(dir3, fileName))


def ZScoreExcel(excelPath, saveExcelPath):
    """
        对表格中的数据逐列进行Z-Score标准化
    """
    df = pd.read_excel(excelPath)
    df.drop(["Patient", "Target"], axis=1, inplace=True)
    df = (df - df.mean()) / df.std()
    df.to_excel(saveExcelPath, index=False)


def MinMaxExcel(excelPath, saveExcelPath):
    """
        对表格中的数据进行MinMax标准化
    """
    df = pd.read_excel(excelPath)
    df.drop(["Patient", "Target"], axis=1, inplace=True)
    df = (df - df.min()) / (df.max() - df.min())
    df.to_excel(saveExcelPath, index=False)


def convert_images_to_white(input_dir, output_dir):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有文件
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith('.png'):
            # 构造文件路径
            file_path = os.path.join(input_dir, file_name)

            # 打开图像
            with Image.open(file_path) as img:
                # 创建一个与原图像尺寸相同的纯白色图像
                white_image = Image.new("RGB", img.size, (255, 255, 255))

                # 保存纯白色图像到输出目录
                output_file_path = os.path.join(output_dir, file_name)
                white_image.save(output_file_path)
                print(f"Processed and saved: {output_file_path}")


def filter_excel(excel1_path, excel2_path, output_path):
    # 读取两个Excel文件
    df1 = pd.read_excel(excel1_path)
    df2 = pd.read_excel(excel2_path)

    # 提取两个表格中的Patient列
    patients1 = set(df1['Patient'])
    patients2 = set(df2['Patient'])

    # 只保留excel2中也存在于excel1中的Patient行
    common_patients = patients1.intersection(patients2)
    filtered_df2 = df2[df2['Patient'].isin(common_patients)]

    # 将结果保存到新的Excel文件
    filtered_df2.to_excel(output_path, index=False)


if __name__ == "__main__":
    # RemoveDir1FilesInDir2("/media/sci/P44_Pro/UltraSoundBreastData/数据集/标准化图像/重医/第二次/Images",
    #                       "/media/sci/P44_Pro/UltraSoundBreastData/数据集/标准化图像/重医/第二次/Labels",
    #                       "/media/sci/P44_Pro/UltraSoundBreastData/数据集/标准化图像/重医/第二次/Labels1")

    # LassoSelectFeaturesWithImportance(r"F:\Data\UltraSoundBreastData\数据集\省医训练及内部验证\裁剪后\256_256\影像组学特征_U检验筛选.xlsx",
    #                                   sheetName='Sheet1',
    #                                   saveExcelPath=r"F:\Data\UltraSoundBreastData\数据集\省医训练及内部验证\裁剪后\256_256\影像组学特征_U检验筛选_LASSO.xlsx")

    # filter_excel(r"F:\Data\UltraSoundBreastData\数据集\省医训练及内部验证\裁剪后\256_256\深度学习特征.xlsx",
    #              r"F:\Data\UltraSoundBreastData\数据集\省医训练及内部验证\裁剪后\256_256\影像组学特征.xlsx",
    #              r"F:\Data\UltraSoundBreastData\数据集\省医训练及内部验证\裁剪后\256_256\影像组学特征1.xlsx")

    # ZScoreExcel(r"C:\Users\Zyn__\Desktop\ly\SeniorSisterRadiomics\Dataset\影像组学特征_U检验筛选_方差筛选_相关性筛选.xlsx",
    #             r"C:\Users\Zyn__\Desktop\ly\SeniorSisterRadiomics\Dataset\影像组学特征_U检验筛选_方差筛选_相关性筛选1.xlsx")

    MinMaxExcel(r"C:\Users\Zyn__\Desktop\ly\SeniorSisterRadiomics\Dataset\影像组学特征_U检验筛选_方差筛选_相关性筛选1.xlsx",
                r"C:\Users\Zyn__\Desktop\ly\SeniorSisterRadiomics\Dataset\影像组学特征_U检验筛选_方差筛选_相关性筛选2.xlsx")