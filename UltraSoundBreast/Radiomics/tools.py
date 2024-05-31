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


def computeICC(excelPath1, excelPath2, saveExcel):
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


if __name__ == "__main__":
    pass
