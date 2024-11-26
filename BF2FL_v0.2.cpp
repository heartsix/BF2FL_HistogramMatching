#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// 直方图匹配函数
Mat histogramMatching(const Mat& source, const Mat& templateImg) {
    CV_Assert(source.type() == templateImg.type() && source.channels() == 3);

    Mat matched = source.clone();

    for (int c = 0; c < 3; ++c) { // 对每个通道进行处理
        // 计算源图和模板图的直方图
        Mat src_hist, tmpl_hist;
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        calcHist(&source, 1, &c, Mat(), src_hist, 1, &histSize, &histRange);
        calcHist(&templateImg, 1, &c, Mat(), tmpl_hist, 1, &histSize, &histRange);

        // 计算累积直方图
        Mat src_cdf, tmpl_cdf;
        src_hist.copyTo(src_cdf);
        tmpl_hist.copyTo(tmpl_cdf);
        for (int i = 1; i < histSize; i++) {
            src_cdf.at<float>(i) += src_cdf.at<float>(i - 1);
            tmpl_cdf.at<float>(i) += tmpl_cdf.at<float>(i - 1);
        }

        // 归一化累积直方图
        src_cdf /= source.total();
        tmpl_cdf /= templateImg.total();

        // 创建像素值的映射表
        uchar lut[256];
        int tmpl_idx = 0;
        for (int src_idx = 0; src_idx < 256; src_idx++) {
            while (tmpl_idx < 255 && tmpl_cdf.at<float>(tmpl_idx) < src_cdf.at<float>(src_idx)) {
                tmpl_idx++;
            }
            lut[src_idx] = tmpl_idx;
        }

        // 应用 LUT 映射像素值
        Mat channel;
        LUT(source.channels() == 3 ? source : matched, Mat(1, 256, CV_8U, lut), channel);
        channel.copyTo(matched);
    }
    return matched;
}

// 主函数
int main() {
    // 读取图像
    Mat bf_img = imread("BF-green-6000.png");
    Mat fl_img = imread("FL-green-6000.png");

    if (bf_img.empty() || fl_img.empty()) {
        cout << "无法加载图像，请检查文件路径！" << endl;
        return -1;
    }

    // 转换图像到 RGB 格式
    Mat bf_img_rgb, fl_img_rgb;
    cvtColor(bf_img, bf_img_rgb, COLOR_BGR2RGB);
    cvtColor(fl_img, fl_img_rgb, COLOR_BGR2RGB);

    // 直方图匹配
    Mat matched_img = histogramMatching(bf_img_rgb, fl_img_rgb);

    // 计算残差图像
    Mat residual_img;
    absdiff(fl_img_rgb, matched_img, residual_img);

    // 转换为灰度图
    Mat residual_gray;
    cvtColor(residual_img, residual_gray, COLOR_RGB2GRAY);

    // 显示结果
    imshow("Original BF Image", bf_img_rgb);
    imshow("Original FL Image", fl_img_rgb);
    imshow("Matched Image", matched_img);
    imshow("Residual Intensity Map", residual_gray);

    waitKey(0);
    return 0;
}
