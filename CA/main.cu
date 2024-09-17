#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

//Global
ifstream inputFile("input.txt");
ifstream refFile("ref.txt");
ofstream output("output.txt");

// CPU function
void goldenanscheaker(int A[], int B[], int sz)
{
    for (int i = 0; i < sz; i++)
    {
        if (A[i] != B[i])
        {
            cout << "wrong answer at " << i << " , CPU:" << B[i] << ", GPU:" << A[i] << endl;
        }
    }
}

void zeropadding(int *pad_img, int *i_img)
{
    int pad_img_width = 10;
    int pad_img_size = pad_img_width * pad_img_width;
    int innerSize = (pad_img_width - 2) * (pad_img_width - 2);

    // 初始化整個填充圖像為 0
    fill(pad_img, pad_img + pad_img_size, 0);

    // 將內部的圖像內容填充進去
    for (int i = 0; i < innerSize; ++i)
    {
        int row = i / (pad_img_width - 2) + 1;
        int col = i % (pad_img_width - 2) + 1;
        pad_img[row * pad_img_width + col] = i_img[i];
    }
}

void edgepadding(int *pad_img, int *i_img)
{
    int pad_img_width = 10;  // Padding img_width
    int imgWidth = 8; // Original image img_width
    for (int i = 0; i < pad_img_width * pad_img_width; i++)
    {
        int row = i / pad_img_width;
        int col = i % pad_img_width;

        if (row == 0)
        { // Top row
            pad_img[i] = i_img[col < 1 ? 0 : (col >= imgWidth + 1 ? imgWidth - 1 : col - 1)];
        }
        else if (row == pad_img_width - 1)
        { // Bottom row
            pad_img[i] = i_img[(imgWidth - 1) * imgWidth + (col < 1 ? 0 : (col >= imgWidth + 1 ? imgWidth - 1 : col - 1))];
        }
        else if (col == 0)
        { // Left column
            pad_img[i] = i_img[(row - 1) * imgWidth];
        }
        else if (col == pad_img_width - 1)
        { // Right column
            pad_img[i] = i_img[(row - 1) * imgWidth + imgWidth - 1];
        }
        else
        { // Inner pixels
            pad_img[i] = i_img[(row - 1) * imgWidth + (col - 1)];
        }
    }
}

void wrappadding(int *pad_img, int *i_img)
{
    int pad_img_width = 10;  // Padding img_width
    int imgWidth = 8; // Original image img_width

    for (int i = 0; i < pad_img_width * pad_img_width; i++)
    {
        int row = i / pad_img_width;
        int col = i % pad_img_width;

        if (row == 0)
        { // Top row
            pad_img[i] = i_img[(imgWidth - 1) * imgWidth + (col < 1 ? imgWidth - 1 : (col > imgWidth ? 0 : col - 1))];
        }
        else if (row == pad_img_width - 1)
        { // Bottom row
            pad_img[i] = i_img[(0) * imgWidth + (col < 1 ? imgWidth - 1 : (col > imgWidth ? 0 : col - 1))];
        }
        else if (col == 0)
        { // Left column
            pad_img[i] = i_img[(row - 1) * imgWidth + (imgWidth - 1)];
        }
        else if (col == pad_img_width - 1)
        { // Right column
            pad_img[i] = i_img[(row - 1) * imgWidth + (0)];
        }
        else
        { // Inner pixels
            pad_img[i] = i_img[(row - 1) * imgWidth + (col - 1)];
        }
    }
}

timespec diff(timespec start, timespec end)
{
    timespec temp;
    if (end.tv_nsec < start.tv_nsec)
    {
        temp.tv_sec = end.tv_sec - start.tv_sec - 1;
        temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    }
    else
    {
        temp.tv_sec = end.tv_sec - start.tv_sec;
        temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    }
    return temp;
}

void print_graph(int graph[], int size)
{
    int img_width = sqrt(size);
    for (int i = 0; i < size; i++)
    {
        output << setw(5) << graph[i] << " ";
        if (i % img_width == img_width - 1)
        {
            output << "\n";
        }
    }
    output << "\n\n";
}

int max_num(int arr[], int size)
{
    // 使用標準庫中的 max_element 函數找到最大元素
    return *std::max_element(arr, arr + size);
}

void cross_correlation(int input_graph[], int padnum, int input_size, int ref[], int output_graph[])
{
    int img_width = sqrt(input_size);

    // zero padding
    int pad_img_width = img_width + 2;
    int *graph_with_padding = new int[pad_img_width * pad_img_width];
    memset(graph_with_padding, 0, pad_img_width * pad_img_width * sizeof(int));
    if (padnum == 0)
        zeropadding(graph_with_padding, input_graph);
    else if (padnum == 1)
        edgepadding(graph_with_padding, input_graph);
    else if (padnum == 2)
        wrappadding(graph_with_padding, input_graph);
    print_graph(graph_with_padding, pad_img_width * pad_img_width);

    // convolution
    fill_n(output_graph, img_width * img_width, 0);
    for (int i = 0; i < img_width; ++i)
    {
        for (int j = 0; j < img_width; ++j)
        {
            int sum = 0;
            // 使用兩個嵌套循環來計算九宮格內的加權和
            for (int di = 0; di < 3; ++di)
            {
                for (int dj = 0; dj < 3; ++dj)
                {
                    sum += graph_with_padding[pad_img_width * (i + di) + (j + dj)] * ref[3 * di + dj];
                }
            }
            output_graph[img_width * i + j] = sum;
        }
    }
}

void max_pooling(int input_graph[], int input_size, int output_graph[])
{
    int img_width = std::sqrt(input_size); // 計算圖像的邊長
    int oWidth = img_width / 2;            // 輸出圖像的邊長

    if (img_width == 4)
    {
        // 如果輸入圖像的寬度是4，直接複製輸入到輸出
        std::copy(input_graph, input_graph + input_size, output_graph);
        return;
    }

    // 對於其他寬度的圖像，進行最大池化操作
    for (int i = 0; i < oWidth; ++i)
    {
        for (int j = 0; j < oWidth; ++j)
        {
            // 構造四個元素的子圖像
            int arr[4] = {
                input_graph[i * img_width * 2 + j * 2],
                input_graph[i * img_width * 2 + j * 2 + 1],
                input_graph[(i * 2 + 1) * img_width + j * 2],
                input_graph[(i * 2 + 1) * img_width + j * 2 + 1]};
            // 將最大值寫入輸出圖像中
            output_graph[i * oWidth + j] = max_num(arr, 4);
        }
    }
}

void horizontal_flip(int input_graph[], int input_size, int output_graph[])
{
    int img_width = std::sqrt(input_size); // 計算圖像的邊長

    // 複製輸入圖像到輸出圖像
    std::copy(input_graph, input_graph + input_size, output_graph);

    // 水平翻轉每一行
    for (int i = 0; i < img_width; ++i)
    {
        for (int j = 0; j < img_width / 2; ++j)
        {
            std::swap(output_graph[i * img_width + j], output_graph[i * img_width + (img_width - 1 - j)]);
        }
    }
}

void vertical_flip(int input_graph[], int input_size, int output_graph[])
{
    int img_width = std::sqrt(input_size); // 計算圖像的邊長

    // 複製輸入圖像到輸出圖像
    std::copy(input_graph, input_graph + input_size, output_graph);

    // 垂直翻轉每一列
    for (int i = 0; i < img_width / 2; ++i)
    {
        for (int j = 0; j < img_width; ++j)
        {
            std::swap(output_graph[i * img_width + j], output_graph[(img_width - 1 - i) * img_width + j]);
        }
    }
}

void leftDiagonal_flip(int input_graph[], int input_size, int output_graph[])
{
    int img_width = std::sqrt(input_size); // 計算圖像的邊長

    // 複製輸入圖像到輸出圖像
    std::copy(input_graph, input_graph + input_size, output_graph);

    // 左對角翻轉
    for (int i = 0; i < img_width; ++i)
    {
        for (int j = 0; j < img_width - i; ++j)
        {
            std::swap(output_graph[i * img_width + j], output_graph[(img_width - 1 - j) * img_width + (img_width - 1 - i)]);
        }
    }
}

void rightDiagonal_flip(int input_graph[], int input_size, int output_graph[])
{
    int img_width = std::sqrt(input_size); // 計算圖像的邊長

    // 複製輸入圖像到輸出圖像
    std::copy(input_graph, input_graph + input_size, output_graph);

    // 右對角翻轉
    for (int i = 0; i < img_width; ++i)
    {
        for (int j = 0; j < img_width - i; ++j)
        {
            std::swap(output_graph[(img_width - 1 - j) * img_width + i], output_graph[i * img_width + (img_width - 1 - j)]);
        }
    }
}

void zoom_in(int input_graph[], int input_size, int output_graph[])
{
    int img_width = std::sqrt(input_size); // 計算圖像的邊長
    double alpha = 0.5;

    if (img_width == 16)
    {
        // 如果圖像邊長為16，直接複製輸入到輸出
        std::copy(input_graph, input_graph + input_size, output_graph);
        return;
    }

    int oWidth = img_width * 2;
    int idx = 0;

    for (int i = 0; i < oWidth; i += 2)
    {
        for (int j = 0; j < oWidth; j += 2)
        {
            // 原始值
            output_graph[i * oWidth + j] = input_graph[idx];
            // 每個像素的三分之一
            output_graph[i * oWidth + j + 1] = std::floor(static_cast<double>(input_graph[idx]) / 3);
            // 每個像素的兩倍三分之二加上20
            output_graph[(i + 1) * oWidth + j] = std::floor(static_cast<double>(input_graph[idx]) * 2 / 3 + 20);
            // 每個像素的alpha倍
            output_graph[(i + 1) * oWidth + j + 1] = std::floor(input_graph[idx] * alpha);
            idx++;
        }
    }
}

void shortcut_and_brightness_adjustment(int input_graph[], int input_size, int output_graph[])
{
    int img_width = std::sqrt(input_size);
    int oWidth = (img_width == 4) ? 4 : img_width / 2; // 輸出圖像的邊長

    for (int i = 0; i < oWidth; ++i)
    {
        for (int j = 0; j < oWidth; ++j)
        {
            int sourceIdx = (img_width == 4) ? (i * img_width + j) : ((i + oWidth / 2) * img_width + (j + oWidth / 2));
            output_graph[i * oWidth + j] = std::floor(0.5 * input_graph[sourceIdx] + 50);
        }
    }
}
void printCPU_time(timespec time1,timespec time2){
    cout << "CPU Execution time = " << (static_cast<long long>(diff(time1, time2).tv_sec))*1000 << "." << setw(6) << setfill('0') << diff(time1, time2).tv_nsec << " ms ." << endl;
}
void print_8ans(float exetime){
    cout << "Thread number = 8 , Execution time =  " << fixed << setprecision(2) << exetime << " ms ." << endl;
}
void print_16ans(float exetime){
    cout << "Thread number = 16 , Execution time =  " << fixed << setprecision(2) << exetime << " ms ." << endl;
}
// GPU function
extern float GPU_kernel_16(int *output_graph_para, int op, int *input_graph, int *ref);
extern float GPU_kernel_8(int *output_graph_para, int op, int *input_graph, int *ref);

int main()
{
    srand(time(NULL));
    int op[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int i;
    int input_graph[64];
    int ref[9];
    int output_graph0[64];
    int output_graph1[16];
    int output_graph2[64];
    int output_graph3[64];
    int output_graph4[64];
    int output_graph5[64];
    int output_graph6[256];
    int output_graph7[16];

    int g_output_graph[256];
    fill(g_output_graph, g_output_graph + 256, 0);
    float exetime;

    // temp input
    for (i = 0; i < 64; i++)
    {
        inputFile >> input_graph[i];
    }
    for (i = 0; i < 9; i++)
    {
        refFile >> ref[i];
    }

    // start testing
    timespec time1, time2;

    for (i = 0; i < sizeof(op) / sizeof(int); i++)
    {
        switch (op[i])
        {
        case 0:
        { 
            int padnum = rand() % 3;
            cout << "Operation 1 - Cross Correlation" << endl;
            // CPU
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
            cross_correlation(input_graph, padnum, 64, ref, output_graph0);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
            printCPU_time(time1,time2);
            print_graph(output_graph0, 64);
            // zero padding
            int *pad_img = new int[256];
            fill_n(pad_img, 256, 0);
            if (padnum == 0)
                zeropadding(pad_img, input_graph);
            else if (padnum == 1)
                edgepadding(pad_img, input_graph);
            else if (padnum == 2)
                wrappadding(pad_img, input_graph);
            // GPU thread 8
            exetime = GPU_kernel_8(g_output_graph, 0, pad_img, ref);
            print_graph(g_output_graph, 64);
            print_8ans(exetime);
            goldenanscheaker(g_output_graph, output_graph0, 64);
            // GPU thread 16
            exetime = GPU_kernel_16(g_output_graph, 0, pad_img, ref);
            print_graph(g_output_graph, 64);
            print_16ans(exetime);
            goldenanscheaker(g_output_graph, output_graph0, 64);
            break;
        }
        case 1:
        { 
            cout << "Operation 2 - Max Pooling" << endl;
            // CPU
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
            max_pooling(input_graph, 64, output_graph1);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
            print_graph(output_graph1, 16);
            printCPU_time(time1,time2);
            // GPU
            // GPU thread 8
            exetime = GPU_kernel_8(g_output_graph, op[i], input_graph, ref);
            print_graph(g_output_graph, 16);
            print_8ans(exetime);
            goldenanscheaker(g_output_graph, output_graph1, 16);
            // GPU thread 16
            exetime = GPU_kernel_16(g_output_graph, op[i], input_graph, ref);
            print_graph(g_output_graph, 16);
            print_16ans(exetime);
            goldenanscheaker(g_output_graph, output_graph1, 16);

            break;
        }
        case 2:
        { 
            cout << "Operation 3 - Right Diagonal Flip" << endl;
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
            rightDiagonal_flip(input_graph, 64, output_graph2);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
            print_graph(output_graph2, 64);
            printCPU_time(time1,time2);
            // GPU
            // GPU thread 8
            exetime = GPU_kernel_8(g_output_graph, op[i], input_graph, ref);
            print_graph(g_output_graph, 64);
            print_8ans(exetime);
            goldenanscheaker(g_output_graph, output_graph2, 64);
            // GPU thread 16
            exetime = GPU_kernel_16(g_output_graph, op[i], input_graph, ref);
            print_graph(g_output_graph, 64);
            print_16ans(exetime);
            goldenanscheaker(g_output_graph, output_graph2, 64);
            break;
        }
        case 3:
        { 
            cout << "Operation 4 - Left Diagonal Flip" << endl;
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
            leftDiagonal_flip(input_graph, 64, output_graph3);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
            print_graph(output_graph3, 64);
            printCPU_time(time1,time2);
            // GPU
            // GPU thread 8
            exetime = GPU_kernel_8(g_output_graph, op[i], input_graph, ref);
            print_graph(g_output_graph, 64);
            print_8ans(exetime);
            goldenanscheaker(g_output_graph, output_graph3, 64);
            // GPU thread 16
            exetime = GPU_kernel_16(g_output_graph, op[i], input_graph, ref);
            print_graph(g_output_graph, 64);
            print_16ans(exetime);
            goldenanscheaker(g_output_graph, output_graph3, 64);
            break;
        }
        case 4:
        { 
            cout << "Operation 5 - Vertical Flip" << endl;
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
            vertical_flip(input_graph, 64, output_graph4);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
            print_graph(output_graph4, 64);
            printCPU_time(time1,time2);
            // GPU
            // GPU thread 8
            exetime = GPU_kernel_8(g_output_graph, op[i], input_graph, ref);
            print_graph(g_output_graph, 64);
            print_8ans(exetime);
            goldenanscheaker(g_output_graph, output_graph4, 64);
            // GPU thread 16
            exetime = GPU_kernel_16(g_output_graph, op[i], input_graph, ref);
            print_graph(g_output_graph, 64);
            print_16ans(exetime);
            goldenanscheaker(g_output_graph, output_graph4, 64);
            break;
        }
        case 5:
        { 
            cout << "Operation 6 - Horizontal Flip" << endl;
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
            horizontal_flip(input_graph, 64, output_graph5);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
            print_graph(output_graph5, 64);
            printCPU_time(time1,time2);
            // GPU
            // GPU thread 8
            exetime = GPU_kernel_8(g_output_graph, op[i], input_graph, ref);
            print_graph(g_output_graph, 64);
            print_8ans(exetime);
            goldenanscheaker(g_output_graph, output_graph5, 64);
            // GPU thread 16
            exetime = GPU_kernel_16(g_output_graph, op[i], input_graph, ref);
            print_graph(g_output_graph, 64);
            print_16ans(exetime);
            goldenanscheaker(g_output_graph, output_graph5, 64);
            break;
        }
        case 6:
        { 
            cout << "Operation 7 - Zoom in" << endl;
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
            zoom_in(input_graph, 64, output_graph6);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
            print_graph(output_graph6, 256);
            printCPU_time(time1,time2);
            // GPU
            // GPU thread 8
            exetime = GPU_kernel_8(g_output_graph, op[i], input_graph, ref);
            print_graph(g_output_graph, 256);
            print_8ans(exetime);
            goldenanscheaker(g_output_graph, output_graph6, 256);
            // GPU thread 16
            exetime = GPU_kernel_16(g_output_graph, op[i], input_graph, ref);
            print_graph(g_output_graph, 256);
            print_16ans(exetime);
            goldenanscheaker(g_output_graph, output_graph6, 256);
            break;
        }
        case 7:
        {
            cout << "Operation 8 - Shortcut & Brightness Adjustment" << endl;
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
            shortcut_and_brightness_adjustment(input_graph, 64, output_graph7);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
            print_graph(output_graph7, 16);
            printCPU_time(time1,time2);
            // GPU
            // GPU thread 8
            exetime = GPU_kernel_8(g_output_graph, op[i], input_graph, ref);
            print_graph(g_output_graph, 16);
            print_8ans(exetime);
            goldenanscheaker(g_output_graph, output_graph7, 16);
            // GPU thread 16
            exetime = GPU_kernel_16(g_output_graph, op[i], input_graph, ref);
            print_graph(g_output_graph, 16);
            print_16ans(exetime);
            goldenanscheaker(g_output_graph, output_graph7, 16);
            break;
        }
        default:
        {
            break;
        }
        }
    }
    inputFile.close();
    refFile.close();
    output.close();
    return 0;
}

