__global__ void cuda_kernel_8(int *o_img, int *i_img, int *i_tmp, int sz, int op)
{
    int stripe, head, cnt, idx;

    // thread = 8
    if (op == 0)
    { // Cross correlation
        stripe = sz;
        head = (blockIdx.x * blockDim.x + threadIdx.x) * stripe;
        for (int i = head; i < head + stripe; i++)
        {
            o_img[i] = i_img[i + 2 * threadIdx.x] * i_tmp[0] + i_img[i + 2 * threadIdx.x + 1] * i_tmp[1] + i_img[i + 2 * threadIdx.x + 2] * i_tmp[2] + i_img[11 + i - 1 + 2 * threadIdx.x] * i_tmp[3] + i_img[11 + i + 2 * threadIdx.x] * i_tmp[4] + i_img[11 + i + 1 + 2 * threadIdx.x] * i_tmp[5] + i_img[11 + i + 2 * threadIdx.x + 9] * i_tmp[6] + i_img[11 + i + 2 * threadIdx.x + 10] * i_tmp[7] + i_img[11 + i + 2 * threadIdx.x + 11] * i_tmp[8];
        }
    }
    else if (op == 1)
    { // Max pooling
        stripe = sz / 4;
        head = blockIdx.x * 4 + threadIdx.x * stripe;
        cnt = 0;
        for (int i = head; i < head + stripe; i++)
        {
            int tmp1 = i_img[blockIdx.x * sz * 2 + threadIdx.x * 4 + cnt * 2];
            int tmp2 = i_img[blockIdx.x * sz * 2 + threadIdx.x * 4 + cnt * 2 + 1];
            int tmp3 = i_img[blockIdx.x * sz * 2 + threadIdx.x * 4 + cnt * 2 + sz];
            int tmp4 = i_img[blockIdx.x * sz * 2 + threadIdx.x * 4 + cnt * 2 + sz + 1];
            int max_final = max(max(tmp1, tmp2), max(tmp3, tmp4));
            o_img[i] = max_final;
            cnt++;
        }
    }
    else if (op == 2)
    { // Right diagonal flip
        stripe = sz;
        head = (blockIdx.x * blockDim.x + threadIdx.x);
        cnt = 0;
        for (int i = head; i < sz * sz; i += stripe)
        {
            o_img[i] = i_img[cnt + sz * threadIdx.x];
            cnt++;
        }
    }
    else if (op == 3)
    { // Left diagonal flip
        stripe = sz;
        head = (blockIdx.x * blockDim.x + threadIdx.x);
        cnt = 0;
        for (int i = head; i < sz * sz; i += stripe)
        {
            o_img[i] = i_img[(sz - 1 - threadIdx.x) * sz + (sz - 1 - cnt)];
            cnt++;
        }
    }
    else if (op == 4)
    { // Vertical flip
        stripe = sz;
        head = (blockIdx.x * blockDim.x + threadIdx.x);
        cnt = 0;
        for (int i = head; i < sz * sz; i += stripe)
        {
            o_img[i] = i_img[sz * sz - (sz - threadIdx.x) - cnt * sz];
            cnt++;
        }
    }
    else if (op == 5)
    { // Horizontal flip
        stripe = sz;
        head = (blockIdx.x * blockDim.x + threadIdx.x) * stripe;
        cnt = 1;
        for (int i = head; i < head + stripe; i++)
        {
            o_img[i] = i_img[head + stripe - cnt];
            cnt++;
        }
    }
    else if (op == 6)
    { // Zoom-in
        stripe = sz * 4;
        head = (blockIdx.x * blockDim.x + threadIdx.x) * stripe;
        cnt = 0;
        idx = 0;
        for (int i = head; i < head + stripe; i += 1)
        {
            if (cnt % 2 == 0 && cnt <= 14)
            {
                if (!cnt)
                    idx = head / 4;
                else
                    idx = idx + 1;
                o_img[i] = i_img[idx];
            }
            else if (cnt % 2 == 1 && cnt <= 15)
                o_img[i] = i_img[idx] / 3;
            else if (cnt % 2 == 0 && cnt <= 30)
            {
                if (cnt == 16)
                    idx = head / 4;
                else
                    idx = idx + 1;
                o_img[i] = i_img[idx] * 2 / 3 + 20;
            }
            else
                o_img[i] = i_img[idx] / 2;
            cnt++;
        }
    }
    else if (op == 7)
    { // Short-cut + brightness
        stripe = sz / 4;
        head = blockIdx.x * 4 + threadIdx.x * stripe;
        cnt = 0;
        for (int i = head; i < head + stripe; i++)
        {
            idx = 18 + blockIdx.x * sz + threadIdx.x * 2 + cnt;
            o_img[i] = i_img[idx] / 2 + 50;
            cnt++;
        }
    }
};
float GPU_kernel_8(int *output_graph_para, int op, int *input_graph, int *ref)
{

    int *dimg, *dtmp, *dimg_o;

    // Creat Timing Event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate Memory Space on Device
    cudaMalloc((void **)&dimg, sizeof(int) * 256);
    cudaMalloc((void **)&dtmp, sizeof(int) * 9);
    cudaMalloc((void **)&dimg_o, sizeof(int) * 256);

    // Copy Data to be Calculated
    cudaMemcpy(dimg, input_graph, sizeof(int) * 256, cudaMemcpyHostToDevice);
    cudaMemcpy(dtmp, ref, sizeof(int) * 9, cudaMemcpyHostToDevice);

    // set parameter
    int thread_num, block_num, sz;
    sz = 8;
    // total thread == 8
    if (op == 0)
        thread_num = 8; // cross correlation
    if (op == 1)
        thread_num = 2; // max_pooling  //when thread = 4 thread_num = 4 block_num = 1
    if (op == 2)
        thread_num = 8; // right-diagonal flip
    if (op == 3)
        thread_num = 8; // left-diagonal flip
    if (op == 4)
        thread_num = 8; // vertical flip
    if (op == 5)
        thread_num = 8; // horizontal flip
    if (op == 6)
        thread_num = 8; // zoom-in
    if (op == 7)
        thread_num = 2; // short-cut + brightness //when thread = 16 thread_num = 4 block_num = 4

    if (op == 7 || op == 1)
        block_num = 4; // short-cut + brightness or max_pooling
    else
        block_num = 1;

    // Start Timer
    cudaEventRecord(start, 0);

    // Lunch Kernel
    dim3 dimGrid(block_num);
    dim3 dimBlock(thread_num);
    cuda_kernel_8<<<dimGrid, dimBlock>>>(dimg_o, dimg, dtmp, sz, op);

    // Stop Timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Copy Output back //DZT
    cudaMemcpy(output_graph_para, dimg_o, sizeof(int) * 256, cudaMemcpyDeviceToHost);

    // Release Memory Space on Device
    cudaFree(dimg);
    cudaFree(dtmp);
    cudaFree(dimg_o);

    // Calculate Elapsed Time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    return elapsedTime;
};