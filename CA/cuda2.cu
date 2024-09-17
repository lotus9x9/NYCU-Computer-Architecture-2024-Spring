__global__ void cuda_kernel_16(int *o_img, int *i_img, int *i_tmp, int sz, int op)
{
    int stripe, head, cnt, idx;

    // thread = 16
    if (op == 0)
    { // Cross correlation
        stripe = sz / 2;
        head = blockIdx.x * sz + threadIdx.x * stripe;
        for (int i = head; i < head + stripe; i++)
        {
            o_img[i] = i_img[i + 2 * blockIdx.x] * i_tmp[0] + i_img[i + 2 * blockIdx.x + 1] * i_tmp[1] + i_img[i + 2 * blockIdx.x + 2] * i_tmp[2] +
                       i_img[11 + i - 1 + 2 * blockIdx.x] * i_tmp[3] + i_img[11 + i + 2 * blockIdx.x] * i_tmp[4] + i_img[11 + i + 1 + 2 * blockIdx.x] * i_tmp[5] +
                       i_img[11 + i + 2 * blockIdx.x + 9] * i_tmp[6] + i_img[11 + i + 2 * blockIdx.x + 10] * i_tmp[7] + i_img[11 + i + 2 * blockIdx.x + 11] * i_tmp[8];
        }
    }
    else if (op == 1)
    { // Max pooling
        head = blockIdx.x * 4 + threadIdx.x;
        int tmp1 = i_img[blockIdx.x * sz * 2 + threadIdx.x * 2];
        int tmp2 = i_img[blockIdx.x * sz * 2 + threadIdx.x * 2 + 1];
        int tmp3 = i_img[blockIdx.x * sz * 2 + threadIdx.x * 2 + sz];
        int tmp4 = i_img[blockIdx.x * sz * 2 + threadIdx.x * 2 + sz + 1];
        int max_final = max(max(tmp1, tmp2), max(tmp3, tmp4));
        o_img[head] = max_final;
    }
    else if (op == 2)
    { // Right diagonal flip
        stripe = sz;
        head = blockIdx.x + threadIdx.x * 32;
        cnt = 0;
        if (threadIdx.x == 0)
        {
            for (int i = head; i < sz * 4; i += stripe)
            {
                int x = blockIdx.x;
                int y = cnt + threadIdx.x * 4;
                o_img[i] = i_img[y + sz * x];
                cnt++;
            }
        }
        else
        {
            for (int i = head; i < sz * sz; i += stripe)
            {
                int x = blockIdx.x;
                int y = cnt + threadIdx.x * 4;
                o_img[i] = i_img[y + sz * x];
                cnt++;
            }
        }
    }
    else if (op == 3)
    { // Left diagonal flip
        stripe = sz;
        head = blockIdx.x + threadIdx.x * 32;
        cnt = 0;
        if (threadIdx.x == 0)
        {
            for (int i = head; i < sz * 4; i += stripe)
            {
                o_img[i] = i_img[(sz - 1 - blockIdx.x) * sz + (sz - 1 - cnt)];
                cnt++;
            }
        }
        else
        {
            for (int i = head; i < sz * sz; i += stripe)
            {
                o_img[i] = i_img[(sz - 1 - blockIdx.x) * sz + (sz - cnt - 5)];
                cnt++;
            }
        }
    }
    else if (op == 4)
    { // Vertical flip
        stripe = sz;
        head = blockIdx.x + threadIdx.x * 32;
        cnt = 0;
        if (threadIdx.x == 0)
        {
            for (int i = head; i < sz * 4; i += stripe)
            {
                o_img[i] = i_img[sz * sz - (sz - blockIdx.x) - cnt * sz];
                cnt++;
            }
        }
        else
        {
            for (int i = head; i < sz * sz; i += stripe)
            {
                o_img[i] = i_img[32 - (sz - blockIdx.x) - cnt * sz];
                cnt++;
            }
        }
    }
    else if (op == 5)
    { // Horizontal flip
        stripe = sz / 2;
        head = blockIdx.x * sz + threadIdx.x * stripe;
        cnt = 0;
        for (int i = head; i < head + stripe; i++)
        {
            if (threadIdx.x == 0)
                o_img[i] = i_img[head + stripe + 3 - cnt];
            else
                o_img[i] = i_img[head - cnt - 1];
            cnt++;
        }
    }
    else if (op == 6)
    { // Zoom-in
        stripe = sz * 2;
        head = blockIdx.x * stripe;
        cnt = 0;
        int y = blockIdx.x / 2;
        if (blockIdx.x % 2 == 0)
        {
            for (int i = head; i < head + stripe; i += 2)
            {
                idx = y * sz + cnt;
                o_img[i] = i_img[idx];
                o_img[i + 1] = i_img[idx] / 3;
                cnt++;
            }
        }
        else
        {
            for (int i = head; i < head + stripe; i += 2)
            {
                idx = y * sz + cnt;
                o_img[i] = i_img[idx] * 2 / 3 + 20;
                o_img[i + 1] = i_img[idx] / 2;
                cnt++;
            }
        }
    }
    else if (op == 7)
    { // Short-cut + brightness
        stripe = 1;
        head = (blockIdx.x * blockDim.x + threadIdx.x) * stripe;
        idx = 18 + sz * blockIdx.x + threadIdx.x;
        o_img[head] = i_img[idx] / 2 + 50;
    }
};
float GPU_kernel_16(int *output_graph_para, int op, int *input_graph, int *ref)
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

    // total thread == 16
    if (op == 0)
        thread_num = 2; // cross correlation
    if (op == 1)
        thread_num = 4; // max_pooling
    if (op == 2)
        thread_num = 2; // right-diagonal flip
    if (op == 3)
        thread_num = 2; // left-diagonal flip
    if (op == 4)
        thread_num = 2; // vertical flip
    if (op == 5)
        thread_num = 2; // horizontal flip
    if (op == 6)
        thread_num = 1; // zoom-in
    if (op == 7)
        thread_num = 4; // short-cut + brightness

    if (op == 7 || op == 1)
        block_num = 4; // short-cut + brightness or max_pooling
    else if (op == 6)
        block_num = 16; // zoom-in
    else
        block_num = 8;

    // Start Timer
    cudaEventRecord(start, 0);

    // Lunch Kernel
    dim3 dimGrid(block_num);
    dim3 dimBlock(thread_num);
    cuda_kernel_16<<<dimGrid, dimBlock>>>(dimg_o, dimg, dtmp, sz, op);

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