#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

const int N = 16, C_in = 3, H_in = 28, W_in = 28;
const int C_out = 8, KH = 3, KW = 3;
const int padding = 1, stride = 1;
const int H_out = H_in, W_out = W_in;

// CUDA Convolution Kernel
__global__ void convForwardKernel(const float* input, const float* kernels, const float* biases, float* output,
                                  int N, int C_in, int H_in, int W_in,
                                  int C_out, int KH, int KW,
                                  int H_out, int W_out, int padding, int stride) {
    int o_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (o_idx >= N * C_out * H_out * W_out) return;

    // Flatten the 1D thread/output index to 4D (n, c_out, h_out, w_out)
    // n: batch index, c_out: output channel index, h_out: output height index, w_out: output width index
    int n = o_idx / (C_out * H_out * W_out);
    int remainder1 = o_idx % (C_out * H_out * W_out);
    int c_out = remainder1 / (H_out * W_out);
    int remainder2 = remainder1 % (H_out * W_out);
    int h_out = remainder2 / W_out;
    int w_out = remainder2 % W_out;

    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int h_in_eff = h_out * stride + kh - padding;
                int w_in_eff = w_out * stride + kw - padding;

                if (h_in_eff >= 0 && h_in_eff < H_in && w_in_eff >= 0 && w_in_eff < W_in) {
                    int input_idx = n * (C_in * H_in * W_in) +
                                    c_in * (H_in * W_in) +
                                    h_in_eff * W_in +
                                    w_in_eff;
                    int kernel_idx = c_out * (C_in * KH * KW) +
                                     c_in * (KH * KW) +
                                     kh * KW +
                                     kw;
                    sum += input[input_idx] * kernels[kernel_idx];
                }
            }
        }
    }
    output[o_idx] = sum + biases[c_out];
}

// CPU Convolution Reference
void conv_forward_cpu(const std::vector<float>& input, const std::vector<float>& kernels, const std::vector<float>& biases,
                      std::vector<float>& output) {
    for (int n = 0; n < N; ++n)
        for (int c_out = 0; c_out < C_out; ++c_out)
            for (int h = 0; h < H_out; ++h)
                for (int w = 0; w < W_out; ++w) {
                    float sum = 0.0f;
                    for (int c_in = 0; c_in < C_in; ++c_in)
                        for (int kh = 0; kh < KH; ++kh)
                            for (int kw = 0; kw < KW; ++kw) {
                                int h_in_eff = h + kh - padding;
                                int w_in_eff = w + kw - padding;
                                if (h_in_eff >= 0 && h_in_eff < H_in && w_in_eff >= 0 && w_in_eff < W_in) {
                                    int input_idx = n * (C_in * H_in * W_in) +
                                                    c_in * (H_in * W_in) +
                                                    h_in_eff * W_in +
                                                    w_in_eff;
                                    int kernel_idx = c_out * (C_in * KH * KW) +
                                                     c_in * (KH * KW) +
                                                     kh * KW +
                                                     kw;
                                    sum += input[input_idx] * kernels[kernel_idx];
                                }
                            }
                    int out_idx = n * (C_out * H_out * W_out) + c_out * (H_out * W_out) + h * W_out + w;
                    output[out_idx] = sum + biases[c_out];
                }
}

int main() {
    size_t input_size = N * C_in * H_in * W_in;
    size_t kernel_size = C_out * C_in * KH * KW;
    size_t bias_size = C_out;
    size_t output_size = N * C_out * H_out * W_out;

    std::vector<float> input(input_size), kernels(kernel_size), biases(bias_size), cpu_output(output_size), gpu_output(output_size);

    // Deterministic input values for simplicity. In practice, these would be initialized with real image data.
    for (size_t i = 0; i < input_size; ++i)
        input[i] = 0.01f * static_cast<float>(i % 10);

    // Initialize kernels and biases with deterministic values.
    // This is just an example; in practice, we would use He or Xavier initialization
    // for kernels and biases.
    for (size_t i = 0; i < kernel_size; ++i)
        kernels[i] = 0.001f * static_cast<float>(i % 5);

    for (size_t i = 0; i < bias_size; ++i)
        biases[i] = 0.1f * static_cast<float>(i);

    // Run CPU
    auto t1 = std::chrono::high_resolution_clock::now();
    conv_forward_cpu(input, kernels, biases, cpu_output);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = t2 - t1;

    // Run GPU
    float *d_input, *d_kernels, *d_biases, *d_output;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_kernels, kernel_size * sizeof(float));
    cudaMalloc(&d_biases, bias_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernels, kernels.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (output_size + threads - 1) / threads;

    t1 = std::chrono::high_resolution_clock::now();
    convForwardKernel<<<blocks, threads>>>(d_input, d_kernels, d_biases, d_output,
                                           N, C_in, H_in, W_in, C_out, KH, KW, H_out, W_out, padding, stride);
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = t2 - t1;

    cudaMemcpy(gpu_output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare 10 evenly spaced samples
    std::cout << "\nSample comparison between CPU and GPU:\n";
    for (int i = 0; i < 10; ++i) {
        int idx = i * (output_size / 10);
        float diff = std::fabs(cpu_output[idx] - gpu_output[idx]);
        if (diff < 1e-6f) diff = 0.0f;
        std::cout << "Index " << idx << ": CPU = " << cpu_output[idx]
                  << ", GPU = " << gpu_output[idx] << ", diff = " << diff << "\n";
    }

    std::cout << "\nCPU Time: " << cpu_time.count() << " seconds\n";
    std::cout << "GPU Time: " << gpu_time.count() << " seconds\n";
    double speedup = cpu_time.count() / gpu_time.count();
    std::cout << "GPU is " << speedup << "x faster than CPU\n";

    cudaFree(d_input);
    cudaFree(d_kernels);
    cudaFree(d_biases);
    cudaFree(d_output);

    return 0;
}