# Simple Convolutional Forward Pass in CUDA C++

This repository contains a basic implementation of a convolutional neural network's forward pass in C++. It includes a naive CUDA kernel for GPU acceleration and a corresponding CPU implementation for verification and performance comparison.


## 🔍 Overview
The core of this project is the convForwardKernel kernel written in CUDA C++, which performs the convolution operation on a batch of images. It's a naive implementation for learning purposes and does not contain any optimizations like tiling and memory coalescing etc. This kernel will eventually be used in a complete convolutional neural network program for classifying MNIST digits.

- The input is of shape **(N, C_in, H, W)**.
- The kernel is of shape **(C_out, C_in, KH, KW)** with `stride=1` and `padding=1`.
- It uses synthetic data instead of actual images for simplicity.


## 📌 Key Parameters
`N` - batch size

`C_in` - number of channels in the input (1 for grayscale MNIST images, 3 for regular RGB images, and a larger number for deeper convolutional layers)

`H` and `W` - height and width fo the input

`C_out` - number of output channels which is same as the number of filters/kernels

`KH` and `KW` - kernel height and width

## 🛠️ Getting Started

### Compilation
You can compile the code using the NVIDIA CUDA Compiler:

```bash
nvcc -o CudaConvolutionForward01 CudaConvolutionForward01
```

### Execution
Run the resulting executable from the terminal:

```bash
./CudaConvolutionForward01
```


## ⏱️ Performance Comparison
As the batch size increases, GPU can process more and more images in parallel and its advantage over CPU becomes more pronounced.

For N=1

```
CPU Time: 0.00182464 seconds
GPU Time: 0.000210617 seconds
GPU is 8.66333x faster than CPU
```

For N=16

```
CPU Time: 0.0290934 seconds
GPU Time: 0.000228604 seconds
GPU is 127.266x faster than CPU
```

For N=128

```
CPU Time: 0.231205 seconds
GPU Time: 0.000375695 seconds
GPU is 615.407x faster than CPU
```

## 🧑‍💻 Author

**Manish Kumar**  
iOS Developer, ML & CUDA Programming Enthusiast


## 📜 License

MIT License — do whatever you want with this. Attribution appreciated.
