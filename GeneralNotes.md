# General Notes

## 4D to 1D flattening and vice-versa
**Note**: All calculations assume row-major layout.

In a convolutional neural network (CNN), the kernel is a 4D tensor with shape `[C_out, C_in, KH, KW]`.

Each dimension represents:

* `C_out` - Number of output channels, i.e., the number of convolutional filters (each producing one output channel)

* `C_in` - Number of input channels (e.g., 1 for grayscale, 3 for RGB, or higher for other types)

* `KH` - Kernel height (e.g., 3 for a 3x3 filter)

* `KW` - Kernel width (e.g., 3 for a 3x3 filter)

The important insight here is that Each filter spans all input channels. Thus, if there are 16 input channels, each filter will be of shape 3x3x16. If there are 64 such filters, total number of kernel parameters becomes 64x3x3x16. These 9216 values, called kernel weights, are what the training process will learn so that the loss is minimized.

Let's take the example of MNIST digits classification. Assuming batch size of 2 and number of filters to be 8, Output shape becomes `[N=2][C_out=8][H=28][W=28]`.

So:

* Total output elements = 2x8x28x28 = 12544
* We launch 12544 threads, each producing one output element
* Each thread has a global linear index in [0, 12543]


When flattening `(n, c_out, h_out, w_out)` to a 1D index `o_idx`, the formula is:

```cpp
o_idx = n * (C_out * H_out * W_out)
      + c_out * (H_out * W_out)
      + h_out * W_out
      + w_out
```

The formula to convert a flattened 1D index to a unflattened 4D index is:

```cpp
int n = o_idx / (C_out * H_out * W_out);
int remainder1 = o_idx % (C_out * H_out * W_out);
int c_out = remainder1 / (H_out * W_out);
int remainder2 = remainder1 % (H_out * W_out);
int h_out = remainder2 / W_out;
int w_out = remainder2 % W_out;
```

Let's take the thread with `o_idx=11000` and unflatten it to `(n, c_out, h_out, w_out)`.

### Given Dimensions

| Parameter | Value | Description                      |
|-----------|-------|----------------------------------|
| `W_out`   | 28    | Output width                     |
| `H_out`   | 28    | Output height                    |
| `C_out`   | 8     | Number of filters/output channels|
| `N`       | 2     | Batch size                       |

### 1D to 4D Calculations

| Step     | Calculation                            | Result |
|----------|----------------------------------------|--------|
| o_idx    | Given                                  | 11000  |
| n        | 11000 / (8×28×28) = 11000 / 6272       | 1      |
| remainder1 | 11000 % 6272                       | 4728   |
| c_out    | 4728 / (28×28) = 4728 / 784            | 6      |
| remainder2 | 4728 % 784                         | 504    |
| h_out    | 504 / 28                               | 18     |
| w_out    | 504 % 28                               | 0      |


Final Unflattened Output:

```cpp
(n, c_out, h_out, w_out) = (1, 6, 18, 0)
```




	
	
