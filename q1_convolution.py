import numpy as np

# Input image
input_matrix = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
])

# 3x3 filter
kernel = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

stride = 1
padding = 0

# Apply zero padding if requested
padded = np.pad(
    input_matrix,
    ((padding, padding), (padding, padding)),
    mode="constant"
)

# Output dimensions
out_h = (padded.shape[0] - kernel.shape[0]) // stride + 1
out_w = (padded.shape[1] - kernel.shape[1]) // stride + 1

output = np.zeros((out_h, out_w), dtype=int)

# Convolution/filtering from scratch
for i in range(0, padded.shape[0] - kernel.shape[0] + 1, stride):
    for j in range(0, padded.shape[1] - kernel.shape[1] + 1, stride):
        window = padded[i:i + kernel.shape[0], j:j + kernel.shape[1]]
        output[i // stride, j // stride] = np.sum(window * kernel)

print("Input:")
print(input_matrix)
print("\nFilter:")
print(kernel)
print("\nOutput feature map:")
print(output)
print("\nOutput shape:", output.shape)

print(
    "\nIf stride changes from 1 to 2, the filter moves two pixels at a time, "
    "so fewer positions are evaluated and the output feature map becomes smaller."
)
