extern "C" __global__ void cosine_similarity(
	const float* a,
	const float* b,
	float* dot_product,
	float* magnitude_a,
	float* magnitude_b,
	const int size
) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x; //global index
	if (index < size) {
		atomicAdd(dot_product, a[index] * b[index]);
		atomicAdd(magnitude_a, a[index] * a[index]);
		atomicAdd(magnitude_b, b[index] * b[index]);
	}
}
