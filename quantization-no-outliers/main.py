import numpy as np
import utils

# suppress scientific notation
np.set_printoptions(suppress=True)

# generate random data
params = np.random.uniform(low=-50, high=150, size=20)
params[0] = params.max() + 1
params[1] = params.min() - 1
params[2] = 0

# round each number to the second decimal place
params = np.round(params, 2)

# asymmetric quantization
asymmetric_q, asymmetric_scale, asymmetric_zero = utils.asymmetric_quantization(params, bits=8)
# symmetric quantization
symmetric_q, symmetric_scale = utils.symmetric_quantization(params, bits=8)

# dequantize and compare with original data
params_deq_asymmetric = utils.asymmetric_dequantize(asymmetric_q, asymmetric_scale, asymmetric_zero)
params_deq_symmetric = utils.symmetric_dequantize(symmetric_q, symmetric_scale)

print('Original data vs. asymmetric dequantized data')
print("Original data:")
print(np.round(params, 2))
print('')
print("Asymmetric dequantized data:")
print(np.round(params_deq_asymmetric, 2))
print('')
print(f'Quantization error: {utils.quantization_error(params, params_deq_asymmetric)}')
print('')
print('Original data vs. symmetric dequantized data')
print("Original data:")
print(np.round(params, 2))
print('')
print("Symmetric dequantized data:")
print(np.round(params_deq_symmetric, 2))
print('')
print(f'Quantization error: {utils.quantization_error(params, params_deq_symmetric)}')