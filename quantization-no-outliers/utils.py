import numpy as np
def clamp(params_q: np.array, lower_bound: int, upper_bound: int) -> np.array:
    params_q[params_q < lower_bound] = lower_bound
    params_q[params_q > upper_bound] = upper_bound
    return params_q

def asymmetric_quantization(params: np.array, bits: int):
    # Calculate the scale and zero point
    alpha = np.max(params)
    beta = np.min(params)
    scale = (alpha - beta) / (2**bits-1)
    zero = -1*np.round(beta / scale)
    lower_bound, upper_bound = 0, 2**bits-1
    # Quantize the parameters
    quantized = clamp(np.round(params / scale + zero), lower_bound, upper_bound).astype(np.int32)
    return quantized, scale, zero

def asymmetric_quantization_percentile(params: np.array, bits: int, percentile: float = 99.99):
    # find the percentile value
    alpha = np.percentile(params, percentile)
    beta = np.percentile(params, 100-percentile)
    scale = (alpha - beta) / (2**bits-1)
    zero = -1*np.round(beta / scale)
    lower_bound, upper_bound = 0, 2**bits-1
    quantized = clamp(np.round(params / scale + zero), lower_bound, upper_bound).astype(np.int32)
    return quantized, scale, zero

def asymmetric_dequantize(params_q: np.array, scale: float, zero: int) -> np.array:
    return (params_q - zero) * scale

def symmetric_dequantize(params_q: np.array, scale: float) -> np.array:
    return params_q * scale

def symmetric_quantization(params: np.array, bits: int):
    # Calculate the scale
    alpha = np.max(np.abs(params))
    scale = alpha / (2**(bits-1)-1)
    lower_bound = -2**(bits-1)
    upper_bound = 2**(bits-1)-1
    # Quantize the parameters
    quantized = clamp(np.round(params / scale), lower_bound, upper_bound).astype(np.int32)
    return quantized, scale

def quantization_error(params: np.array, params_q: np.array):
    # calculate the MSE
    return np.mean((params - params_q)**2)