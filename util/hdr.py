import numpy as np

def hdr_merge(
    data, 
    model=True, 
    exposure_times=exposure_times, 
    read_ns_m=read_ns_m, 
    bp_m=bp_m,
    gain_cv_m=gain_cv_m
  ):
  # Matrices to hold final values
  w_m = np.zeros((len(exposure_times), 1280, 1024)).astype(np.float32)
  var_m = np.zeros((1280, 1024)).astype(np.float32)
  rad_m = np.zeros((1280, 1024)).astype(np.float32)
  snr_sq_m = np.zeros((1280, 1024)).astype(np.float32)

  ones = np.ones((1280, 1024)) # Matrix of ones to help with calcs

  # Matrices we saved from camera calibrations
  read_sq = read_ns_m if model else np.ones((1280, 1024))*0.00001
  read_sq[read_sq < 0] = 0.00001  # Mask negative read noise to value close to 0
  bp = bp_m
  g = gain_cv_m

  # Our sequence of images to analyze given sequence number i
  seq = data

  # Calculate variance 
  for j, exp in enumerate(exposure_times):
    var_n = seq[j]-bp_m+read_sq                  # DN^2
    temp = np.multiply(np.multiply(g,g), var_n)  # e-^2
    w_m[j,:,:] = np.divide(exp*exp*ones, temp)
    w_m[j,:,:][(seq[j] > 254.5)] = 0             # Mask saturated values
  var_m = np.divide(ones, np.sum(w_m, axis=0))

  # Calculate radiance estimate
  temp = np.zeros((len(exposure_times), 1280, 1024))
  for j, exp in enumerate(exposure_times):
    temp[j,:,:] = np.multiply(w_m[j,:,:], np.divide(np.multiply(seq[j]-bp, g), exp*ones))
  rad_m = np.divide(np.sum(temp, axis=0), np.sum(w_m, axis=0))

  # Use gamma correction as global tone mapping
  rad_toned = cv.pow(rad_m/rad_m.max(), 1.0/2.2)

  # Calculate SNR
  snr_sq_m = np.divide(np.multiply(rad_m, rad_m), var_m)

  return w_m, var_m, rad_m, snr_sq_m, rad_toned