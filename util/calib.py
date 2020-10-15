import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit


def calc_stats(DIR, exp,  num=100):
  """Calculates the min, max, std, avg per pixel per exposure, default 200 imgs"""
  min_m = np.zeros((1280, 1024)).astype(np.uint8)
  max_m = np.zeros((1280, 1024)).astype(np.uint8)
  std_m = np.zeros((1280, 1024)).astype(np.float16)
  avg_m = np.zeros((1280, 1024)).astype(np.float16)
  snr_m = np.zeros((1280, 1024)).astype(np.float16)
  sig_m = np.zeros((1280, 1024)).astype(np.float16)
  all_m = []

  for i in range(1, num+1):
    try:
      mat_bp = loadmat(DIR+str(exp)+'ms_%d.mat' % i)['color']
    except:
      mat_bp = loadmat(DIR+str(exp)+'ms_%d.mat' % i)['image']
    all_m.append(mat_bp)
  all_m = np.array(all_m).astype(np.uint8)

  for x in range(1280):
    for y in range(1024):
      min_m[x,y] = np.min(all_m[:,x,y])
      max_m[x,y] = np.max(all_m[:,x,y])
      std_m[x,y] = np.std(all_m[:,x,y])
      avg_m[x,y] = np.average(all_m[:,x,y])
      snr_m[x,y] = avg_m[x,y]/std_m[x,y] if avg_m[x,y] < 245.5 and std_m[x,y] > 0 else 0
      sig_m[x,y] = std_m[x,y]*std_m[x,y]

  np.save(DIR+'processed/'+str(exp)+'ms_min.npy', min_m, allow_pickle=False)
  np.save(DIR+'processed/'+str(exp)+'ms_max.npy', max_m, allow_pickle=False)
  np.save(DIR+'processed/'+str(exp)+'ms_std.npy', std_m, allow_pickle=False)
  np.save(DIR+'processed/'+str(exp)+'ms_avg.npy', avg_m, allow_pickle=False)
  np.save(DIR+'processed/'+str(exp)+'ms_all.npy', all_m, allow_pickle=False)
  np.save(DIR+'processed/'+str(exp)+'ms_snr.npy', snr_m, allow_pickle=False)
  np.save(DIR+'processed/'+str(exp)+'ms_sig.npy', sig_m, allow_pickle=False)

  return min_m, max_m, std_m, avg_m, snr_m, sig_m