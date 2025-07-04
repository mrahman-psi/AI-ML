import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import os
import torch
import uuid
# ---------------------
# Device Setup
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

output_dir = "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/outputs/images/"


# Plot the velocity map
def plot_velocity(velocity, sample):
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    img = ax.imshow(velocity[sample, 0, :, :], cmap='jet')
    ax.set_xticks(range(0, 70, 10))
    ax.set_xticklabels(range(0, 700, 100))
    ax.set_yticks(range(0, 70, 10))
    ax.set_yticklabels(range(0, 700, 100))
    ax.set_ylabel('Depth (m)', fontsize=12)
    ax.set_xlabel('Offset (m)', fontsize=12)
    clb = plt.colorbar(img, ax=ax)
    clb.ax.set_title('km/s', fontsize=8)
    plt.savefig(os.path.join(output_dir, f"velocity_map_sample_{sample}.png"))
    plt.close()
    plt.figure(figsize=(8, 2.5))
    plt.plot(np.arange(5, 700, 10), np.mean(velocity[sample, 0, :, :], axis=1))
    plt.xlabel("Depth (m)")
    plt.ylabel("Ave Velocity (m/s)")
    plt.savefig(os.path.join(output_dir, f"velocity_profile_sample_{sample}.png"))
    plt.close()

# Get information from the velocity map
def info_velocity(velocity, sample, for_show=True):
    ave_vel = np.mean(velocity[sample, 0, :, :])
    min_vel = np.min(velocity[sample, 0, :, :])
    max_vel = np.max(velocity[sample, 0, :, :])
    medi_vel = np.median(velocity[sample, 0, :, :])
    num_vels = len(np.unique(velocity[sample, 0, :, :]))
    y0_velL = np.mean(velocity[sample, 0, 0, 0:35])
    y0_velR = np.mean(velocity[sample, 0, 0, 35:])
    y09L_medi = np.median(velocity[sample, 0, 0:10, 0:35])
    y09R_medi = np.median(velocity[sample, 0, 0:10, 35:])
    y1029_medi = np.median(velocity[sample, 0, 10:30, :])
    y3049_medi = np.median(velocity[sample, 0, 30:50, :])
    y5069_medi = np.median(velocity[sample, 0, 50:, :])
    y09L_mean = np.mean(velocity[sample, 0, 0:10, 0:35])
    y09R_mean = np.mean(velocity[sample, 0, 0:10, 35:])
    y1029_mean = np.mean(velocity[sample, 0, 10:30, :])
    y3049_mean = np.mean(velocity[sample, 0, 30:50, :])
    y5069_mean = np.mean(velocity[sample, 0, 50:, :])
    if for_show:
        print("Number of distinct velocities: {}".format(num_vels))
        print("Average velocity: {:.2f} m/s".format(ave_vel))
        print(" Median velocity: {:.2f} m/s".format(medi_vel),
              "Min, Max: {:.2f}, {:.2f}".format(min_vel, max_vel))
        print("Ave y=0 velocities L,R: {:.2f}, {:.2f}".format(y0_velL, y0_velR))
        print("Median velocities in rows:  {:.2f}(0-9:L), {:.2f}(0-9:R),".format(
            y09L_medi, y09R_medi),
            "{:.2f}(10-29), {:.2f}(30-49), {:.2f}(50-69)".format(
            y1029_medi, y3049_medi, y5069_medi))
        print("  Mean velocities in rows:  {:.2f}(0-9:L), {:.2f}(0-9:R),".format(
            y09L_mean, y09R_mean),
            "{:.2f}(10-29), {:.2f}(30-49), {:.2f}(50-69)".format(
            y1029_mean, y3049_mean, y5069_mean))
    else:
        return (num_vels, y0_velL, y0_velR, y09L_medi, y09R_medi,
                y1029_medi, y3049_medi, y5069_medi)

# Make a gray-scale image of the seismic data
def plot_data(data, sample=-1):
    fig, ax = plt.subplots(1, 5, figsize=(20, 7))
    if len(data.shape) == 3:
        thisdata = data[:, :, :]
    else:
        thisdata = data[sample, :, :, :]
    maxabs = []
    for srclocid, xloc in enumerate([0, 17, 34, 52, 69]):
        maxabs.append(np.max(np.abs(thisdata[srclocid, 180:, xloc])))
    vrange = np.max(maxabs) * 0.5
    for iax in range(5):
        ax[iax].imshow(thisdata[iax, :, :], extent=[0, 70, 1000, 0],
                       aspect='auto', cmap='gray', vmin=-vrange, vmax=vrange)
    for axis in ax:
        axis.set_xticks(range(0, 70, 10))
        axis.set_xticklabels(range(0, 700, 100))
        axis.set_yticks(range(0, 2000, 1000))
        axis.set_yticklabels(range(0, 2, 1))
        axis.set_ylabel('Time (s)', fontsize=12)
        axis.set_xlabel('Offset (m)', fontsize=12)
    plt.savefig(os.path.join(output_dir, f"seismic_data_sample_{sample}.png"))
    plt.close()

def time_max_peak(isrc, xloc, thisdata):
    ipeak = np.argmax(thisdata[isrc, :, xloc])
    peakvals = thisdata[isrc, ipeak-3:ipeak+4, xloc]
    timevals = np.linspace(ipeak-3, ipeak+3, num=7, endpoint=True)
    if len(peakvals) == len(timevals):
        fitcoefs = np.poly1d(np.polyfit(timevals, peakvals, 2)).coef
        return -0.5 * fitcoefs[1] / fitcoefs[0]
    else:
        print("mis-matched lengths:\n", timevals, "\n", peakvals)
        return ipeak

def info_data(data, sample=-1, for_show=True):
    if len(data.shape) == 3:
        thisdata = data[:, :, :]
    else:
        thisdata = data[sample, :, :, :]
    partdata = thisdata[:, 0:225, :]
    vsurfaceL = 4*5*10*1000 / (
        time_max_peak(0, 6, partdata) - time_max_peak(0, 1, partdata) +
        time_max_peak(1, 11, partdata) - time_max_peak(1, 16, partdata) +
        time_max_peak(1, 23, partdata) - time_max_peak(1, 18, partdata) +
        time_max_peak(2, 28, partdata) - time_max_peak(2, 33, partdata)
    )
    vsurfaceR = 4*5*10*1000 / (
        time_max_peak(2, 40, partdata) - time_max_peak(2, 35, partdata) +
        time_max_peak(3, 46, partdata) - time_max_peak(3, 51, partdata) +
        time_max_peak(3, 58, partdata) - time_max_peak(3, 53, partdata) +
        time_max_peak(4, 63, partdata) - time_max_peak(4, 68, partdata)
    )
    vsurface = (vsurfaceL + vsurfaceR) / 2
    if for_show:
        idists = np.arange(0, 70)
        dists = []
        times = []
        timeref = np.argmax(thisdata[2, :, 34])
        for idist in idists:
            dists.append(10 * idist)
            times.append(np.sign(idist-34) * (np.argmax(thisdata[2, :, idist]) - timeref) / 1000.0)
        times = -1.0 * (np.array(times) - times[0])
        plt.figure(figsize=(6, 3))
        plt.plot(dists, times, '.b', alpha=0.7)
        plt.plot([dists[0], dists[-1]], [times[0], times[-1]], c='orange', alpha=0.6)
        plt.ylabel("$-$ Time (s)")
        plt.xlabel("Surface Distance (m)")
        plt.title("Time vs Distance from Source 2")
        plt.savefig(os.path.join(output_dir, f"time_vs_distance_sample_{np.random.randint(100000)}.png"))
        plt.close()
        print("Surface velocities : {:.2f}-Left, {:.2f}-Average, {:.2f}-Right".format(
            vsurfaceL, vsurface, vsurfaceR))
    else:
        return vsurfaceL, vsurface, vsurfaceR

def sources_data(data, sample=-1, for_show=True):
    if len(data.shape) == 3:
        thisdata = data[:, :, :]
    else:
        thisdata = data[sample, :, :, :]
    maxamps = []
    minamps = []
    for srclocid, xloc in enumerate([0, 17, 34, 52, 69]):
        maxamps.append(np.max(thisdata[srclocid, 180:, xloc]))
        minamps.append(np.min(thisdata[srclocid, 180:, xloc]))
    max_amp = np.max(maxamps)
    min_amp = np.min(minamps)
    delta_amp = 0.05 * (max_amp - min_amp)
    plt.figure(figsize=(8, 5))
    for srclocid, xloc in enumerate([0, 17, 34, 52, 69]):
        timeseries = thisdata[srclocid, :, xloc]
        offset = delta_amp * (xloc - 34) / 35.0
        plt.plot(np.array(range(1000)), timeseries + offset, alpha=0.7)
    plt.plot([0, 1000], [0.0, 0.0], c='gray', alpha=0.5)
    plt.ylim(1.10 * min_amp - delta_amp, 1.10 * max_amp + delta_amp)
    plt.xlabel('Time (ms)')
    plt.ylabel("Amplitude     Traces are offset.")
    plt.title("Waveforms at the 5 source locations")
    # Save the plot to output_dir with a unique name
    unique_name = f"waveforms_{uuid.uuid4().hex[:8]}.png"
    plt.savefig(os.path.join(output_dir, unique_name))
    plt.close()

last_data_file = "None"

def get_train_sample(dfind, ftscale=True):
    global velocity, data, last_data_file
    #D:\Personal\Mamun\Training\AI ML Bootcamp\Python\Workspace\Yale Project\train_samples
    train_dir = r"D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/train_samples/"
    output_dir = "D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/outputs/images/"
    veltype, ifile, isample = traindf.loc[dfind, ["veltype", "ifile", "isample"]]
    if ("Vel" in veltype) or ("Style" in veltype):
        data_file = train_dir + veltype + "/data/data" + str(ifile) + ".npy"
        model_file = train_dir + veltype + "/model/model" + str(ifile) + ".npy"
    else:
        fault_num = 2 * ifile + 4 * ("_B" in veltype)
        data_file = train_dir + veltype + "/seis" + str(fault_num) + "_1_0.npy"
        model_file = train_dir + veltype + "/vel" + str(fault_num) + "_1_0.npy"
    if data_file != last_data_file:
        data = np.load(data_file)
        if ftscale:
            for itime in range(1000):
                data[:, :, itime, :] = (1.0 + (itime / 200) ** 1.5) * data[:, :, itime, :]
        velocity = np.load(model_file)
        last_data_file = data_file
    return velocity, data, isample

# The rest of your code (dataframe creation, feature extraction, modeling, plotting, and submission)
# should be kept only once, not repeated. 
# Please ensure you only keep one copy of each logical block and function as above.
# There are 10,000 training samples on kaggle, organized as: 10 x 2 x 500 data-vel pairs
##!ls /kaggle/input/waveform-inversion/train_samples/*

# Make a dataframe with 10,000 rows labeled by:
#   type - 5 x 2 string values
#   ifile - two numeric values: 0,1 or 1,2 or 2,4 or 6,8 depending on type
#   isample - 0 to 499
veltypes = ["FlatVel","FlatFault", "CurveVel", "CurveFault", "Style"]
veltype = []; ifile = []; isample = []
for this_type in veltypes:
    for this_AB in ["_A","_B"]:
        for this_ifile in [1,2]:
            for this_isample in range(500):  # **************************************
                veltype.append(this_type+this_AB); ifile.append(this_ifile); isample.append(this_isample)
# Make a dataframe from these
traindf = pd.DataFrame({"veltype":veltype, "ifile":ifile, "isample":isample})
# Select a dataframe index to look at
dfind = int(0.87*len(traindf))


print(list(traindf.loc[dfind,["veltype","ifile","isample"]]))
velocity, data, isample = get_train_sample(dfind)

print('Velocity map size:', velocity.shape)
print('Seismic data size:', data.shape)

# Look at the velocity map for the training sample
# isample defined above
plot_velocity(velocity, isample)
info_velocity(velocity, isample)

# Look at the seismic data for the sample
# isample = same as for the velocity map above
plot_data(data, isample)
info_data(data, isample)
sources_data(data, isample)

# Look at one of them (they seem to be shuffled)
##testdata = np.load('/kaggle/input/waveform-inversion/test/000039dca2.npy')  # messy
##testdata = np.load('/kaggle/input/waveform-inversion/test/0001026c8a.npy')  # very simple
testdata = np.load('D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/test/00015b24d5.npy')  # weird straight lines
##testdata = np.load('/kaggle/input/waveform-inversion/test/800222ab0d.npy')  # messy
##testdata = np.load('/kaggle/input/waveform-inversion/test/a00269f1eb.npy')  # messy
##testdata = np.load('/kaggle/input/waveform-inversion/test/c0021521e5.npy')  # simple-ish

# Scale the seismic data by ~ (1+(t/a)^b) to help equalize the amplitudes vs time.
# (This is similar to applying AGC for visualization, but is included in the analysis too.)
for itime in range(1000):
    testdata[ : , itime, : ] = (1.0+(itime/200)**1.5)*testdata[ : , itime, : ]

print('Test data size:', testdata.shape)

plot_data(testdata)
info_data(testdata)
sources_data(testdata)

# For each sample,
# add y_ targets: y0_velL, y0_velR, y09L_medi, y09R_medi, y1039_medi, y4069_medi
nunique = []; y0_aves = []; y0_diffs = []
y09L_medis = []; y09R_medis = []; y1029_medis = []; y3049_medis = []; y5069_medis = []
# add x_ features: surface velocity average and R-L difference
surf_aves = []; surf_diffs = []
for dfind in traindf.index:
    velocity, data, isample = get_train_sample(dfind, ftscale=False)
    # velocity, target, values
    (num_vels, y0_velL, y0_velR, y09L_medi, y09R_medi,
         y1029_medi, y3049_medi, y5069_medi) = info_velocity(
                                            velocity, isample, for_show=False)
    nunique.append(num_vels)
    y0_aves.append((y0_velL + y0_velR)/2); y0_diffs.append(y0_velR - y0_velL)
    y09L_medis.append(y09L_medi); y09R_medis.append(y09R_medi); y1029_medis.append(y1029_medi)
    y3049_medis.append(y3049_medi); y5069_medis.append(y5069_medi)
    # seismic, feature, values
    velL, velave, velR = info_data(data, isample, for_show=False)
    surf_aves.append(velave)
    surf_diffs.append(velR-velL)
traindf["y_numVels"] = nunique
traindf["y_y0Ave"] = y0_aves
traindf["y_y0Diff"] = y0_diffs
traindf["y_09LMedi"] = y09L_medis
traindf["y_09RMedi"] = y09R_medis
traindf["y_1029Medi"] = y1029_medis
traindf["y_3049Medi"] = y3049_medis
traindf["y_5069Medi"] = y5069_medis
traindf["x_surfAve"] = surf_aves
traindf["x_surfDiff"] = surf_diffs

# Add color-coding based on the surfDiff and surfAve values
# Red = R-L not zero; Blue = R-L near zero
traindf["diff_clr"] = 'red'
# Use measured R-L difference to set color
seldiff = traindf["x_surfAve"] > (1300.0 + 1200.0*np.log10(1+np.abs(traindf["x_surfDiff"])))
# Use known R-L difference from the target (can't do this for test)
##seldiff = (np.abs(traindf["y_y0Diff"]) < 0.1*diff_color_change)
traindf.loc[seldiff, "diff_clr"] = 'blue'

# Print the dataframe
print(traindf.head(10))

# Summary values for the columns
traindf_means = traindf.describe().loc["mean",]
traindf.describe()

# Save the Training Dataframe
traindf.to_csv("traindf.csv", header=True, index=False, float_format='%.2f')
# For plots
velocity_range = (1400,4600)
print("\nMedian Ave Surface Velocity: {:.2f}".format(np.median(traindf["x_surfAve"])))
print("Average Ave Surface Velocity: {:.2f}\n".format(np.mean(traindf["x_surfAve"])))

diffs = traindf["x_surfDiff"]

plt.figure(figsize=(8,4))
plt.hist(traindf["x_surfAve"],bins=100)
plt.title("Train: Histogram of the Average Surface Velocity")
plt.xlabel("Surface Velocity (m/s)")
plt.xlim(velocity_range)
plt.savefig(os.path.join(output_dir, f"train_hist_surface_velocity.png"))
#plt.show()

plt.figure(figsize=(8,4))
plt.hist(np.sign(diffs)*np.log10(np.abs(diffs) + 1.0), log=True, bins=100)
plt.title("Train: Histogram of the R-L Velocity Difference")
plt.xlabel("Signed Log10[1+ R-L Surface Velocity Difference (m/s) ]")
plt.savefig(os.path.join(output_dir, f"train_hist_velocity_difference.png"))
#plt.show()

# Start Cell 18

plt.figure(figsize=(8,5))
plt.scatter( np.sign(diffs)*(np.log10(np.abs(diffs) + 1.0)), traindf["x_surfAve"],
                             color=traindf["diff_clr"], s=2, alpha=0.25)
lindiffs = np.linspace(-3.0,3.0,100)  # <-- This is log10(1+ abs(surfDiff) )
plt.plot(lindiffs, 1300.0 + 1200.0*np.abs(lindiffs),c='gray',alpha=0.5)
plt.ylabel("Average Surface Velocity (m/s)")
plt.xlabel("Signed Log10[1+ R-L Velocity Difference (m/s) ]")
plt.title("Train: Average Surface Velocity vs. R-L Velocity Difference")
plt.ylim(velocity_range)
plt.savefig(os.path.join(output_dir, f"train_scatter_velocity_vs_difference.png"))  
#plt.show()

# Look into the y=0 Average and Difference

# Scatter plot of the y=0 row Average and y=0 R-L Difference velocities
if True:
    diffs = traindf["y_y0Diff"]

    plt.figure(figsize=(6,3))
    plt.scatter( np.sign(diffs)*(np.log10(np.abs(diffs) + 1.0)), traindf["y_y0Ave"]/1000,
                             color=traindf["diff_clr"], s=2, alpha=0.25)
    plt.ylabel("Ave y=0 Velocity (km/s)")
    plt.xlabel("Signed Log10[1+ y=0 R-L Velocity Diff (m/s) ]")
    plt.title("Train: y=0 Average Velocity vs. y=0 R-L Velocity Difference")
    #plt.savefig(os.path.join(output_dir, f"train_y0_scatter_velocity_vs_diff.png"))
    plt.ylim(1.4,4.6) # in km/s
    plt.savefig(os.path.join(output_dir, f"train_y0_scatter_velocity_vs_diff.png"))
    #plt.show()

    # Histogra of the y=0 R-L Diff
    plt.figure(figsize=(6,3))
    plt.hist(np.sign(diffs)*np.log10(np.abs(diffs) + 1.0), log=True, bins=100)
    plt.title("Train: Histogram of the y=0 R-L Velocity Difference")
    plt.xlabel("Signed Log10[1+ R-L y=0 Velocity Difference (m/s) ]")
    plt.savefig(os.path.join(output_dir, f"train_hist_y0_difference.png"))  
    #plt.show()

    # Scatter plot of the Seismic R-L Diff vs the y=0 R-L Diff
    diffs = traindf["x_surfDiff"]
    diffy0 = traindf["y_y0Diff"]

    plt.figure(figsize=(6,3))
    plt.scatter( np.sign(diffy0)*(np.log10(np.abs(diffy0) + 1.0)),
                    np.sign(diffs)*(np.log10(np.abs(diffs) + 1.0)),
                             color=traindf["diff_clr"], s=2, alpha=0.25)
    plt.xlabel("y=0  Log10[1+ R-L Velocity Diff (m/s) ]")
    plt.ylabel("Seismic  Log10[1+ R-L Velocity Diff (m/s) ]")
    plt.title("Train: Seismic R-L Difference vs the y=0 R-L Difference")
    plt.savefig(os.path.join(output_dir, f"train_y0_scatter_velocity_vs_diff.png"))
    #plt.show()
# Find some with y=0 diff = 0 and yet seismic R-L is high
##traindf[(traindf["y_y0Diff"] == 0) & (traindf["x_surfDiff"] > 200)]

# Simple degree 1 polynomial fit 
model = np.poly1d(np.polyfit(np.array(traindf["y_y0Ave"]), 
                             np.array(traindf["x_surfAve"]), 1))
# for polynomial line visualization 
polyline = np.linspace(1400, 4500, 100)  

plt.figure(figsize=(4,4))
plt.scatter( traindf["y_y0Ave"], traindf["x_surfAve"],
                color=traindf["diff_clr"], s=2, alpha=0.25)
plt.plot(polyline, model(polyline), c='orange',alpha=0.6)
plt.xlabel("y=0 Average Velocity")
plt.ylabel("Seismic Average Surface Velocity (m/s)")
plt.title("Train: Seismic Surface Velocity vs. y=0 Velocity")
plt.xlim(velocity_range)
plt.ylim(velocity_range)
plt.savefig(os.path.join(output_dir, f"train_surf_vs_y0.png"))
#plt.show()

print("   Fit coefs [slope, intercept]:", model.coef,"\n")

# Compare the velocities in row ranges vs the surface velocity and velocity difference.
# Create simple model fits for each region.

surfAves = traindf["x_surfAve"]
surfDiffs = traindf["x_surfDiff"]
log_surfDiffs = np.sign(surfDiffs)*(np.log10(np.abs(surfDiffs) + 1.0))

# Fit red, blue separately, limit the surfAve range used
selblue = (traindf["diff_clr"] == 'blue') & (traindf["x_surfAve"] < 4000)
selred = (traindf["diff_clr"] == 'red') & (traindf["x_surfAve"] < 4000)

# for polynomial line visualization 
polyline = np.linspace(1400, 4000, 100)

# Save the fit models
rows_models = []
for y_rows in ["09L", "09R", "1029", "3049", "5069"]:
    
    rows_values = traindf["y_"+y_rows+"Medi"]
    surf_values = surfAves.copy()
    vel_axis_label = "Ave Surface Velocity (m/s)"
    # Modify surf_values for the 09L,R data
    if "09L" in y_rows:
        surf_values = surf_values - 0.5*surfDiffs
        vel_axis_label = "L Surface Velocity (m/s)"
    if "09R" in y_rows:
        surf_values = surf_values + 0.5*surfDiffs
        vel_axis_label = "R Surface Velocity (m/s)"
   
    plt.figure(figsize=(7,4))
    plt.scatter(surf_values, rows_values, color=traindf["diff_clr"], s=2, alpha=0.25)

    degree = 3
    # Blue polynomial fit:
    if "09" in y_rows:
        # Use combined L and R data for the model, selblue:
        surf_RLvalues = np.concatenate( ( (surfAves - 0.5*surfDiffs)[selblue], 
                                            (surfAves + 0.5*surfDiffs)[selblue] ) )
        rows_RLvalues = np.concatenate( ( traindf.loc[selblue,"y_09LMedi"], 
                                            traindf.loc[selblue,"y_09RMedi"] ) )
        model = np.poly1d(np.polyfit(surf_RLvalues, 
                                        rows_RLvalues, degree))
    else:
        model = np.poly1d(np.polyfit(np.array(surf_values[selblue]), 
                             np.array(rows_values[selblue]), degree))
    
    rows_models.append(model)
    blue_resids = (-1.0*model(np.array(surf_values[selblue])) + 
                             np.array(rows_values[selblue]))
    print("  Blue Fit coefs:", model.coef)
    plt.plot(polyline, model(polyline), c='blue',alpha=0.6)
    #
    # Red polynomial fit:
    if "09" in y_rows:
        # Use combined L and R data for the model, selred:
        surf_RLvalues = np.concatenate( ( (surfAves - 0.5*surfDiffs)[selred], 
                                            (surfAves + 0.5*surfDiffs)[selred] ) )
        rows_RLvalues = np.concatenate( ( traindf.loc[selred,"y_09LMedi"], 
                                            traindf.loc[selred,"y_09RMedi"] ) )
        model = np.poly1d(np.polyfit(surf_RLvalues, 
                                        rows_RLvalues, degree))
    else:
        model = np.poly1d(np.polyfit(np.array(surf_values[selred]), 
                             np.array(rows_values[selred]), degree))
rows_models.append(model)
red_resids = (-1.0*model(np.array(surf_values[selred])) + 
                             np.array(rows_values[selred]))
print("  Red Fit coefs:", model.coef)
plt.plot(polyline, model(polyline), c='purple',alpha=0.6)

plt.xlabel(vel_axis_label)
plt.xlim(1400, 4100) # reduce because of fitting range
plt.ylabel("y_"+y_rows+" Median")
plt.ylim(velocity_range)
plt.title("Train: y_"+y_rows+" Median vs. Surface Velocity")
plt.savefig(os.path.join(output_dir, f"train_rows"+y_rows+"_vs_average.png"))
#plt.show()


# Show the residuals vs surface difference for the 09L, 09R
if "09" in y_rows:
    plt.figure(figsize=(7,2))
    plt.scatter( log_surfDiffs[selblue], blue_resids,
                            color=traindf.loc[selblue,"diff_clr"], s=2, alpha=0.25)
    plt.scatter( log_surfDiffs[selred], red_resids,
                            color=traindf.loc[selred,"diff_clr"], s=2, alpha=0.25)
    plt.ylim(-1000,1000)
    plt.xlabel("Signed Log10[1+ R-L Velocity Diff (m/s) ]")
    plt.ylabel("y_"+y_rows+" Residuals")
    plt.title("Train: y_"+y_rows+" * Residuals * vs. Surface Difference")
    plt.savefig(os.path.join(output_dir, f"train_residuals"+y_rows+"_vs_difference.png"))
    #plt.show()
    
    
# Show the median values vs surface difference
plt.figure(figsize=(7,2))
plt.scatter(log_surfDiffs, rows_values,
                            color=traindf["diff_clr"], s=2, alpha=0.25)
plt.xlabel("Signed Log10[1+ R-L Velocity Diff (m/s) ]")
plt.ylabel("y_"+y_rows+" Median")
plt.ylim(velocity_range)
plt.title("Train: y_"+y_rows+" Median vs. Surface Difference")
plt.savefig(os.path.join(output_dir, f"train_rows"+y_rows+"_vs_difference.png"))
#plt.show()
print("\n")

polyline = np.linspace(1400, 4000, 100)  
plt.figure(figsize=(6,3))
num_models = len(rows_models) // 2
for imod in range(num_models):
    plt.plot(polyline, rows_models[2*imod](polyline), c='blue',alpha=0.6)
    plt.plot(polyline, rows_models[2*imod+1](polyline), c='red',alpha=0.6)
plt.xlabel("Average Surface Velocity (m/s)")
plt.ylabel("Median of Rows")
plt.title("Fits of Row-Ranges Medians vs. Surface Velocity")
plt.savefig(os.path.join(output_dir, f"Average Surface Velocity.png"))
#plt.show()

# What/why are the blue lines in the 1029 and 3049 median vs surface velocity plots?
# Find the samples in these lines
trainblue = traindf[traindf["diff_clr"] == 'blue']

print("\n\n  Look for 'blue' samples that have Rows Medians equal to the y=0 Average.")
print("  - List the counts of Velocity-Map Types.")
print("  - Check the y0Diff values: they are all 0, so vmaps are R-L symmetric.\n\n")

for yrows in ["09L","09R","1029","3049","5069"]:
    plt.figure(figsize=(6,2))
    plt.hist(np.clip(trainblue["y_"+yrows+"Medi"] - trainblue["y_y0Ave"],-800,800),
             log=True, bins=160)
    plt.xlim(-500,500)
    plt.xlabel("Rows "+yrows+" Median  -  y=0 Average")
    plt.savefig(os.path.join(output_dir, f"train_hist_{yrows}_difference.png"))
    #plt.show()

    matchdf = trainblue[np.abs(trainblue["y_"+yrows+"Medi"] - trainblue["y_y0Ave"]) < 0.0001]
    print(matchdf["veltype"].value_counts())
    print(matchdf["y_y0Diff"].value_counts())

# Use the sample submission to get the test ids
# Print 
submis = pd.read_csv("D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/waveform-inversion/sample_submission.csv")

# Print the first 10 rows of the sample submission
#print("\nSample Submission Dataframe:\n", submis.head(10))

# Create a df of just the test ids (with _y_0)
oiddf = submis.loc[0:4607260:70,["oid_ypos"]].copy()
oiddf = oiddf.reset_index(drop=True)

# For each sample, add the measured surface velocity average and the R-L difference
ave_vels = []
diff_vels = []
for indoid in oiddf.index:
    testdata = np.load('D:/Personal/Mamun/Training/AI ML Bootcamp/Python/Workspace/Yale Project/test/' + 
                   oiddf.loc[indoid,"oid_ypos"][0:10]+'.npy')
    velL, velave, velR = info_data(testdata, for_show=False)
    ave_vels.append(velave)
    diff_vels.append(velR-velL)

oiddf["x_surfAve"] = ave_vels
oiddf["x_surfDiff"] = diff_vels

# Add color-coding based on the surfDiff and surfAve values
oiddf["diff_clr"] = 'red'
# Set blue, same criteria as for the training data
seldiff = oiddf["x_surfAve"] > (1300.0 + 1200.0*np.log10(1+np.abs(oiddf["x_surfDiff"])))
oiddf.loc[seldiff, "diff_clr"] = 'blue'

# Add model-predicted columns to the test dataframe based on x_surfAve
# Order is blue then red model for each rows range.
surf_values = oiddf["x_surfAve"]
surf_diffs = oiddf["x_surfDiff"]
surf_L_values = surf_values - 0.5*surf_diffs
surf_R_values = surf_values + 0.5*surf_diffs
selblue = oiddf["diff_clr"] == 'blue'
# Blue and Red model for each row range
# for imod, y_rows in enumerate(["09L", "09R", "1029", "3049", "5069"]):
#     if y_rows == "09L":
#         oiddf.loc[selblue,"pred_"+y_rows] = rows_models[2*imod](surf_L_values[selblue])
#         oiddf.loc[~selblue,"pred_"+y_rows] = rows_models[2*imod+1](surf_L_values[~selblue])
#     elif y_rows == "09R":
#         oiddf.loc[selblue,"pred_"+y_rows] = rows_models[2*imod](surf_R_values[selblue])
#         oiddf.loc[~selblue,"pred_"+y_rows] = rows_models[2*imod+1](surf_R_values[~selblue])
#     else:
#         oiddf.loc[selblue,"pred_"+y_rows] = rows_models[2*imod](surf_values[selblue]).values
#         oiddf.loc[~selblue,"pred_"+y_rows] = rows_models[2*imod+1](surf_values[~selblue]).values

for imod, y_rows in enumerate(["09L", "09R", "1029", "3049", "5069"]):
    if y_rows == "09L":
        oiddf.loc[selblue, "pred_" + y_rows] = rows_models[2 * imod](surf_L_values[selblue])
        oiddf.loc[~selblue, "pred_" + y_rows] = rows_models[2 * imod + 1](surf_L_values[~selblue])
    elif y_rows == "09R":
        oiddf.loc[selblue, "pred_" + y_rows] = rows_models[2 * imod](surf_R_values[selblue])
        oiddf.loc[~selblue, "pred_" + y_rows] = rows_models[2 * imod + 1](surf_R_values[~selblue])
    else:
        oiddf.loc[selblue, "pred_" + y_rows] = rows_models[2 * imod](surf_values[selblue])
        oiddf.loc[~selblue, "pred_" + y_rows] = rows_models[2 * imod + 1](surf_values[~selblue])


# Save the Test Dataframe
oiddf.to_csv("oiddf.csv", header=True, index=False, float_format='%.2f')

print("\nMedian Ave Surface Velocity: {:.2f}".format(np.median(oiddf["x_surfAve"])))
print("Average Ave Surface Velocity: {:.2f}\n".format(np.mean(oiddf["x_surfAve"])))

plt.figure(figsize=(8,4))
plt.hist(oiddf["x_surfAve"],bins=100)
plt.title("Test: Histogram of the Average Surface Velocity")
plt.xlabel("Surface Velocity (m/s)")
plt.xlim(velocity_range)
plt.savefig(os.path.join(output_dir, f"test_hist_surface_velocity.png"))
#plt.show()

plt.figure(figsize=(8,4))
diffs = oiddf["x_surfDiff"]
plt.hist(np.sign(diffs)*np.log10(np.abs(diffs) + 1.0), log=True, bins=100)
plt.title("Test: Histogram of the R-L Velocity Difference")
plt.xlabel("Signed Log10[1+ R-L Surface Velocity Difference (m/s) ]")
plt.savefig(os.path.join(output_dir, f"test_hist_velocity_difference.png"))
#plt.show()

plt.figure(figsize=(8,5))
plt.scatter( np.sign(diffs)*(np.log10(np.abs(diffs) + 1.0)), oiddf["x_surfAve"],
                             color=oiddf["diff_clr"], s=2, alpha=0.15)
plt.ylabel("Ave Surface Velocity (m/s)")
plt.xlabel("Signed Log10[1+ R-L Velocity Diff (m/s) ]")
plt.title("Test: Average Surface Velocity vs R-L Velocity Difference")
plt.ylim(velocity_range)
plt.savefig(os.path.join(output_dir, f"test_scatter_velocity_vs_difference.png"))   
#plt.show()

# Enter the predictions into the submis dataframe
# The predictions are 3 values for each of the 65818 test samples:
# pred_09L value --> rows 0-9
# pred_09R value --> rows 0-9
# pred_1029 value --> rows 10-29
# pred_3049 value --> rows 30-49
# pred_5069 value --> rows 50-69

# For each range of rows,
# fill all 35 x_j values of the 65818 y_i values with the 65818 predicted values.
all_xs = list(submis.columns[1:])
left_xs = list(submis.columns[1:17+1])
right_xs = list(submis.columns[18:])

# Loop over each set of y_i rows and set them equal to the corresponding predicted values

len_oiddf = len(oiddf)

# Rows 0-9, with values adjusted for L (1,3,...,33) and R (35,37,...,69) halves.
fill_values = (np.ones([17,len_oiddf]) * np.array(oiddf["pred_09L"])).T
for iy in range(10):
    rowsel = (submis.index % 70) == iy
    submis.loc[rowsel, left_xs] = fill_values
fill_values = (np.ones([18,len_oiddf]) * np.array(oiddf["pred_09R"])).T
for iy in range(10):
    rowsel = (submis.index % 70) == iy
    submis.loc[rowsel, right_xs] = fill_values


# Rows 10-29
fill_values = (np.ones([35,len_oiddf]) * np.array(oiddf["pred_1029"])).T
for iy in range(10,29+1):
    rowsel = (submis.index % 70) == iy
    submis.loc[rowsel, all_xs] = fill_values
    
# Rows 30-49
fill_values = (np.ones([35,len_oiddf]) * np.array(oiddf["pred_3049"])).T
for iy in range(30,49+1):
    rowsel = (submis.index % 70) == iy
    submis.loc[rowsel, all_xs] = fill_values

# Rows 50-69
fill_values = (np.ones([35,len_oiddf]) * np.array(oiddf["pred_5069"])).T
for iy in range(50,69+1):
    rowsel = (submis.index % 70) == iy
    submis.loc[rowsel, all_xs] = fill_values

# Generate the submission file
#submis.to_csv("submission.csv", header=True, index=False, float_format='%.0f')    