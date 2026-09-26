> **Not applicable to the current recordings.** The plan below asks for at least 200 cycles of a
> repeated movement, and largest-Lyapunov estimates depend on series length (detecting condition
> effects needed more than 150 strides in Bruijn et al., 2009). The present recordings are one
> performance of a form in which every movement occurs once — about 28 chest turns, none of them a
> repetition — and the form is not a stationary cyclic task. It is kept for a dedicated long cyclic
> recording (see Recommendations_for_next_recordings.md); for short, non-stationary data, recurrence
> quantification analysis (Webber & Zbilut, 1994; Riley et al., 1999) is the more appropriate family.

Implementation of the Rosenstein algorithm for calculating the largest Lyapunov exponent (LyE) from IMU data, 
tailored specifically for Tai Chi movements. 
The goal is to quantify **local dynamic stability** – how well a practitioner maintains consistent, controlled movement despite continuous perturbations.

---

## 1. Preliminary Setup and Hardware Validation

Before implementation, ensure your hardware and software environment meet the specific demands of nonlinear time‑series analysis.

| **Parameter** | **Recommendation** | **Justification / Notes** |
| :--- | :--- | :--- |
| **Sensors** | 6‑axis IMU (gyro + accel) per segment | Minimum required for orientation tracking. Magnetometers are optional; they can be used for initial calibration but may become unreliable in complex Tai Chi environments. |
| **Sampling Rate** | **≥100 Hz** | Lower rates bias LyE estimates and degrade reconstruction quality; a previous study found IMU data undersampled at 30 Hz produced significantly different LyE values during over‑ground walking. For Tai Chi, 100 Hz captures the relatively slow joint velocities while still preserving subtle fluctuations. |
| **Data Duration** | ≥200 gait cycles (or repetitions) | Embedding dimension stabilisation requires a minimum data length. For walking, 200–300 gait cycles are recommended. For Tai Chi, ensure the recorded segment contains at least 200 repetitive cycles of the studied form. |
| **Preprocessing** | **No low‑pass filtering** | Filtering removes high‑frequency variability that is an essential component of local stability. Use raw data or a very high cut‑off (e.g., 10–15 Hz for Tai Chi). |
| **Software** | Python (NumPy, SciPy, scikit‑learn, matplotlib)| provides efficient nearest‑neighbour search and linear algebra routines. Python is preferred for multi‑sensor pipelines and integration with sensor‑fusion libraries. |

---

## 2. Phase Space Reconstruction (Takens Embedding)

The core of the Rosenstein algorithm is the reconstruction of the system’s phase space from a single observed variable. For each IMU sensor’s time series \(x_1, x_2, \dots, x_N\) (e.g., sagittal plane angular velocity), we form delay vectors:

$$
Y(i) = [x_i,\ x_{i+\tau},\ x_{i+2\tau},\ \dots,\ x_{i+(m-1)\tau}], \quad i = 1, \dots, M
$$

where \(M = N - (m-1)\tau\).

| **Parameter** | **Symbol** | **Estimation Method** | **Implementation (Python)** |
| :--- | :--- | :--- | :--- |
| **Time delay** | \(\tau\) | First minimum of the **average mutual information (AMI)** | `nlmeasures.mutual_information()` from `nlmeasures`; or compute in Python using `scipy.stats.entropy` with a binning grid |
| **Embedding dimension** | \(m\) | **False Nearest Neighbors (FNN)** | `nlmeasures.false_nearest_neighbors()`; or implement the Kennel algorithm |

> **Important:** In gait and similar repetitive movements, using a fixed number of **normalised cycles** yields more stable delay and dimension values than using a fixed number of raw data points. For Tai Chi, normalise each movement cycle to the same number of samples (e.g., 100 samples per repetition).

---

## 3. Rosenstein Algorithm: Step‑by‑Step

The algorithm estimates the largest Lyapunov exponent \(\lambda_{\max}\) by measuring the average exponential divergence of initially close trajectories in the reconstructed phase space.

### 3.1 Nearest Neighbour Search with Temporal Exclusion

For each embedded vector \(Y(i)\) find its nearest neighbour \(Y(j)\) with the constraint that \(|i - j| > T_{\text{sep}}\). \(T_{\text{sep}}\) is typically set to the **mean period** of the signal (in samples), derived from the dominant frequency of the FFT. This prevents the algorithm from selecting temporally correlated vectors.

```python
import numpy as np
from scipy.spatial import KDTree

def find_neighbors(Y, min_sep):
    tree = KDTree(Y)
    neighbors = []
    for i, y in enumerate(Y):
        # find all neighbors within a distance; skip those too close in time
        # (Simplified: find the nearest neighbor that satisfies |i-j| > min_sep)
        dist, idx = tree.query(y, k=2)  # k=2 because the point itself is distance 0
        if abs(i - idx[1]) > min_sep:
            neighbors.append(idx[1])
        else:
            # fallback: search for next valid neighbor
            ...
    return np.array(neighbors)
```
> **Source:** The `nolds` library implements this logic efficiently.

### 3.2 Track Divergence

For each pair \((Y(i), Y(j))\), track the Euclidean distance \(d_i(k)\) after \(k\) time steps:

\[
d_i(k) = \| Y(i+k) - Y(j+k) \|, \quad k = 0, 1, \dots, k_{\max}
\]

where \(k_{\max}\) is typically 5–10 cycles (or 50–100 samples for Tai Chi). Rosenstein’s original paper defines \(k_{\max}\) as a fraction of the total time series length.

```python
def track_divergence(Y, neighbor_idx, max_k):
    distances = np.zeros((len(Y), max_k))
    for i in range(len(Y) - max_k):
        j = neighbor_idx[i]
        for k in range(max_k):
            distances[i, k] = np.linalg.norm(Y[i+k] - Y[j+k])
    return distances
```

### 3.3 Compute Average Logarithmic Divergence

For each time step \(k\), compute the mean logarithm of all \(d_i(k)\):

\[
\langle \ln d(k) \rangle = \frac{1}{M} \sum_{i=1}^{M} \ln d_i(k)
\]

where \(M\) is the number of valid pairs at step \(k\).

### 3.4 Linear Fit to Obtain \(\lambda_{\max}\)

If the system is chaotic, the average divergence grows exponentially: \(\langle \ln d(k) \rangle \approx \ln C + \lambda_{\max} \cdot k \). Fit a straight line (e.g., using least squares) to the curve \(\langle \ln d(k) \rangle\) vs. \(k\). The **slope** of the line is the estimate of the largest Lyapunov exponent.

```python
from scipy.stats import linregress

def calc_lambda(mean_log_distances):
    k = np.arange(len(mean_log_distances))
    slope, intercept, r_value, p_value, std_err = linregress(k, mean_log_distances)
    return slope   # λ_max
```

> **Important:** Some implementations use the **short‑term** LyE by fitting only the initial linear region (e.g., up to one stride or one movement cycle). For Tai Chi, the initial 2–3 cycles may be most relevant for local stability.

---

## 4. Multi‑Sensor Integration: Univariate vs. Multivariate Approaches

When multiple IMUs are placed on different body segments, you have two principal strategies:

| **Approach** | **Description** | **Interpretation** | **Example** |
| :--- | :--- | :--- | :--- |
| **Univariate (per‑joint)** | Compute LyE separately for each IMU channel (e.g., shoulder, elbow, wrist angular velocity). Each sensor yields its own \(\lambda_{\max}\). | Shows stability of each individual joint/segment. | All joints are stable → low LyE; a wobbling shoulder → higher LyE. |
| **Multivariate (combined)** | Concatenate the delay vectors of all sensors to form a joint state space of dimension \(m \times \text{(number of sensors)}\). | Captures whole‑body coordination. Higher values indicate loss of inter‑segment coordination. | Used successfully to differentiate age groups during quiet standing. |

> **Implementation trick:** For the multivariate method, simply stack the embedded vectors from each sensor horizontally. For example, if each sensor gives a delay vector \(Y_{s}(i)\) of length \(m\), the combined vector is \([Y_{1}(i), Y_{2}(i), \dots, Y_{S}(i)]\).

---

## 5. Validation Using Reference Data

Before applying the algorithm to Tai Chi data, validate your implementation with a known chaotic system. The **Lorenz system** is the standard benchmark.

**Parameters:** \(\sigma = 10, \rho = 28, \beta = 8/3\)

Generate a time series (e.g., the \(x\) coordinate) using a numerical integrator (e.g., Runge‑Kutta 4) with a step size of 0.01. The known largest Lyapunov exponent for the Lorenz attractor is approximately **0.90**.

*   If your algorithm yields a value near 0.90, your implementation is correct.
*   If the value deviates significantly, check your time delay (\(\tau\)), embedding dimension (\(m\)), and temporal separation threshold.

---

## 6. Parameter Tuning and Sensitivity Analysis

Because there is no single “correct” set of parameters for Tai Chi data, you must perform a **sensitivity analysis**:

1.  **Vary \(\tau\)** around the AMI first minimum (e.g., \(\tau = T_{\text{AMI}} \pm 2\) samples) and observe changes in \(\lambda_{\max}\). A stable estimate will change little over a small range.
2.  **Vary \(m\)** from 3 to 10. The FNN method suggests the minimal \(m\) needed; use that value or one slightly higher.
3.  **Vary the temporal separation** \(T_{\text{sep}}\) from the mean period to twice the mean period.
4.  **Vary the number of cycles** (repetitions) included. If \(\lambda_{\max}\) continues to change significantly after 200 repetitions, longer recordings may be required.

> **Recommended values for Tai Chi (starting point):**
> *   Time delay \(\tau\): 10–20 samples (at 100 Hz)
> *   Embedding dimension \(m\): 5–7
> *   Temporal separation \(T_{\text{sep}}\): mean period (e.g., 50 samples for a 0.5 s movement repetition)
> *   Max tracking steps \(k_{\max}\): 50–100

---

## 7. Complete Python Workflow Example

```python
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import linregress
from scipy.fft import fft, fftfreq

def lyap_rosenstein(data, fs, tau=None, emb_dim=None, min_tsep=None, max_k=100):
    # 1. Estimate parameters if not provided
    if tau is None:
        tau = estimate_time_delay(data)            # AMI first minimum
    if emb_dim is None:
        emb_dim = estimate_embedding_dim(data, tau) # FNN
    if min_tsep is None:
        freq = fftfreq(len(data), 1/fs)
        power = np.abs(fft(data))
        dominant_freq = freq[np.argmax(power[1:])+1]
        mean_period = int(1/dominant_freq * fs)    # samples
        min_tsep = mean_period

    # 2. Phase space reconstruction
    N = len(data)
    M = N - (emb_dim-1)*tau
    Y = np.zeros((M, emb_dim))
    for i in range(M):
        Y[i] = data[i : i+emb_dim*tau : tau]

    # 3. Nearest neighbor search with temporal separation
    tree = KDTree(Y)
    neighbors = np.zeros(M, dtype=int)
    for i in range(M):
        # find the nearest neighbor with |i - j| > min_tsep
        dist, idx = tree.query(Y[i], k=2)  # k=2: itself + nearest other
        if abs(i - idx[1]) > min_tsep:
            neighbors[i] = idx[1]
        else:
            # fallback: search further neighbors until condition satisfied
            found = False
            for jj in range(2, min(20, M)):
                dist, idx = tree.query(Y[i], k=jj+1)
                if abs(i - idx[jj]) > min_tsep:
                    neighbors[i] = idx[jj]
                    found = True
                    break
            if not found:
                neighbors[i] = idx[1]  # accept risk
    # 4. Track divergence
    max_k = min(max_k, M - 1)
    log_distances = np.full((M, max_k), np.nan)
    for i in range(M - max_k):
        j = neighbors[i]
        for k in range(1, max_k+1):
            dist = np.linalg.norm(Y[i+k] - Y[j+k])
            log_distances[i, k-1] = np.log(dist)

    # 5. Average over all trajectories
    mean_log_dist = np.nanmean(log_distances, axis=0)

    # 6. Linear fit to obtain λ_max
    k_vals = np.arange(1, max_k+1)
    slope, intercept, r, p, se = linregress(k_vals, mean_log_dist)
    return slope   # λ_max
```

> **Note:** For production use, consider using well‑tested libraries like `nolds.lyap_r()` or `neurokit2.complexity_lyapunov()`, which implement the Rosenstein algorithm with robust parameter estimation.

---

## 8. Quality Control: Interpreting the Divergence Curve

After running the algorithm, always plot the **mean log divergence** curve. A valid chaotic system produces a curve that:

1.  **Rises linearly** in the initial segment (representing exponential divergence).
2.  **Saturates** after some time (limited by the finite size of the attractor).
3.  **Does not exhibit systematic oscillations** (which would indicate an insufficient temporal separation or a periodic signal).

If the curve never saturates, your time series may be too short. If the initial region is not linear, try adjusting \(\tau\) or \(m\).

---

## Reference Implementation Repositories

For a fully validated implementation, consult these resources:

| **Repository** | **Language** | **Features** |
| :--- | :--- | :--- |
| [SjoerdBruijn/LocalDynamicStability](https://github.com/SjoerdBruijn/LocalDynamicStability) | MATLAB | Used in many gait studies; includes time normalisation and automatic period estimation. |
| [`nolds` Python library](https://nolds.readthedocs.io/) | Python | Robust, with RANSAC fit option to reduce outlier influence. Default parameters are well‑tested for biomedical signals. |
| [`neurokit2`](https://neuropsychology.github.io/NeuroKit/) | Python | Includes a “Makowski” modification of Rosenstein with KDTree for faster computation. |
| [`nolitsa`](https://github.com/manu-mannattil/nolitsa) | Python | Pure Python implementation of many nonlinear time‑series tools. |

---

## 9. Interpretation for Tai Chi Movements

Once you have computed \(\lambda_{\max}\) for each sensor (or for the multivariate system), interpret the values in the context of Tai Chi performance:

*   **Lower LyE** → greater local stability, indicating that the practitioner can maintain consistent movement trajectories despite the cognitive and balance demands of Tai Chi.
*   **Higher LyE** → reduced stability, suggesting larger fluctuations between cycles (e.g., arm path variability) or impaired coordination between segments.

Use the **multivariate LyE** to assess whole‑body coordination, and **univariate LyE** of individual joints to pinpoint which body segments contribute most to overall instability.