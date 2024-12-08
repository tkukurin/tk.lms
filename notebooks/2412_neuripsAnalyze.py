"""NeurIPS papers.

Make sure to fetch them first via neuripsFetch.
"""
# %%
import tk
import pandas as pd

_nameit = lambda x: tk.datadir / f"2412_openreviewNotes{x}"
fname = _nameit('.json')
df = pd.read_json(fname)

# %%
df['topic'] = df['topic'].astype(pd.CategoricalDtype())
print(
    df['topic'].value_counts()[:5],
    df['topic'].value_counts()[-5:],
    sep='\n\n'
)

# topic
# machine_vision                 605
# natural_language_processing    313
# reinforcement_learning         288
# learning_theory                264
# diffusion_based_models         229
# Name: count, dtype: int64
# 
# topic
# speech_and_audio                        30
# infrastructure                          28
# active_learning                         25
# human-AI_interaction                    22
# machine_learning_for_social_sciences    21
# Name: count, dtype: int64

# %%
df['authors'].apply(lambda x: len(x)).hist(bins=10)
# %%
import itertools as it
from collections import Counter
auth2count = Counter(it.chain(*df['authors']))
# %%
auth2df = (
    df.explode('authors')
    .apply({'authors': lambda x: x.apply(str.title)})
    .groupby('authors')
)
print(f'{len(auth2df.groups)=}')

auth2count = sorted(
    [(len(v), k) for k, v in auth2df.groups.items()])
print(
    auth2count[:5],
    auth2count[-5:],
    sep='\n\n'
)
# %%
def clean(xs):
    replace = lambda text: (
    ''.join(c for c in text if ord(c) > 31 or c in '\t\n\r')
    if isinstance(text, str) else text
    )
    return xs.apply(replace)

# for viewing in gsheets, need to remove some tab chars tho
# df.apply(clean, axis=1).to_excel(_nameit(".xlsx"))
# %%

gemini_flash_answer = """
The provided text covers a wide range of topics in machine learning, focusing on several novel themes.  Here's a grouping of related IDs based on common themes:


**I. Large Language Models (LLMs) and Alignment:**

* **Alignment and Safety:** 30, 38, 46, 60, 101, 176, 178, 206, 227, 230, 246, 262, 270, 297, 300, 304, 309, 310, 342, 384, 388, 391, 442, 461, 466, 482, 662, 842, 911, 913, 927, 965, 991, 1001, 1046, 1058, 1110, 1176, 1290, 1313, 1315, 1346, 1388, 1391, 1457, 1508, 1516, 1547, 1557, 1628, 1666, 1676, 1766, 1778, 1880, 1891, 1931, 1966, 2005, 2027, 2041, 2060, 2070, 2108, 2121, 2124, 2154, 2177, 2187, 2207, 2265, 2311, 2333, 2368, 2420, 2430, 2442, 2456, 2477, 2487, 2524, 2566, 2574, 2641, 2666, 2679, 2704, 2708, 2713, 2766, 2799, 2800, 2815, 2829, 2842, 2868, 2891, 2966, 2980, 3001, 3006, 3058, 3071, 3087, 3119, 3137, 3188, 3207, 3261, 3278, 3287, 3304, 3347, 3378, 3388, 3401, 3430, 3447, 3461, 3554, 3608, 3641, 3701, 3727, 3782, 3818, 3827, 3847, 3878, 3911, 3947, 3967, 3977, 4006, 4015, 4018, 4041, 4074, 4137, 4167, 4178, 4205, 4207, 4209
* **LLM Capabilities and Evaluation:** 70, 144, 166, 187, 207, 209, 228, 263, 285, 314, 339, 445, 483, 497, 556, 636, 642, 660, 738, 754, 807, 817, 846, 860, 870, 878, 887, 913, 920, 970, 979, 983, 994, 1006, 1016, 1031, 1054, 1064, 1087, 1102, 1117, 1123, 1144, 1153, 1170, 1180, 1209, 1219, 1230, 1244, 1264, 1277, 1307, 1335, 1345, 1357, 1360, 1373, 1387, 1404, 1407, 1428, 1445, 1462, 1477, 1488, 1497, 1500, 1537, 1547, 1552, 1555, 1567, 1577, 1583, 1602, 1611, 1633, 1660, 1662, 1688, 1692, 1702, 1714, 1728, 1731, 1735, 1743, 1746, 1779, 1796, 1800, 1802, 1810, 1826, 1852, 1870, 1876, 1884, 1906, 1920, 1921, 1934, 1943, 1945, 1957, 1961, 1978, 1980, 1992, 2006, 2020, 2034, 2040, 2041, 2052, 2055, 2058, 2075, 2083, 2111, 2127, 2135, 2152, 2154, 2163, 2181, 2182, 2184, 2202, 2207, 2225, 2238, 2241, 2258, 2261, 2263, 2267, 2271, 2277, 2293, 2307, 2314, 2317, 2322, 2327, 2331, 2334, 2345, 2351, 2352, 2354, 2368, 2372, 2382, 2386, 2391, 2395, 2397, 2401, 2411, 2416, 2430, 2432, 2440, 2442, 2450, 2457, 2471, 2476, 2488, 2491, 2497, 2500, 2506, 2509, 2512, 2517, 2520, 2527, 2529, 2530, 2532, 2547, 2554, 2559, 2560, 2562, 2565, 2566, 2569, 2571, 2574, 2577, 2580, 2584, 2591, 2595, 2602, 2604, 2614, 2620, 2623, 2637, 2658, 2662, 2665, 2675, 2680, 2691, 2696, 2701, 2704, 2714, 2721, 2725, 2734, 2740, 2742, 2744, 2749, 2754, 2757, 2761, 2766, 2770, 2772, 2773, 2788, 2796, 2801, 2805, 2807, 2813, 2820, 2823, 2827, 2831, 2835, 2845, 2854, 2856, 2864, 2868, 2870, 2872, 2874, 2877, 2880, 2886, 2891, 2901, 2903, 2907, 2911, 2912, 2915, 2925, 2927, 2931, 2934, 2940, 2942, 2954, 2962, 2965, 2976, 2980, 2983, 2992, 2997, 3001, 3015, 3020, 3034, 3042, 3058, 3067, 3071, 3081, 3086, 3095, 3101, 3115, 3131, 3133, 3146, 3151, 3154, 3166, 3174, 3178, 3186, 3191, 3200, 3202, 3215, 3218, 3225, 3237, 3240, 3247, 3254, 3265, 3271, 3272, 3276, 3280, 3287, 3291, 3296, 3304, 3311, 3316, 3328, 3336, 3346, 3353, 3361, 3364, 3372, 3380, 3381, 3388, 3401, 3411, 3415, 3435, 3440, 3444, 3447, 3455, 3461, 3472, 3475, 3480, 3481, 3489, 3506, 3509, 3517, 3520, 3527, 3533, 3549, 3561, 3563, 3566, 3568, 3583, 3601, 3604, 3611, 3615, 3631, 3637, 3647, 3658, 3662, 3672, 3683, 3691, 3700, 3703, 3707, 3715, 3720, 3722, 3734, 3742, 3745, 3753, 3756, 3770, 3787, 3790, 3797, 3800, 3807, 3812, 3818, 3827, 3833, 3837, 3845, 3847, 3861, 3864, 3870, 3873, 3878, 3880, 3882, 3884, 3891, 3901, 3907, 3911, 3918, 3929, 3931, 3934, 3947, 3951, 3957, 3966, 3972, 3976, 3987, 3997, 4001, 4005, 4009, 4017, 4020, 4024, 4029, 4047, 4049, 4059, 4073, 4080, 4098, 4105, 4108, 4121, 4128, 4131, 4137, 4145, 4148, 4150, 4154, 4161, 4169, 4173, 4179, 4182, 4188, 4203, 4211, 4218, 4221, 4223, 4227, 4230, 4236, 4240, 4247, 4249, 4250, 4251, 4254, 4262, 4267, 4271, 4272, 4276, 4280, 4285, 4290, 4292


**II.  Reinforcement Learning (RL):**

* **Offline RL:** 142, 200, 355, 439, 464, 537, 571, 671, 681, 712, 794, 855, 905, 964, 995, 1008, 1053, 1062, 1142, 1143, 1214, 1253, 1355, 1388, 1416, 1432, 1449, 1464, 1488, 1690, 1744, 1817, 1847, 1861, 1906, 1933, 1953, 1964, 1998, 2002, 2053, 2055, 2078, 2136, 2155, 2177, 2182, 2200, 2224, 2227, 2233, 2274, 2284, 2307, 2320, 2333, 2337, 2353, 2357, 2376, 2383, 2406, 2411, 2414, 2424, 2444, 2455, 2471, 2484, 2506, 2507, 2537, 2553, 2555, 2568, 2584, 2607, 2621, 2630, 2653, 2661, 2662, 2666, 2670, 2681, 2686, 2712, 2730, 2736, 2744, 2757, 2765, 2777, 2797, 2806, 2827, 2836, 2848, 2857, 2861, 2866, 2874, 2883, 2888, 2906, 2909, 2912, 2930, 2944, 2961, 2970, 2973, 2990, 2993, 3007, 3027, 3045, 3057, 3062, 3070, 3088, 3092, 3111, 3136, 3139, 3157, 3162, 3177, 3182, 3193, 3201, 3205, 3222, 3227, 3233, 3239, 3245, 3253, 3257, 3260, 3270, 3288, 3300, 3307, 3313, 3321, 3333, 3343, 3355, 3358, 3366, 3376, 3382, 3395, 3405, 3413, 3427, 3433, 3438, 3455, 3465, 3474, 3488, 3490, 3510, 3512, 3516, 3523, 3533, 3539, 3553, 3564, 3571, 3588, 3600, 3606, 3613, 3621, 3636, 3645, 3650, 3662, 3666, 3681, 3688, 3706, 3709, 3723, 3730, 3736, 3739, 3744, 3755, 3761, 3766, 3771, 3777, 3784, 3786, 3792, 3806, 3816, 3824, 3833, 3848, 3858, 3861, 3866, 3871, 3878, 3883, 3888, 3906, 3913, 3917, 3923, 3933, 3935, 3939, 3948, 3958, 3966, 3976, 3983, 4007, 4010, 4016, 4024, 4039, 4045, 4055, 4065, 4070, 4077, 4093, 4106, 4112, 4120, 4123, 4125, 4138, 4146, 4151, 4153, 4162, 4172, 4177, 4186, 4193, 4198, 4201, 4206, 4212, 4216, 4219, 4224, 4229, 4233, 4245, 4248, 4255, 4261, 4266, 4271, 4278, 4283, 4286, 4295


**III.  Computer Vision (CV):**

* **Image Generation and Editing:** 22, 75, 115, 131, 156, 167, 211, 231, 251, 275, 303, 312, 333, 345, 356, 365, 382, 404, 473, 496, 517, 579, 591, 610, 667, 732, 757, 781, 787, 790, 803, 806, 821, 851, 862, 884, 912, 914, 932, 960, 977, 1000, 1004, 1021, 1045, 1065, 1075, 1091, 1129, 1145, 1150, 1165, 1167, 1191, 1204, 1221, 1257, 1275, 1291, 1309, 1332, 1346, 1365, 1371, 1382, 1394, 1405, 1444, 1456, 1467, 1491, 1507, 1521, 1526, 1530, 1541, 1556, 1567, 1577, 1598, 1612, 1618, 1627, 1639, 1644, 1667, 1672, 1684, 1694, 1703, 1707, 1729, 1757, 1766, 1770, 1775, 1781, 1786, 1812, 1830, 1832, 1837, 1851, 1863, 1867, 1871, 1889, 1892, 1922, 1924, 1930, 1944, 1962, 1970, 1990, 1998, 2000, 2003, 2006, 2011, 2016, 2017, 2029, 2032, 2038, 2045, 2050, 2057, 2078, 2082, 2084, 2089, 2092, 2095, 2105, 2115, 2117, 2127, 2129, 2138, 2140, 2145, 2146, 2151, 2159, 2165, 2167, 2172, 2180, 2186, 2191, 2194, 2200, 2206, 2217, 2226, 2231, 2235, 2239, 2246, 2247, 2251, 2257, 2264, 2267, 2275, 2281, 2291, 2294, 2300, 2305, 2306, 2308, 2316, 2320, 2322, 2329, 2334, 2341, 2344, 2354, 2359, 2362, 2366, 2371, 2377, 2380, 2382, 2388, 2391, 2394, 2399, 2402, 2405, 2408, 2414, 2418, 2425, 2431, 2440, 2444, 2448, 2458, 2462, 2471, 2473, 2489, 2492, 2501, 2507, 2525, 2534, 2545, 2550, 2556, 2572, 2576, 2582, 2588, 2594, 2610, 2619, 2625, 2627, 2632, 2635, 2644, 2654, 2659, 2663, 2668, 2672, 2682, 2688, 2702, 2709, 2711, 2718, 2723, 2729, 2736, 2740, 2743, 2746, 2751, 2759, 2766, 2779, 2780, 2790, 2795, 2803, 2814, 2819, 2821, 2829, 2831, 2848, 2850, 2857, 2859, 2860, 2862, 2871, 2881, 2892, 2897, 2901, 2904, 2908, 2917, 2919, 2921, 2924, 2930, 2932, 2946, 2948, 2952, 2955, 2959, 2968, 2970, 2977, 2981, 2996, 3005, 3011, 3016, 3020, 3023, 3032, 3035, 3037, 3039, 3043, 3046, 3050, 3052, 3055, 3060, 3068, 3071, 3078, 3082, 3090, 3091, 3100, 3103, 3113, 3122, 3132, 3142, 3153, 3158, 3171, 3186, 3189, 3190, 3204, 3207, 3210, 3222, 3229, 3235, 3240, 3243, 3248, 3250, 3252, 3257, 3264, 3270, 3273, 3279, 3289, 3290, 3299, 3300, 3309, 3316, 3317, 3321, 3326, 3330, 3333, 3335, 3340, 3344, 3352, 3355, 3362, 3369, 3373, 3380, 3382, 3384, 3389, 3395, 3397, 3399, 3402, 3405, 3409, 3414, 3420, 3421, 3424, 3425, 3427, 3431, 3434, 3437, 3440, 3444, 3446, 3449, 3454, 3456, 3458, 3460, 3462, 3467, 3469, 3470, 3473, 3476, 3479, 3480, 3482, 3484, 3486, 3490, 3493, 3496, 3501, 3503, 3507, 3513, 3515, 3521, 3524, 3528, 3534, 3536, 3542, 3544, 3548, 3550, 3552, 3556, 3560, 3564, 3568, 3570, 3579, 3586, 3591, 3594, 3605, 3606, 3610, 3616, 3620, 3626, 3628, 3632, 3635, 3639, 3644, 3648, 3651, 3654, 3658, 3663, 3665, 3667, 3673, 3675, 3680, 3682, 3690, 3692, 3704, 3707, 3710, 3713, 3716, 3720, 3722, 3725, 3728, 3733, 3735, 3738, 3741, 3743, 3749, 3751, 3757, 3760, 3762, 3764, 3769, 3770, 3774, 3779, 3780, 3783, 3785, 3787, 3791, 3797, 3801, 3804, 3809, 3813, 3816, 3821, 3823, 3826, 3831, 3836, 3840, 3844, 3846, 3850, 3853, 3856, 3860, 3862, 3866, 3870, 3872, 3875, 3876, 387
"""

# %%
import re

def parse(line: str):
    # line = "* **words**: 1, 2, 3"

    pattern = r"\*\s+\*\*(.*?)\*\*:?\s+([\d,\s]+)"
    if match := re.match(pattern, line):
        key = match.group(1)
        values = list(map(int, match.group(2).split(',')))
        result = (key.strip(': '), values)
        return result

kvs = []
for line in gemini_flash_answer.split("\n"):
    if kv := parse(line):
        kvs.append(kv)
kvs = dict(kvs)
print(len(kvs))
# %%
import itertools as it
overlaps = {}
for (k1, v1), (k2, v2) in it.combinations(kvs.items(), 2):
    diff = set(v1) & set(v2)
    if diff:
        overlaps[(k1, k2)] = diff
print({k:len(v) for k, v in overlaps.items()})

# %%
claude_sonnet_35 = """
## Key Research Areas

**Machine Learning & Deep Learning**
- Novel architectures like TAS-GNN improve graph classification performance by up to 27.20% through optimized neuron utilization
- S3, a modular neural network layer, enhances time-series representation learning by rearranging segments
- PGN introduced as a successor to RNN for long-range time series forecasting

**Language Models & Alignment**
- MAmmoTH2 harvests 10M high-quality instruction data from web corpus for fine-tuning language models without costly human annotation
- FLAME proposes factuality-aware alignment while maintaining instruction-following capability
- DropBP accelerates LLM fine-tuning by selectively dropping backward propagation based on layer sensitivity[191]

**Computer Vision & Graphics**
- G2D framework learns global and dense visual features from image-text pairs, achieving strong performance with just 1% of training data
- Neural Gaffer enables relighting any object image under any environmental lighting using diffusion models
- MeshFormer delivers high-quality meshes from sparse-view reconstruction[186]

**Reinforcement Learning**
- REBEL simplifies policy optimization by regressing relative rewards with theoretical guarantees
- Dynamic Model Predictive Shielding enables safe RL with goal progress while recovering from unsafe situations[193]
- Focus On What Matters (SMG) improves RL generalization through task-relevant representation extraction[196]

**AI Safety & Robustness**
- Password-locked models are used to study capability elicitation through supervised fine-tuning and RL
- Improved few-shot jailbreaking techniques can circumvent aligned language models and their defenses
- Pure Tune, Safe Test principle introduced to maintain model alignment after fine-tuning[176]

## Theoretical Advances

**Mathematical Foundations**
- First global convergence proof for gradient EM on over-parameterized Gaussian mixture models
- New generalization bounds for DNNs learning composition of functions
- Novel theoretical framework for f-divergence-based domain learning[169]

**Optimization & Learning Theory**
- Simple universal approach achieves optimal gradient-variation regret guarantees[109]
- Quantum speedups demonstrated for finding Goldstein stationary points[202]
- Minimax optimal bounds established for corruption-robust linear bandits[203]
"""

# %%
# %%
