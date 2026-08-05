"""
卫星信道 CSI 数据集生成器

模式:
  - 单星 (默认): 每样本 1 颗卫星 + 1 个地面终端
  - 多星: 每样本 2-4 颗共视卫星 (Walker Delta 星座)

策略: 基于轨道力学 + 地面终端自适应可见性搜索 (多星模式)
"""

import numpy as np
import pandas as pd
import os
import argparse

from config import (
    CARRIER_FREQ, LIGHT_SPEED, SATELLITE_HEIGHT,
    NUM_ANTENNAS, NUM_PATHS, DT, K_FACTOR,
    HISTORY_STEPS, FUTURE_STEPS, RESIDUAL_VEL_FACTOR,
    NUM_PLANES, SATS_PER_PLANE, INCLINATION, MIN_ELEVATION_VIS,
    MAX_SATELLITES_PER_SAMPLE, EARTH_RADIUS, EARTH_MU,
    TERMINAL_LAT_MIN, TERMINAL_LAT_MAX, TERMINAL_LON_MIN, TERMINAL_LON_MAX,
)

# ULA 天线间距 (半波长)
ANTENNA_SPACING = 0.5  # d/λ = 0.5


def _deg2rad(d):
    return d * np.pi / 180


def compute_satellite_positions_in_eci(t0_seconds):
    """
    计算 Walker Delta 星座所有卫星在 t0 时刻的 ECI 坐标。
    圆轨道 (e=0)，平均运动 n = sqrt(mu / (R+h)^3)
    """
    R_orbit = EARTH_RADIUS + SATELLITE_HEIGHT
    n = np.sqrt(EARTH_MU / R_orbit ** 3)

    f = 1  # Walker phasing factor
    N = NUM_PLANES
    S = SATS_PER_PLANE

    sats = []
    for p in range(N):
        for s in range(S):
            u0 = 2 * np.pi * s / S + 2 * np.pi * f * s / (N * S)
            u = u0 + n * t0_seconds
            x_orb = R_orbit * np.cos(u)
            y_orb = R_orbit * np.sin(u)
            x_eci = x_orb
            y_eci = y_orb * np.cos(INCLINATION)
            z_eci = y_orb * np.sin(INCLINATION)
            sats.append({
                "sat_id": p * S + s,
                "plane": p,
                "slot": s,
                "eci_pos": np.array([x_eci, y_eci, z_eci]),
            })
    return sats


def geodetic_to_ecef(lat_deg, lon_deg, alt_m=0):
    """大地坐标 → ECEF (球体近似)。"""
    lat = _deg2rad(lat_deg)
    lon = _deg2rad(lon_deg)
    R = EARTH_RADIUS + alt_m
    return np.array([
        R * np.cos(lat) * np.cos(lon),
        R * np.cos(lat) * np.sin(lon),
        R * np.sin(lat),
    ])


def compute_elevation_azimuth(sat_eci, ground_ecef):
    """计算地面终端看卫星的仰角、方位角、距离。"""
    vec = sat_eci - ground_ecef
    dist = np.linalg.norm(vec)
    if dist < 1e-10:
        return np.pi / 2, 0.0, dist
    ground_norm = ground_ecef / np.linalg.norm(ground_ecef)
    cos_angle = np.clip(np.dot(vec, ground_norm) / dist, -1, 1)
    elevation = np.arcsin(cos_angle)
    # 方位角
    north = np.array([0, 0, 1]) - ground_norm * ground_norm[2]
    north_norm = np.linalg.norm(north)
    north = north / north_norm if north_norm > 1e-10 else np.array([1, 0, 0])
    east = np.cross(ground_norm, north)
    vec_horiz = vec - ground_norm * np.dot(vec, ground_norm)
    azimuth = np.arctan2(np.dot(vec_horiz, east), np.dot(vec_horiz, north))
    return elevation, azimuth, dist


def find_terminal_with_visibility(sats, min_elevation, min_sats=2, max_attempts=50):
    """
    搜索一个能看到至少 min_sats 颗卫星的地面终端位置。
    策略：随机采样纬度 (覆盖带内)，对每个纬度搜索最优经度。
    """
    for _ in range(max_attempts):
        lat = np.random.uniform(TERMINAL_LAT_MIN, TERMINAL_LAT_MAX)
        lon = np.random.uniform(TERMINAL_LON_MIN, TERMINAL_LON_MAX)
        ground = geodetic_to_ecef(lat, lon)
        visible = []
        for sat in sats:
            elev, az, dist = compute_elevation_azimuth(sat["eci_pos"], ground)
            if elev >= min_elevation:
                visible.append({**sat, "elevation": elev, "azimuth": az, "distance": dist})
        if len(visible) >= min_sats:
            visible.sort(key=lambda s: s["elevation"], reverse=True)
            return visible
    # 如果实在找不到，返回仰角最高的几颗 (降低阈值)
    all_with_el = []
    lat = np.random.uniform(TERMINAL_LAT_MIN, TERMINAL_LAT_MAX)
    lon = np.random.uniform(TERMINAL_LON_MIN, TERMINAL_LON_MAX)
    ground = geodetic_to_ecef(lat, lon)
    for sat in sats:
        elev, az, dist = compute_elevation_azimuth(sat["eci_pos"], ground)
        all_with_el.append({**sat, "elevation": elev, "azimuth": az, "distance": dist})
    all_with_el.sort(key=lambda s: s["elevation"], reverse=True)
    return all_with_el[:max(2, min_sats)]


def generate_single_satellite_csi(sat_geo, snr_db, v_dev_kmh, rain_att_db,
                                  motion_azimuth, history_steps=None, future_steps=None):
    """为单颗卫星生成 CSI 序列。"""
    if history_steps is None:
        history_steps = HISTORY_STEPS
    if future_steps is None:
        future_steps = FUTURE_STEPS

    v_dev = v_dev_kmh / 3.6
    total_steps = history_steps + future_steps
    base_elevation = sat_geo["elevation"]

    los_aoa = sat_geo["azimuth"]
    sat_azimuth = sat_geo["azimuth"]
    direction_cosine = np.cos(motion_azimuth - sat_azimuth)
    elevation_cosine = np.cos(base_elevation)

    # 多普勒 (残差)
    doppler_full = (CARRIER_FREQ / LIGHT_SPEED) * v_dev * direction_cosine * elevation_cosine
    residual_doppler = doppler_full * RESIDUAL_VEL_FACTOR

    # NLOS 多普勒
    path_aoa_for_doppler = np.random.uniform(-np.pi / 6, np.pi / 6, NUM_PATHS)
    path_dopplers_full = (
        (CARRIER_FREQ / LIGHT_SPEED)
        * v_dev * np.cos(path_aoa_for_doppler)
        * elevation_cosine * np.random.uniform(0.5, 1.0, NUM_PATHS)
    )
    path_dopplers = path_dopplers_full * RESIDUAL_VEL_FACTOR

    # 大尺度增益
    distance_km = sat_geo["distance"] / 1000
    fspl_db = 20 * np.log10(distance_km) + 20 * np.log10(CARRIER_FREQ / 1e9) + 92.45
    total_loss_db = fspl_db + rain_att_db
    large_scale_gain = 10 ** (-total_loss_db / 20)

    # 多径
    path_delays = np.sort(np.random.uniform(0, 30e-9, NUM_PATHS))
    path_phases = np.random.uniform(0, 2 * np.pi, NUM_PATHS)
    path_gains = np.exp(-path_delays / (30e-9 / 3))
    path_aoa = path_aoa_for_doppler

    k_linear = 10 ** (K_FACTOR / 10)
    los_scale = np.sqrt(k_linear / (k_linear + 1))
    nlos_scale = np.sqrt(1 / (k_linear + 1))

    data_rows = []
    for t_step in range(total_steps):
        los_spatial = np.exp(
            1j * 2 * np.pi * ANTENNA_SPACING * np.sin(los_aoa) * np.arange(NUM_ANTENNAS)
        )
        phase = 2 * np.pi * residual_doppler * (t_step * DT)
        los_comp = np.exp(1j * phase) * los_spatial

        nlos_comp = np.zeros(NUM_ANTENNAS, dtype=complex)
        for p in range(NUM_PATHS):
            path_phase = 2 * np.pi * path_dopplers[p] * t_step * DT + path_phases[p]
            path_spatial = np.exp(
                1j * 2 * np.pi * ANTENNA_SPACING * np.sin(path_aoa[p]) * np.arange(NUM_ANTENNAS)
            )
            nlos_comp += path_gains[p] * np.exp(1j * path_phase) * path_spatial
        nlos_comp /= NUM_PATHS

        h_vec = large_scale_gain * (los_scale * los_comp + nlos_scale * nlos_comp)

        if snr_db > -np.inf:
            signal_power = np.mean(np.abs(h_vec) ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(NUM_ANTENNAS) + 1j * np.random.randn(NUM_ANTENNAS)
            )
            h_vec_noisy = h_vec + noise
        else:
            h_vec_noisy = h_vec

        rain_att_db_t = rain_att_db * (1 + 0.1 * np.sin(2 * np.pi * t_step / total_steps))

        row = {
            "Sample_ID": sat_geo["sample_id"],
            "Sat_Index": sat_geo["sat_index"],
            "Sat_Plane": sat_geo["plane"],
            "Sat_Slot": sat_geo["slot"],
            "Sat_ID": sat_geo["sat_id"],
            "Time_Step": t_step,
            "V_Device_kmh": v_dev_kmh,
            "SNR_dB": snr_db,
            "Distance_km": distance_km,
            "Doppler_Hz": residual_doppler,
            "Rain_Att_dB": rain_att_db_t,
            "cos_el": elevation_cosine,
            "dir_cos": direction_cosine,
            "Sat_Azimuth": sat_geo["azimuth"],
            "Sat_Elevation": base_elevation,
            "Is_Future": 1 if t_step >= history_steps else 0,
        }
        for i in range(NUM_ANTENNAS):
            row[f"H_Real_{i}"] = h_vec_noisy[i].real
            row[f"H_Imag_{i}"] = h_vec_noisy[i].imag
        data_rows.append(row)

    return data_rows


def find_terminal_random(sats):
    """单星模式：随机选 1 个地面终端 + 仰角最高的 1 颗卫星。"""
    lat = np.random.uniform(TERMINAL_LAT_MIN, TERMINAL_LAT_MAX)
    lon = np.random.uniform(TERMINAL_LON_MIN, TERMINAL_LON_MAX)
    ground = geodetic_to_ecef(lat, lon)
    all_with_el = []
    for sat in sats:
        elev, az, dist = compute_elevation_azimuth(sat["eci_pos"], ground)
        all_with_el.append({**sat, "elevation": elev, "azimuth": az, "distance": dist})
    all_with_el.sort(key=lambda s: s["elevation"], reverse=True)
    return all_with_el[:1]  # 只取仰角最高的 1 颗


def generate_single_sat_sample(sample_id, speed, snr_db, t0_seconds):
    """
    单星样本：1 个随机地面终端 + 仰角最高 1 颗卫星。
    返回 DataFrame 行列表 (无 Sat_Index 等多星列)。
    """
    all_sats = compute_satellite_positions_in_eci(t0_seconds)
    selected = find_terminal_random(all_sats)

    if len(selected) == 0:
        return []

    sat = selected[0]
    sat_geo = {**sat, "sample_id": sample_id, "sat_index": 0}

    rain_att_db_base = 0.5 * np.random.exponential(2)
    motion_azimuth = np.random.uniform(0, 2 * np.pi)

    return generate_single_satellite_csi(
        sat_geo, snr_db, speed, rain_att_db_base, motion_azimuth
    )


def generate_multi_sat_sample(sample_id, speed, snr_db, t0_seconds):
    """
    生成一个多样本：所有共视卫星的 CSI。
    1. 计算星座位置
    2. 搜索能看到足够卫星的终端位置
    3. 取最多 MAX_SATELLITES_PER_SAMPLE 颗 (按仰角排序)
    4. 生成每颗卫星的 CSI
    """
    all_sats = compute_satellite_positions_in_eci(t0_seconds)
    visible = find_terminal_with_visibility(all_sats, MIN_ELEVATION_VIS, min_sats=2)

    if len(visible) == 0:
        return []

    selected = visible[:MAX_SATELLITES_PER_SAMPLE]

    rain_att_db_base = 0.5 * np.random.exponential(2)
    motion_azimuth = np.random.uniform(0, 2 * np.pi)

    data_rows = []
    for idx, sat in enumerate(selected):
        sat_geo = {**sat, "sample_id": sample_id, "sat_index": idx}
        rows = generate_single_satellite_csi(
            sat_geo, snr_db, speed, rain_att_db_base, motion_azimuth
        )
        data_rows.extend(rows)

    return data_rows


def generate_exp1_speed(speeds=None, output_dir="./dataset", n_samples=10000, multi_sat=False):
    """实验一: 不同终端速度下的 NMSE"""
    if speeds is None:
        speeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    mode_str = "多星" if multi_sat else "单星"
    gen_fn = generate_multi_sat_sample if multi_sat else generate_single_sat_sample

    os.makedirs(output_dir, exist_ok=True)
    train_snr = 20
    n_train = int(0.8 * n_samples)

    for speed in speeds:
        print(f"生成{mode_str}速度 {speed} km/h 的数据 (SNR={train_snr} dB)...")
        samples = []
        skipped = 0
        for s in range(n_samples):
            t0 = np.random.uniform(0, 3600)
            rows = gen_fn(s, speed, train_snr, t0)
            if len(rows) == 0:
                skipped += 1
                continue
            samples.extend(rows)
        if skipped > 0:
            print(f"  跳过 {skipped} 个无效样本")

        if len(samples) == 0:
            print(f"  警告: 速度 {speed} 无有效样本")
            continue

        df = pd.DataFrame(samples)
        unique_ids = df.Sample_ID.unique()
        train_ids = unique_ids[:n_train]
        df_train = df[df.Sample_ID.isin(train_ids)]
        df_test = df[~df.Sample_ID.isin(train_ids)]

        df_train.to_csv(f"{output_dir}/Exp1_Speed_{speed}_SNR{train_snr}_train.csv", index=False)
        df_test.to_csv(f"{output_dir}/Exp1_Speed_{speed}_SNR{train_snr}_test.csv", index=False)
        print(f"  样本数: train={len(train_ids)}, test={len(unique_ids) - n_train}, "
              f"总行数={len(df)}")

    print("实验一多星数据集生成完成！")


def generate_exp2_snr(snrs=None, output_dir="./dataset", n_samples=10000, multi_sat=False):
    """实验二: 不同 SNR 下的 NMSE"""
    if snrs is None:
        snrs = [0, 5, 10, 15, 20, 25, 30]

    mode_str = "多星" if multi_sat else "单星"
    gen_fn = generate_multi_sat_sample if multi_sat else generate_single_sat_sample

    os.makedirs(output_dir, exist_ok=True)
    n_train = int(0.8 * n_samples)

    for snr in snrs:
        print(f"生成{mode_str} SNR {snr} dB 的数据 (速度=100 km/h)...")
        samples = []
        skipped = 0
        for s in range(n_samples):
            t0 = np.random.uniform(0, 3600)
            rows = gen_fn(s, 100, snr, t0)
            if len(rows) == 0:
                skipped += 1
                continue
            samples.extend(rows)
        if skipped > 0:
            print(f"  跳过 {skipped} 个无效样本")

        if len(samples) == 0:
            print(f"  警告: SNR {snr} 无有效样本")
            continue

        df = pd.DataFrame(samples)
        unique_ids = df.Sample_ID.unique()
        train_ids = unique_ids[:n_train]
        df_train = df[df.Sample_ID.isin(train_ids)]
        df_test = df[~df.Sample_ID.isin(train_ids)]

        df_train.to_csv(f"{output_dir}/Exp2_Speed100_SNR_{snr}_train.csv", index=False)
        df_test.to_csv(f"{output_dir}/Exp2_Speed100_SNR_{snr}_test.csv", index=False)
        print(f"  样本数: train={len(train_ids)}, test={len(unique_ids) - n_train}, "
              f"总行数={len(df)}")

    print("实验二多星数据集生成完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="星地链路 CSI 数据集生成器")
    parser.add_argument(
        "--multi-sat", action="store_true",
        help="启用多星并发模式 (默认单星)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=10000,
        help="每个场景的样本数 (默认 10000)"
    )
    args = parser.parse_args()

    mode = "多星" if args.multi_sat else "单星"
    print(f"{'='*50}")
    print(f"数据集生成模式: {mode}")
    print(f"样本数: {args.n_samples}")
    print(f"{'='*50}")

    generate_exp1_speed(multi_sat=args.multi_sat, n_samples=args.n_samples)
    generate_exp2_snr(multi_sat=args.multi_sat, n_samples=args.n_samples)
