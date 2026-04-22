#
# Script used for first validation of OTSun
#

import sys
import os
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

FREECAD_BIN = r"D:\software\FreeCAD 1.1\bin"
sys.path.append(FREECAD_BIN)

try:
    import FreeCAD
    print("FreeCAD 导入成功！版本:", FreeCAD.Version())
except Exception as e:
    print("FreeCAD 导入失败，错误:", str(e))

import otsun
import numpy as np
from multiprocessing import Pool


# ---
# Single parallelizable computation (fixed angles)
# 注意：sel、light_spectrum 等不能跨进程传递 FreeCAD 对象
# 改为每个子进程独立加载 FreeCAD 文件
# ---
def single_computation(args):
    (ph, th,
     freecad_file,
     data_file_spectrum,
     aperture_collector_Th,
     number_of_rays,
     CSR) = args

    # 子进程中独立导入并加载 FreeCAD 文件
    import sys
    sys.path.append(r"D:\software\FreeCAD 1.1\bin")
    import FreeCAD
    import otsun
    import os

    FreeCAD.openDocument(freecad_file)
    doc = FreeCAD.ActiveDocument
    sel = doc.Objects

    # 定义材料（每个子进程都需要定义）
    otsun.ReflectorSpecularLayer("Mir1", 0.95)
    otsun.ReflectorSpecularLayer("Mir2", 0.91)
    otsun.AbsorberSimpleLayer("Abs", 0.95)
    otsun.TransparentSimpleLayer("Trans", 0.965)

    direction_distribution = otsun.buie_distribution(CSR)
    light_spectrum = otsun.cdf_from_pdf_file(data_file_spectrum)

    main_direction = otsun.polar_to_cartesian(ph, th) * -1.0
    current_scene = otsun.Scene(sel)
    tracking = otsun.MultiTracking(main_direction, current_scene)
    tracking.make_movements()
    emitting_region = otsun.SunWindow(current_scene, main_direction)
    l_s = otsun.LightSource(current_scene, emitting_region, light_spectrum, 1.0, direction_distribution)
    exp = otsun.Experiment(current_scene, l_s, number_of_rays, None)
    exp.run()
    tracking.undo_movements()

    print(f"Computed ph={ph}, th={th}")
    efficiency_from_source_Th = (exp.captured_energy_Th / aperture_collector_Th) / (
        exp.number_of_rays / exp.light_source.emitting_region.aperture)
    return ph, th, efficiency_from_source_Th


# ---
# Full parallel computation
# ---
def full_computation(ph, freecad_file, data_file_spectrum,
                     aperture_collector_Th, number_of_rays, CSR,
                     power_emitted_by_m2):
    theta_ini = 0.0
    theta_end = 90.0 + 1.E-4
    theta_step = 5.0

    args_list = [
        (ph, th, freecad_file, data_file_spectrum,
         aperture_collector_Th, number_of_rays, CSR)
        for th in np.arange(theta_ini, theta_end, theta_step)
    ]

    with Pool() as pool:
        values = pool.map(single_computation, args_list)

    # ---
    # Save results in file
    # ---
    output_folder = 'output1'
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f'efficiency_results_{int(ph)}.txt')

    with open(output_file, 'w') as f:
        f.write(f"{aperture_collector_Th * 1E-6} # Collector Th aperture in m2\n")
        f.write(f"{power_emitted_by_m2} # Source power emitted by m2\n")
        f.write(f"{number_of_rays} # Rays emitted per sun position\n")
        f.write("# phi theta efficiency_from_source_Th\n")
        for value in values:
            (ph_v, th_v, eff) = value
            f.write(f"{ph_v:.3f} {th_v:.3f} {eff:.6f}\n")

    print(f"结果已保存到: {output_file}")


if __name__ == '__main__':
    # ---
    # Load freecad file（主进程只用于获取 power_emitted_by_m2）
    # ---
    freecad_file = os.path.abspath('LFR.FCStd')
    data_file_spectrum = os.path.join('data', 'ASTMG173-direct.txt')

    # 主进程加载一次，仅用于计算 power_emitted_by_m2
    import otsun as _otsun
    power_emitted_by_m2 = _otsun.integral_from_data_file(data_file_spectrum)

    # ---
    # Simulation parameters
    # ---
    phi_ini   = 0.0
    phi_end   = 90.0 + 1.E-4
    phi_step  = 90.0

    number_of_rays        = 100000
    aperture_collector_Th = 11 * 0.5 * 32 * 1_000_000  # mm²
    CSR                   = 0.05

    # ---
    # Launch computations
    # ---
    for ph in np.arange(phi_ini, phi_end, phi_step):
        full_computation(
            ph,
            freecad_file,
            data_file_spectrum,
            aperture_collector_Th,
            number_of_rays,
            CSR,
            power_emitted_by_m2,
        )