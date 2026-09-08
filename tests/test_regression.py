"""semcorr 包的合成图像回归测试。

不依赖真实显微图像：程序化渲染已知畸变（旋转 + 各向异性缩放 + 透视）
的十字标记网格合成图，验证
    检测 → 亚像素精度 → 网格指派 → 缺失恢复 → 留一验证 → 校正 → 自检
全链路，以及 fit_affine / decompose_affine / build_ideal_grid /
subpixel_peak 等纯函数。

运行：
    python -m unittest tests.test_regression -v
    python -m pytest tests/test_regression.py -q
"""
import math
import unittest
from pathlib import Path

import cv2
import numpy as np

from semcorr import compat as mdc

# ---------- 合成场景参数 ----------
N_ROWS, N_COLS = 2, 3
IMG_W, IMG_H = 640, 420
MARGIN = 70.0
PITCH_X, PITCH_Y = 160.0, 160.0   # 正方形网格（与版图设计约定一致）
SPAN, ARM = 60, 12


def design_points():
    """理想网格坐标（行优先 M1..M6，与 assign_grid 编号一致）。"""
    return [(MARGIN + c * PITCH_X, MARGIN + r * PITCH_Y)
            for r in range(N_ROWS) for c in range(N_COLS)]


def true_homography():
    """已知"理想→畸变"映射：3° 旋转 × 各向异性缩放 + 平移 + 透视项。"""
    th = math.radians(3.0)
    A = (np.array([[math.cos(th), -math.sin(th)],
                   [math.sin(th), math.cos(th)]])
         @ np.diag([1.02, 0.98]))
    H = np.eye(3)
    H[:2, :2] = A
    H[:2, 2] = [6.0, -4.0]
    H[2, 0] = 2.0e-6
    H[2, 1] = 1.5e-6
    return H


def project(H, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts.astype(np.float32),
                                   np.asarray(H, np.float32))
    return out.reshape(-1, 2).astype(np.float64)


def draw_cross(img, cx, cy, val=210):
    xi, yi = int(round(cx)), int(round(cy))
    h = ARM // 2
    x0, y0 = xi - SPAN // 2, yi - SPAN // 2
    img[yi - h:yi + h + 1, x0:x0 + SPAN] = val
    img[y0:y0 + SPAN, xi - h:xi + h + 1] = val


def make_scene(faint_idx=None):
    """渲染带照明梯度的合成图；faint_idx 指定一个近背景对比度的暗十字。"""
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    img = (45 + 25.0 * xx / IMG_W).astype(np.uint8)
    for i, (x, y) in enumerate(design_points()):
        draw_cross(img, x, y, val=66 if i == faint_idx else 210)
    return img


class PureFunctionTests(unittest.TestCase):
    def test_fit_affine_recovers_exact_transform(self):
        rng = np.random.default_rng(0)
        src = rng.uniform(0, 1000, size=(10, 2))
        th = math.radians(17.0)
        R = np.array([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]])
        A = R @ np.array([[1.3, 0.2], [0.0, 1.1]])   # 旋转+缩放+剪切
        t = np.array([35.0, -80.0])
        dst = src @ A.T + t
        A_fit, t_fit, resid = mdc.fit_affine(src, dst)
        self.assertLess(np.max(np.abs(resid)), 1e-9)
        np.testing.assert_allclose(A_fit, A, atol=1e-9)
        np.testing.assert_allclose(t_fit, t, atol=1e-9)

    def test_decompose_affine_rotation_scale_orthogonal(self):
        th = math.radians(12.0)
        R = np.array([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]])
        d = mdc.decompose_affine(R @ np.diag([1.25, 0.8]))
        self.assertAlmostEqual(d["scale_x"], 1.25, places=9)
        self.assertAlmostEqual(d["scale_y"], 0.8, places=9)
        self.assertAlmostEqual(d["rotation_deg"], 12.0, places=9)
        self.assertAlmostEqual(d["non_orthogonal_deg"], 0.0, places=9)

    def test_decompose_affine_shear_angle(self):
        k = 0.2
        th = math.radians(7.0)
        R = np.array([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]])
        d = mdc.decompose_affine(R @ np.array([[1.0, k], [0.0, 1.0]]))
        self.assertAlmostEqual(abs(d["non_orthogonal_deg"]),
                               math.degrees(math.atan(k)), places=9)

    def test_build_ideal_grid_square_median_pitch(self):
        pts = [(c * 100.0 + r * 3.0, r * 100.0 + c * 2.0)
               for r in range(2) for c in range(3)]
        ideal, (pitch, dx_med, dy_med) = mdc.build_ideal_grid(pts, 2, 3)
        self.assertAlmostEqual(pitch, 100.0, places=9)
        arr = np.asarray(pts, np.float64)
        cx, cy = arr[:, 0].mean(), arr[:, 1].mean()
        expected = [(cx + (c - 1.0) * 100.0, cy + (r - 0.5) * 100.0)
                    for r in range(2) for c in range(3)]
        np.testing.assert_allclose(np.asarray(ideal), expected, atol=1e-9)

    def test_subpixel_peak_recovers_parabola_vertex(self):
        # 顶点偏移需在 ±0.5 px 限幅窗口内（该窗口另有边界 clamp 测试）
        score = np.zeros((11, 11), np.float32)
        yy, xx = np.mgrid[0:11, 0:11]
        score = -0.05 * ((xx - 5.3) ** 2 + (yy - 4.4) ** 2)
        x, y = mdc.subpixel_peak(score.astype(np.float32), 5, 4)
        self.assertAlmostEqual(x, 5.3, places=5)
        self.assertAlmostEqual(y, 4.4, places=5)

    def test_subpixel_peak_clamps_at_border(self):
        score = np.zeros((7, 7), np.float32)
        score[0, 0] = 1.0
        x, y = mdc.subpixel_peak(score, 0, 0)
        self.assertEqual((x, y), (0.0, 0.0))


class GridAssignmentTests(unittest.TestCase):
    def test_assign_grid_row_major_order(self):
        truth = project(true_homography(), design_points())
        cands = []
        for i in np.random.default_rng(1).permutation(6):
            cands.append({"rx": truth[i][0], "ry": truth[i][1],
                          "combined": 1.0})
        ordered = mdc.assign_grid(cands, N_ROWS, N_COLS)
        np.testing.assert_allclose(np.asarray(ordered), truth, atol=1e-9)


class SimilarityTests(unittest.TestCase):
    """fit_similarity（Umeyama）：2x2 等无冗余网格的几何指派校验基础。"""

    def test_recovers_exact_similarity(self):
        rng = np.random.default_rng(7)
        src = rng.uniform(0, 100, size=(8, 2))
        th = math.radians(17.0)
        R = np.array([[math.cos(th), -math.sin(th)],
                      [math.sin(th), math.cos(th)]])
        s, t = 1.3, np.array([35.0, -80.0])
        dst = src @ (s * R).T + t
        s_f, R_f, t_f, resid = mdc.fit_similarity(src, dst)
        self.assertLess(np.max(np.abs(resid)), 1e-9)
        self.assertAlmostEqual(s_f, s, places=9)
        np.testing.assert_allclose(R_f, R, atol=1e-9)
        np.testing.assert_allclose(t_f, t, atol=1e-9)

    def test_sheared_parallelogram_rejected(self):
        # 0937 式列错位：上下行取自不同列对 → 平行四边形（任意平行四边形
        # 都是某正方形的仿射像，但不是相似像）→ 相似变换 RMS 必须巨大
        quad = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        unit = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        s, _, _, resid = mdc.fit_similarity(unit, quad)
        self.assertGreater(mdc.rms_of(resid) / s, 0.2,
                           "列错位四边形必须被相似变换 RMS 暴露")


class LeaveOneOutTests(unittest.TestCase):
    def test_outlier_mark_is_flagged(self):
        ideal = design_points()
        dst = [tuple(p) for p in project(true_homography(), ideal)]
        dst[3] = (dst[3][0] + 8.0, dst[3][1] - 6.0)   # M4 注入离群
        for method in ("affine", "homography"):
            loo = mdc.leave_one_out(ideal, dst, method)
            self.assertIn("M4", loo["suspects"],
                          "%s 模型未标记注入的离群点" % method)

    def test_clean_points_not_flagged(self):
        ideal = design_points()
        dst = [tuple(p) for p in project(true_homography(), ideal)]
        loo = mdc.leave_one_out(ideal, dst, "homography")
        self.assertEqual(loo["suspects"], [])

    def test_no_redundancy_note_for_2x2(self):
        """2x2 = 4 点：剔除 1 个后剩 3 个，不足以拟合单应 →
        必须返回"不适用"而不是崩溃（真实 0937 4-mark 场景的回归）。"""
        ideal = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        loo = mdc.leave_one_out(ideal, ideal, "homography")
        self.assertIn("note", loo)
        self.assertEqual(loo["per_mark"], [])
        self.assertEqual(loo["suspects"], [])


class Grid2x2Tests(unittest.TestCase):
    """标准 4-mark（2x2）图像端到端：单格精确单应 + 自检。"""

    def test_full_pipeline_on_synthetic_2x2(self):
        import argparse
        import json
        import tempfile
        ideal2 = [(90.0, 90.0), (370.0, 90.0), (90.0, 370.0), (370.0, 370.0)]
        img = np.full((460, 460), 50, np.uint8)
        for x, y in ideal2:
            draw_cross(img, x, y)
        th = math.radians(2.0)
        H2 = np.eye(3)
        H2[:2, :2] = (np.array([[math.cos(th), -math.sin(th)],
                                [math.sin(th), math.cos(th)]])
                      @ np.diag([1.02, 0.99]))
        H2[:2, 2] = [5.0, -3.0]
        H2[2, 0] = 1.0e-6
        distorted = cv2.warpPerspective(img, H2, (460, 460))
        tmpdir = Path(tempfile.mkdtemp())
        scene = tmpdir / "grid22.png"
        cv2.imwrite(str(scene), distorted)
        outdir = tmpdir / "out"
        args = argparse.Namespace(image=str(scene), design=None,
                                  grid="2x2", outdir=str(outdir),
                                  affine=False, diagnostics=True)
        mdc.process_single(str(scene), args, outdir=str(outdir))
        with open(outdir / "diagnostics" / "grid22_report.json") as handle:
            report = json.load(handle)
        self.assertEqual(report["method"], "exact-square")
        self.assertEqual(len(report["exact_square_cells"]), 1)
        sc = report["self_check"]
        self.assertEqual(sc["n_detected"], 4)
        self.assertLess(sc["rms"], 0.5,
                        "2x2 校正后自检 RMS %.3f 应处于噪声量级" % sc["rms"])
        # 校正后 4 mark 必须构成正方形
        det = {m["id"]: m["detected"] for m in sc["marks"]}
        TL, TR, BL, BR = [det[i] for i in ("M1", "M2", "M3", "M4")]
        sides = sorted([math.dist(TL, TR), math.dist(BL, BR),
                        math.dist(TL, BL), math.dist(TR, BR)])
        self.assertLess(sides[-1] - sides[0], 0.8,
                        "2x2 边长极差 %.3f px 过大" % (sides[-1] - sides[0]))


class EndToEndTests(unittest.TestCase):
    """合成畸变图 → 检测 → 拟合 → 校正 → 自检 的全链路验证。"""

    @classmethod
    def setUpClass(cls):
        cls.ideal = design_points()
        cls.H_true = true_homography()
        cls.truth = project(cls.H_true, cls.ideal)
        cls.distorted = cv2.warpPerspective(
            make_scene(), cls.H_true, (IMG_W, IMG_H))

    def test_detection_subpixel_accuracy(self):
        accepted, _, _ = mdc.detect_marks(self.distorted, verbose=False)
        self.assertEqual(len(accepted), 6)
        ordered = mdc.assign_grid(accepted, N_ROWS, N_COLS)
        err = np.linalg.norm(np.asarray(ordered) - self.truth, axis=1)
        self.assertLess(err.max(), 0.5,
                        "亚像素定位误差应 < 0.5 px，实际 %s" % np.round(err, 3))

    def test_homography_fit_and_self_check(self):
        accepted, _, _ = mdc.detect_marks(self.distorted, verbose=False)
        ordered = mdc.assign_grid(accepted, N_ROWS, N_COLS)
        H_fit, resid = mdc.fit_model(self.ideal, ordered, "homography")
        self.assertLess(mdc.rms_of(resid), 0.5)
        corrected = cv2.warpPerspective(
            self.distorted, H_fit, (IMG_W, IMG_H),
            flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE)
        sc = mdc.self_check(corrected, self.ideal, N_ROWS, N_COLS)
        self.assertEqual(sc["n_detected"], 6,
                         "校正图上应重新检出全部标记")
        self.assertLess(sc["rms"], 1.0,
                        "校正后残差 RMS 应 < 1 px，实际 %.3f" % sc["rms"])

    def test_missing_faint_mark_is_recovered(self):
        """十字存在但对比度过低被粗筛漏检 → 网格预测 + 局部重搜应找回。"""
        faint = cv2.warpPerspective(
            make_scene(faint_idx=4), self.H_true, (IMG_W, IMG_H))
        accepted, _, blur = mdc.detect_marks(faint, verbose=False)
        self.assertEqual(len(accepted), 5, "暗十字应被粗筛漏检")
        rec = mdc.recover_missing(accepted, N_ROWS, N_COLS, blur)
        self.assertIsNotNone(rec)
        self.assertGreaterEqual(rec["score"], mdc.RECOVER_SCORE)
        err = math.hypot(rec["rx"] - self.truth[4][0],
                         rec["ry"] - self.truth[4][1])
        self.assertLess(err, 3.0, "恢复位置偏差应 < 3 px，实际 %.2f" % err)

    def test_truly_absent_mark_fails_recovery(self):
        """十字确实不存在时，恢复必须失败而不是捏造一个位置。"""
        img = self.distorted.copy()
        x, y = self.truth[4]
        img[max(0, int(y) - 70):int(y) + 70,
            max(0, int(x) - 70):int(x) + 70] = 55
        accepted, _, blur = mdc.detect_marks(img, verbose=False)
        self.assertEqual(len(accepted), 5)
        rec = mdc.recover_missing(accepted, N_ROWS, N_COLS, blur)
        self.assertIsNone(rec)


class CrossShapeTests(unittest.TestCase):
    """臂-角形状验证：区分十字与圆形颗粒（后者模板/对称分都可能很高）。"""

    @staticmethod
    def _patch(mask):
        return np.where(mask, 220, 50).astype(np.uint8)

    def test_cross_scores_high(self):
        mask = np.zeros((61, 61), bool)
        mask[27:34, :] = True
        mask[:, 27:34] = True
        self.assertGreater(mdc.cross_shape_score(self._patch(mask),
                                                 30, 30, 61), 0.9)

    def test_disk_scores_low(self):
        mask = np.zeros((61, 61), bool)
        cv2.circle(mask.astype(np.uint8), (30, 30), 26, 1, -1)
        self.assertLess(mdc.cross_shape_score(self._patch(mask.astype(bool)),
                                              30, 30, 61), 0.2)


class GridSelfHealTests(unittest.TestCase):
    """复现真实 3131-2 场景：碎屑污染真 mark（图像门误拒）+ 亮圆斑
    挤占空格位（图像门误收）——网格几何自愈必须逐出假点、找回真 mark。"""

    def test_occluded_mark_recovered_impostor_expelled(self):
        import argparse
        import json
        import tempfile
        img = make_scene()
        # 干扰物：一个图像特征完美（模板/对称/形状全过）的真十字，
        # 但不在网格上——芯片上其他结构的对准标记。挤占 M5 格位。
        # 几何裁判必须逐出它；模板分数再高也不能留。
        draw_cross(img, 330, 160)
        # 暗色污染物咬掉真 M5 的左臂 → 对称分掉线、被图像门误拒
        m5x, m5y = design_points()[4]
        cv2.circle(img, (int(m5x) - 14, int(m5y)), 14, 50, -1)
        tmpdir = Path(tempfile.mkdtemp())
        scene = tmpdir / "selfheal.png"
        cv2.imwrite(str(scene), img)
        outdir = tmpdir / "out"
        args = argparse.Namespace(image=str(scene), design=None,
                                  grid="2x3", outdir=str(outdir),
                                  affine=False, diagnostics=True)
        mdc.process_single(str(scene), args, outdir=str(outdir))
        with open(outdir / "diagnostics" / "selfheal_report.json") as handle:
            report = json.load(handle)
        # 必须触发自愈：逐出挤位的圆斑，按几何预测找回被污染的 M5
        self.assertTrue(report["repair"], "未触发网格自愈")
        self.assertLess(report["used_model_rms_px"], 2.0,
                        "自愈后 RMS 应回到亚像素/低像素量级")
        sc = report["self_check"]
        self.assertGreaterEqual(sc["n_detected"], 5)
        self.assertLess(sc["rms"], 2.0)
        # 回填位置应落在真实 M5 附近（而不是圆斑处）
        rec = report["repair"][0]["recovered_at"]
        err = math.hypot(rec[0] - m5x, rec[1] - m5y)
        self.assertLess(err, 5.0, "回填位置偏离真实 M5 %.1f px" % err)


class ExactSquareWarpTests(unittest.TestCase):
    """分格精确单应（默认校正模型）：校正后 6 个 mark 中心必须严格
    构成正方形（样品定位坐标系的硬性要求）。"""

    @classmethod
    def setUpClass(cls):
        cls.ideal = design_points()
        cls.H_true = true_homography()
        cls.truth = project(cls.H_true, cls.ideal)
        distorted = cv2.warpPerspective(
            make_scene(), cls.H_true, (IMG_W, IMG_H))
        accepted, _, _ = mdc.detect_marks(distorted, verbose=False)
        cls.ordered = mdc.assign_grid(accepted, N_ROWS, N_COLS)
        cls.corrected, cls.cells = mdc.warp_exact_square(
            distorted, cls.ideal, cls.ordered, N_ROWS, N_COLS)
        _, _, cls.blur = mdc.detect_marks(cls.corrected, verbose=False)

    def test_per_cell_solves_exactly(self):
        for cell in self.cells:
            # findHomography 内部为 float32，精度极限 ~1e-5 px
            self.assertLess(cell["max_corner_residual_px"], 1e-3,
                            "4 点单应必须精确解")

    def test_both_cells_share_marks(self):
        flat = {m for cell in self.cells for m in cell["marks"]}
        self.assertEqual(flat, {"M%d" % (i + 1) for i in range(6)})

    def test_marks_form_exact_square_after_correction(self):
        sc = mdc.self_check(self.corrected, self.ideal, N_ROWS, N_COLS)
        self.assertEqual(sc["n_detected"], 6)
        det = {m["id"]: m["detected"] for m in sc["marks"]}
        for ids in (["M1", "M2", "M4", "M5"], ["M2", "M3", "M5", "M6"]):
            TL, TR, BL, BR = [det[i] for i in ids]
            sides = sorted([math.dist(TL, TR), math.dist(BL, BR),
                            math.dist(TL, BL), math.dist(TR, BR)])
            # 边长极差限制在检测噪声量级（160 px 间距）
            self.assertLess(sides[-1] - sides[0], 0.8,
                            "%s 边长极差 %.3f px 过大" % (ids, sides[-1] - sides[0]))
            self.assertLess(abs(math.dist(TL, BR) - math.dist(TR, BL)),
                            0.8, "%s 对角线不相等" % ids)

    def test_exact_square_reaches_noise_floor(self):
        _, resid_h = mdc.fit_model(self.ideal, self.ordered, "homography")
        sc = mdc.self_check(self.corrected, self.ideal, N_ROWS, N_COLS)
        # 纯投影真值下全局单应已达噪声水平；分格精确模型同样应处于
        # 噪声量级（远小于 1 px），即 mark 中心在校正图中构成正方形
        self.assertLess(sc["rms"], 0.3,
                        "校正后自检 RMS %.3f px 应处于检测噪声量级" % sc["rms"])
        self.assertLess(sc["rms"], 3.0 * mdc.rms_of(resid_h) + 0.5)


class RobustRefineTests(unittest.TestCase):
    """污染感知重定位：数字/碎屑压在 mark 旁时，剔除污染像素、
    只用十字干净结构定位。（基线对照用真实图像验证：0937 的 M1
    两次定位不一致从 1.12 px 降到 0.85 px；二值合成图上普通 NCC
    因局部归一化对污染不敏感，无法充当对照，故只测精度本身。）"""

    @staticmethod
    def _scene(contam):
        img = np.full((201, 201), 50, np.uint8)
        img[94:107, 70:131] = 210
        img[70:131, 94:107] = 210
        if contam == "bright":
            # 数字块角落压进模板零区（对角象限，大小不等）
            img[60:92, 60:92] = 210
            img[126:146, 126:141] = 210
        elif contam == "dark":
            # 暗污染物咬掉左臂端与上臂端
            img[88:113, 55:90] = 50
            img[55:90, 88:113] = 50
        # SEM 图像有 PSF 模糊，灰度梯度是亚像素定位的前提
        return cv2.GaussianBlur(img, (0, 0), 2.0)

    def test_bright_contamination_accuracy(self):
        rx, ry, sc = mdc.robust_refine(self._scene("bright").astype(np.float64),
                                       100.5, 100.5, 60.0, 0.2)
        self.assertLess(math.hypot(rx - 100, ry - 100), 0.35,
                        "亮污染下鲁棒定位偏差 %.3f px 过大" % math.hypot(rx - 100, ry - 100))
        self.assertGreater(sc, 0.8)

    def test_dark_occlusion_accuracy(self):
        rx, ry, sc = mdc.robust_refine(self._scene("dark").astype(np.float64),
                                       99.0, 99.0, 60.0, 0.2)
        self.assertLess(math.hypot(rx - 100, ry - 100), 0.35,
                        "遮挡下鲁棒定位偏差 %.3f px 过大" % math.hypot(rx - 100, ry - 100))
        self.assertGreater(sc, 0.8)

    def test_clean_mark_unchanged(self):
        rx, ry, score = mdc.robust_refine(self._scene(None).astype(np.float64),
                                          100.0, 100.0, 60.0, 0.2)
        self.assertLess(math.hypot(rx - 100, ry - 100), 0.1)
        self.assertGreater(score, 0.95)


if __name__ == "__main__":
    unittest.main()
