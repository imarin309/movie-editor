"""sample.pngで頭検出のテスト（形状判定付き）"""

import cv2
import numpy as np

# パラメータ
BOTTOM_REGION_RATIO = 0.1  # 下部10%
DARK_THRESHOLD = 50  # V < 50を暗いとする
DARK_RATIO_THRESHOLD = 0.3  # 30%以上

# 形状判定のパラメータ
MIN_CIRCULARITY = 0.1  # 最小円形度（緩い条件）
MAX_CIRCULARITY = 1.0  # 最大円形度
MIN_ASPECT_RATIO = 1.0  # 最小アスペクト比
MAX_ASPECT_RATIO = 15.0  # 最大アスペクト比（横長の形状も許容）


def is_semicircle_shape(contour):
    """輪郭が「上下平ら、左右丸い」形状かどうかを判定"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return False, 0, 0, {}, {}

    # 円形度を計算
    circularity = 4 * np.pi * area / (perimeter * perimeter)

    # アスペクト比を計算
    x, y, w, h = cv2.boundingRect(contour)
    if h == 0:
        return False, circularity, 0, {}, {}

    aspect_ratio = w / h

    # 輪郭の点を取得して形状を詳細に解析
    points = contour.reshape(-1, 2)

    # 上下の平坦性を確認
    y_coords = points[:, 1]
    y_min, y_max = y_coords.min(), y_coords.max()
    y_range = y_max - y_min

    if y_range == 0:
        return False, circularity, aspect_ratio, {}, {}

    # 上端10%の領域
    top_threshold = y_min + y_range * 0.1
    top_points = points[y_coords <= top_threshold]
    top_flatness = 0.0
    if len(top_points) > 1:
        top_y_std = np.std(top_points[:, 1])
        top_flatness = top_y_std / y_range

    # 下端10%の領域
    bottom_threshold = y_max - y_range * 0.1
    bottom_points = points[y_coords >= bottom_threshold]
    bottom_flatness = 0.0
    if len(bottom_points) > 1:
        bottom_y_std = np.std(bottom_points[:, 1])
        bottom_flatness = bottom_y_std / y_range

    # 左右の丸みを確認（片方だけでもOK）
    mid_y_min = y_min + y_range * 0.4
    mid_y_max = y_min + y_range * 0.6
    mid_points = points[(y_coords >= mid_y_min) & (y_coords <= mid_y_max)]

    has_side_bulge = False
    left_bulge_ratio = 0
    right_bulge_ratio = 0
    if len(mid_points) > 0:
        x_coords_mid = mid_points[:, 0]
        x_min_mid, x_max_mid = x_coords_mid.min(), x_coords_mid.max()

        # 左側の膨らみ：左端から中央部までの広がり
        left_bulge = x_min_mid - x
        # 右側の膨らみ：中央部から右端までの広がり
        right_bulge = (x + w) - x_max_mid

        # バウンディングボックスの幅に対する割合
        left_bulge_ratio = left_bulge / w if w > 0 else 0
        right_bulge_ratio = right_bulge / w if w > 0 else 0

        # 片方でも膨らみがあればOK（幅の15%以内であれば丸みがあると判定）
        if left_bulge_ratio < 0.15 or right_bulge_ratio < 0.15:
            has_side_bulge = True

    # 判定基準
    is_horizontal = aspect_ratio >= MIN_ASPECT_RATIO
    is_top_flat = top_flatness < 0.15
    is_bottom_flat = bottom_flatness < 0.15
    is_valid_circularity = MIN_CIRCULARITY <= circularity <= MAX_CIRCULARITY
    is_valid_aspect = aspect_ratio <= MAX_ASPECT_RATIO

    is_valid = (
        is_horizontal
        and is_top_flat
        and is_bottom_flat
        and has_side_bulge
        and is_valid_circularity
        and is_valid_aspect
    )

    details = {
        "top_flatness": top_flatness,
        "bottom_flatness": bottom_flatness,
        "left_bulge_ratio": left_bulge_ratio,
        "right_bulge_ratio": right_bulge_ratio,
        "has_side_bulge": has_side_bulge,
    }

    checks = {
        "is_horizontal": is_horizontal,
        "is_top_flat": is_top_flat,
        "is_bottom_flat": is_bottom_flat,
        "is_valid_circularity": is_valid_circularity,
        "is_valid_aspect": is_valid_aspect,
    }

    return is_valid, circularity, aspect_ratio, details, checks


# 画像を読み込み
img = cv2.imread("sample.png")
if img is None:
    print("エラー: sample.png が見つかりません")
    exit(1)

print(f"画像サイズ: {img.shape}")

# RGBに変換
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 画面下部の領域を取得
height, width = rgb.shape[:2]
bottom_region_start = int(height * (1.0 - BOTTOM_REGION_RATIO))
bottom_region = rgb[bottom_region_start:, :]
print(f"下部領域サイズ: {bottom_region.shape}")
print(f"下部領域開始位置: {bottom_region_start} (画像高さ: {height})")

# HSV色空間に変換
hsv = cv2.cvtColor(bottom_region, cv2.COLOR_RGB2HSV)

# 暗い色（黒）を検出
lower_dark = np.array([0, 0, 0])
upper_dark = np.array([180, 255, DARK_THRESHOLD])
mask = cv2.inRange(hsv, lower_dark, upper_dark)

# 暗いピクセルの割合を計算
total_pixels = bottom_region.shape[0] * bottom_region.shape[1]
dark_pixels = np.sum(mask > 0)
dark_ratio = dark_pixels / total_pixels

print("\n色検出結果:")
print(f"  総ピクセル数: {total_pixels}")
print(f"  暗いピクセル数: {dark_pixels}")
print(f"  暗い割合: {dark_ratio:.3f} ({dark_ratio * 100:.1f}%)")
print(f"  閾値 (30%): {DARK_RATIO_THRESHOLD}")
print(f"  色検出: {'あり' if dark_ratio >= DARK_RATIO_THRESHOLD else 'なし'}")

# 輪郭検出と形状判定
if dark_ratio >= DARK_RATIO_THRESHOLD:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("\n輪郭検出:")
    print(f"  検出された輪郭数: {len(contours)}")

    valid_contours = []
    debug_img = cv2.cvtColor(bottom_region, cv2.COLOR_RGB2BGR).copy()

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        min_area = total_pixels * 0.05

        if area < min_area:
            continue

        is_valid, circularity, aspect_ratio, details, checks = is_semicircle_shape(
            contour
        )

        print(f"\n  輪郭 {i + 1}:")
        print(f"    面積: {area:.0f} px ({area / total_pixels * 100:.1f}%)")
        print(
            f"    円形度: {circularity:.3f} "
            f"(範囲: {MIN_CIRCULARITY} - {MAX_CIRCULARITY})"
        )
        print(
            f"    アスペクト比: {aspect_ratio:.2f} "
            f"(範囲: {MIN_ASPECT_RATIO} - {MAX_ASPECT_RATIO})"
        )
        if details:
            print(f"    上部平坦性: {details['top_flatness']:.3f} (< 0.15)")
            print(f"    下部平坦性: {details['bottom_flatness']:.3f} (< 0.15)")
            print(
                f"    左側丸み: {details['left_bulge_ratio']:.3f}, "
                f"右側丸み: {details['right_bulge_ratio']:.3f} (< 0.15)"
            )
            print(f"    左右丸み判定: {details['has_side_bulge']}")
        print(f"    形状判定: {'✓ OK' if is_valid else '✗ NG'}")

        # 輪郭を描画
        color = (0, 255, 0) if is_valid else (0, 0, 255)  # 緑=OK, 赤=NG
        cv2.drawContours(debug_img, [contour], -1, color, 2)

        # バウンディングボックスを描画
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 1)

        if is_valid:
            valid_contours.append(contour)

    print("\n形状判定結果:")
    print(f"  有効な半円形状: {len(valid_contours)}個")
    print(f"  最終判定: {'頭あり ✓' if valid_contours else '頭なし ✗'}")

    # デバッグ画像を保存
    cv2.imwrite(
        "debug_bottom_region.png", cv2.cvtColor(bottom_region, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite("debug_mask.png", mask)
    cv2.imwrite("debug_contours.png", debug_img)

    print("\nデバッグ画像を保存しました:")
    print("  - debug_bottom_region.png: 下部領域")
    print("  - debug_mask.png: 暗い部分のマスク（白が検出された部分）")
    print("  - debug_contours.png: 輪郭と形状判定結果（緑=OK, 赤=NG）")
else:
    print("\n暗い領域が閾値未満のため、形状判定をスキップしました")
