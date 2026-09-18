from pathlib import Path

import cv2
import numpy as np

ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "mean-shift"
if not ASSETS_DIR.exists():
    raise FileNotFoundError(f"Ожидалось увидеть папку {ASSETS_DIR}")
FRAME_PATHS = [ASSETS_DIR / "inputs" / f"frame{i}.jpg" for i in range(1, 4)]
OUTPUT_DIR = ASSETS_DIR / "tracked"

# Начальный bounding box.
INITIAL_BBOX = (300, 200, 100, 50)

# Константы для фильтрации темных и малонасыщенных пикселей
SATURATION_GE_THAN = 60
VALUE_GE_THAN = 32


def clamp_window(
    x: int, y: int, w: int, h: int, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """
    Вспомогательная функция: проверяет, что bbox не заходит за границу изображения
    При необходимости, обрезает.
    """
    x = max(0, min(x, img_w - w))
    y = max(0, min(y, img_h - h))
    return x, y, w, h


def build_target_histogram(
    frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]
) -> np.ndarray:
    """Построение гистограммы по ROI объекта

    Args:
        frame_bgr (np.ndarray): Кадр в формате bgr
        bbox (tuple[int, int, int, int]): bbox для ROI

    Returns:
        np.ndarray: np.float32 массив размера 180 (кол-во бинов)
    """
    # 1. Выделяем из кадра ROI по bbox и переводим в HSV пространство
    pass

    # 2. Делаем маску для Hue:
    # маска состоит только из тех пикселей, гже
    # saturation >= SATURATION_GE_THAN
    # и value >= VALUE_GE_THAN
    pass

    # 3. Строим гистограмма по диапазону Hue [0, 180) и нормализация до суммы 1
    # Количество бинов == 180
    # Hint: Ипользуйте `np.histogram`
    # bins, range -- интересующие параметры
    # Не забываем провести нормализацию (сумма эл-тов == 1)
    pass


def back_project(frame_bgr: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """Построение матрицы вероятностей по гистограме
    (x, y) -> Hist[Hue(x, y)]

    Args:
        frame_bgr (np.ndarray): Кадр
        hist (np.ndarray): Гистограмма

    Returns:
        np.ndarray: Матрица вероятностей. W, H == W, H кадра (sanity check)
    """
    # 1. Перевод в HSV
    pass

    # 2. Строим карту вероятностей: каждому пикселю сопоставляем hist[Hue]
    pass

    # 3. Обнуляем вероятности по маске
    # маска состоит только из тех пикселей, гже
    # saturation >= SATURATION_GE_THAN
    # и value >= VALUE_GE_THAN
    pass


def mean_shift(
    prob_map: np.ndarray,
    window: tuple[int, int, int, int],
    max_iter: int = 20,
    eps: float = 1.0,
) -> tuple[tuple[int, int, int, int], int]:
    """Итеративный Mean Shift. В цикле пересчитываем центр масс, пока не сойдемся.
    Для реализации можете еще раз посмотреть .md файл

    Args:
        prob_map (np.ndarray): матрица вероятностей (по ней считаем центр масс)
        window (tuple[int, int, int, int]): Изначальное окно. Центр масс ищется в нем
        max_iter (int, optional): Максимальное колво итераций на пересчет центра масс. Defaults to 20.
        eps (float, optional): Толерантность. Defaults to 1.0.

    Returns:
        tuple[tuple[int, int, int, int], int]: новый bbox и кол-во итераций, потребовавшихся для пересчета
    """

    # 0. Не забываем приводить bbox в безопасный range
    img_h, img_w = prob_map.shape
    x, y, w, h = clamp_window(*window, img_w, img_h)

    for i in range(1, max_iter + 1):
        # 1. выделяем интересующий нас регион
        # 2. считаем центр масс -- новый центр нашего bbox
        # 3. пересчитываем координаты bbox
        # 4. если сдвиг окна меньше, чем eps -- останавливаемся
        pass

    return (x, y, w, h), max_iter


def draw_tracking(
    frame: np.ndarray, bbox: tuple[int, int, int, int], frame_idx: int, iters: int
) -> np.ndarray:
    """Вспомогательная функция для визуализации кадров с bboxами. Здесь не надо ничего изменять"""
    x, y, w, h = bbox
    out = frame.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    center = (x + w // 2, y + h // 2)
    cv2.circle(out, center, 4, (0, 0, 255), -1)
    cv2.putText(
        out,
        f"frame={frame_idx} iter={iters} bbox=({x},{y},{w},{h})",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in FRAME_PATHS:
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"Не найден фрейм: {path}")
        frames.append(frame)

    # Инициализация трекера по первому кадру
    track_window = INITIAL_BBOX
    roi_hist = build_target_histogram(frames[0], track_window)

    # Визуализация первого кадра
    vis1 = draw_tracking(frames[0], track_window, frame_idx=1, iters=0)
    cv2.imwrite(str(OUTPUT_DIR / "frame1_tracked.jpg"), vis1)

    for i in range(1, len(frames)):
        # 1. Back projection
        prob_map = back_project(frames[i], roi_hist)
        # 2. Mean shift
        track_window, iters = mean_shift(prob_map, track_window, max_iter=20, eps=1.0)
        # 3. Визуализация
        vis = draw_tracking(frames[i], track_window, frame_idx=i + 1, iters=iters)
        out_path = OUTPUT_DIR / f"frame{i + 1}_tracked.jpg"
        cv2.imwrite(str(out_path), vis)


if __name__ == "__main__":
    main()
