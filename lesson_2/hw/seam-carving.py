from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def find_vertical_seam(
    image: np.ndarray,
) -> np.ndarray:
    # 0. Переводим изображение в grayscale: энергия и переходы считаем по яркости.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    # m[i, j] — минимальная накопленная стоимость шва,
    # который заканчивается в пикселе (i, j).
    m = np.zeros((h, w), dtype=np.float32)
    # backtrack[i, j] хранит смещение по столбцу для предыдущей строки:
    # -1 (слева), 0 (сверху), +1 (справа).
    backtrack = np.zeros((h, w), dtype=np.int16)

    # Градиенты Sobel по x и y.
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # 1. База динамики для первой строки: локальная энергия без переходов.
    m[0] = np.abs(dx[0]) + np.abs(dy[0])
    # Шов начинается заполняться после полного подсчета m
    # Для востановления путей используйте backtrack
    seam = np.zeros(h, dtype=np.int32)

    for i in range(1, h):
        # Реализуйте динамику
        pass
    # Реализуйте восстановление шва
    pass
    return seam


def remove_vertical_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """удалить 1 вертикальный шов

    Args:
        image (np.ndarray): изображение (h, w, c)
        seam (np.ndarray): массив размера h. В iй позиции стоит индекс j
          -> шов проходит через (i,j)

    Returns:
        np.ndarray: изображение размера (h, w - 1, c)
    """
    # Hint: советуем использовать булевые маски
    pass


def carve_vertical(
    image: np.ndarray,
    num_seams: int,
) -> np.ndarray:
    assert num_seams > 0 and num_seams < image.shape[1]

    iterator = tqdm(range(num_seams), desc="Removing columns", unit="seam")

    out = image
    for _ in iterator:
        seam = find_vertical_seam(out)
        out = remove_vertical_seam(out, seam)
    return out


def carve_horizontal(
    image: np.ndarray,
    num_seams: int,
) -> np.ndarray:
    assert num_seams > 0 and num_seams < image.shape[0]

    # Просто поверните, используйте функцию для удаления вертикальных швов
    # и поверните обратно
    pass


def main() -> None:
    ASSETS_DIR = Path(__file__).parent.resolve() / "assets" / "seam-carving"
    input_path = ASSETS_DIR / "Broadway_tower_edit.jpg"
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не смог прочесть: {str(input_path)}")

    REMOVE_COLS = 200
    REMOVE_ROWS = 30

    out = image

    out = carve_vertical(out, REMOVE_COLS)
    out = carve_horizontal(out, REMOVE_ROWS)

    outp = ASSETS_DIR / "result.jpg"
    ok = cv2.imwrite(str(outp), out)
    if not ok:
        raise RuntimeError(f"Не получилось записать: {outp}")


if __name__ == "__main__":
    main()
