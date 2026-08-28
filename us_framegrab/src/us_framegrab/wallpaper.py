"""Saved sidebar wallpaper themes: portrait 9:16 crops plus a locked Default gray."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_ID = "default"
DEFAULT_NAME = "Default"
PORTRAIT_W = 9
PORTRAIT_H = 16
MIN_PORTRAIT_W = 18
INDEX_NAME = "index.yaml"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


@dataclass(frozen=True)
class WallpaperTheme:
    id: str
    name: str
    image: str = ""
    thumb: str = ""

    @property
    def locked(self) -> bool:
        return self.id == DEFAULT_ID


DEFAULT_THEME = WallpaperTheme(id=DEFAULT_ID, name=DEFAULT_NAME)


def ui_wallpaper_dir() -> Path:
    return Path(__file__).resolve().parent / "ui" / "wallpapers"


def migrate_wallpapers(dest: Path, legacy: Path | None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if legacy is None:
        return
    src = Path(legacy)
    if not src.is_dir() or src.resolve() == dest.resolve():
        return
    if (dest / INDEX_NAME).is_file():
        return
    import shutil

    for item in src.iterdir():
        if item.name.startswith("."):
            continue
        target = dest / item.name
        if not target.exists():
            shutil.copy2(item, target)


def default_wallpaper_dir(config_path: Path | None = None) -> Path:
    dest = ui_wallpaper_dir()
    dest.mkdir(parents=True, exist_ok=True)
    if config_path is not None:
        migrate_wallpapers(dest, Path(config_path).resolve().parent / "wallpapers")
    return dest


def cover_source_box(
    img_w: int,
    img_h: int,
    dest_w: int,
    dest_h: int,
    *,
    align: str = "bottom",
) -> tuple[float, float, float, float]:
    """Image-space rectangle that covers ``dest`` (x, y, w, h)."""
    iw, ih = max(float(img_w), 1.0), max(float(img_h), 1.0)
    dw, dh = max(float(dest_w), 1.0), max(float(dest_h), 1.0)
    scale = min(iw / dw, ih / dh)
    sw, sh = dw * scale, dh * scale
    sx = (iw - sw) * 0.5
    sy = (ih - sh) if align == "bottom" else (ih - sh) * 0.5
    return sx, sy, sw, sh


def list_image_paths(folder: Path) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        return []
    found = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    found.sort(key=lambda path: path.name.lower())
    return found


def portrait_wh(width: int) -> tuple[int, int]:
    width = max(MIN_PORTRAIT_W, int(width))
    return width, int(round(width * PORTRAIT_H / PORTRAIT_W))


def snap_portrait_size(width: int, height: int, max_w: int, max_h: int) -> tuple[int, int]:
    """Largest 9:16 size that fits, preferring the requested width."""
    max_w = max(2, int(max_w))
    max_h = max(2, int(max_h))
    min_w = min(MIN_PORTRAIT_W, max_w)
    width = max(min_w, min(int(width), max_w))
    height = max(1, int(round(width * PORTRAIT_H / PORTRAIT_W)))
    if height <= max_h:
        return width, height
    height = min(int(height), max_h)
    width = max(min_w, min(int(round(height * PORTRAIT_W / PORTRAIT_H)), max_w))
    height = max(1, int(round(width * PORTRAIT_H / PORTRAIT_W)))
    if height > max_h:
        height = max_h
        width = max(min_w, min(int(round(height * PORTRAIT_W / PORTRAIT_H)), max_w))
    return width, height


def largest_portrait_box(img_w: int, img_h: int) -> list[int]:
    width, height = snap_portrait_size(img_w, img_h, img_w, img_h)
    x0 = max(0, (int(img_w) - width) // 2)
    y0 = max(0, (int(img_h) - height) // 2)
    return [x0, x0 + width, y0, y0 + height]


def move_portrait_box(box: list[int], dx: int, dy: int, img_w: int, img_h: int) -> list[int]:
    x0, x1, y0, y1 = (int(v) for v in box)
    bw, bh = x1 - x0, y1 - y0
    x0 = min(max(0, x0 + int(dx)), max(0, int(img_w) - bw))
    y0 = min(max(0, y0 + int(dy)), max(0, int(img_h) - bh))
    return [x0, x0 + bw, y0, y0 + bh]


def resize_portrait_box(
    box: list[int],
    handle: str,
    dx: int,
    dy: int,
    img_w: int,
    img_h: int,
) -> list[int]:
    """Resize a 9:16 box from a handle; ``move`` translates it."""
    if handle == "move" or not handle:
        return move_portrait_box(box, dx, dy, img_w, img_h)
    x0, x1, y0, y1 = (int(v) for v in box)
    if handle in ("e", "ne", "se"):
        width, height = snap_portrait_size((x1 + int(dx)) - x0, y1 - y0, img_w, img_h)
        return _place_portrait(x0, y0 if handle != "ne" else y1 - height, width, height, img_w, img_h)
    if handle in ("w", "nw", "sw"):
        width, height = snap_portrait_size(x1 - (x0 + int(dx)), y1 - y0, img_w, img_h)
        return _place_portrait(x1 - width, y0 if handle != "nw" else y1 - height, width, height, img_w, img_h)
    if handle == "s":
        width, height = snap_portrait_size(x1 - x0, (y1 + int(dy)) - y0, img_w, img_h)
        return _place_portrait(x0, y0, width, height, img_w, img_h)
    if handle == "n":
        width, height = snap_portrait_size(x1 - x0, y1 - (y0 + int(dy)), img_w, img_h)
        return _place_portrait(x0, y1 - height, width, height, img_w, img_h)
    return move_portrait_box(box, dx, dy, img_w, img_h)


def crop_bgr(image: np.ndarray, box: list[int]) -> np.ndarray:
    height, width = image.shape[:2]
    x0, x1, y0, y1 = (int(v) for v in box)
    x0 = min(max(x0, 0), max(width - 1, 0))
    x1 = min(max(x1, x0 + 1), width)
    y0 = min(max(y0, 0), max(height - 1, 0))
    y1 = min(max(y1, y0 + 1), height)
    return np.ascontiguousarray(image[y0:y1, x0:x1])


def make_thumbnail(bgr: np.ndarray, thumb_w: int = 90) -> np.ndarray:
    import cv2

    width, height = portrait_wh(thumb_w)
    return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)


def _place_portrait(x0: int, y0: int, width: int, height: int, img_w: int, img_h: int) -> list[int]:
    x0 = min(max(0, int(x0)), max(0, int(img_w) - width))
    y0 = min(max(0, int(y0)), max(0, int(img_h) - height))
    return [x0, x0 + width, y0, y0 + height]


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        import cv2

        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        import cv2

        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(image[:, :, :3])


class WallpaperStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.active_id = DEFAULT_ID
        self.last_browse = ""
        self.themes: list[WallpaperTheme] = []
        self.reload()

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_NAME

    def reload(self) -> None:
        self.themes = []
        self.active_id = DEFAULT_ID
        self.last_browse = ""
        if not self.index_path.is_file():
            return
        payload = yaml.safe_load(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return
        self.last_browse = str(payload.get("last_browse") or "")
        active = str(payload.get("active") or DEFAULT_ID)
        raw_themes = payload.get("themes") or []
        if isinstance(raw_themes, list):
            for item in raw_themes:
                theme = _theme_from_mapping(item)
                if theme is not None and theme.id != DEFAULT_ID:
                    self.themes.append(theme)
        if active == DEFAULT_ID or any(theme.id == active for theme in self.themes):
            self.active_id = active
        else:
            self.active_id = DEFAULT_ID

    def list_all(self) -> list[WallpaperTheme]:
        return [DEFAULT_THEME, *self.themes]

    def get(self, theme_id: str) -> WallpaperTheme | None:
        if theme_id == DEFAULT_ID:
            return DEFAULT_THEME
        for theme in self.themes:
            if theme.id == theme_id:
                return theme
        return None

    def load_bgr(self, theme_id: str) -> np.ndarray | None:
        theme = self.get(theme_id)
        if theme is None or theme.locked or not theme.image:
            return None
        path = self.root / theme.image
        if not path.is_file():
            return None
        import cv2

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return None if image is None else image

    def load_thumb_bgr(self, theme_id: str) -> np.ndarray | None:
        theme = self.get(theme_id)
        if theme is None or theme.locked:
            return None
        for name in (theme.thumb, theme.image):
            if not name:
                continue
            path = self.root / name
            if path.is_file():
                import cv2

                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is not None:
                    return image
        return None

    def set_active(self, theme_id: str) -> str:
        if theme_id != DEFAULT_ID and self.get(theme_id) is None:
            theme_id = DEFAULT_ID
        self.active_id = theme_id
        self._write_index()
        return self.active_id

    def set_last_browse(self, folder: Path) -> None:
        self.last_browse = str(Path(folder))
        self._write_index()

    def add_theme(self, name: str, image: np.ndarray) -> WallpaperTheme:
        import cv2

        bgr = _as_bgr(image)
        if bgr.size == 0:
            raise ValueError("empty wallpaper image")
        theme_id = uuid.uuid4().hex[:10]
        stem = f"wp_{theme_id}"
        image_name = f"{stem}.jpg"
        thumb_name = f"{stem}_thumb.jpg"
        cv2.imwrite(str(self.root / image_name), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(str(self.root / thumb_name), make_thumbnail(bgr), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        label = str(name).strip() or stem
        theme = WallpaperTheme(id=theme_id, name=label, image=image_name, thumb=thumb_name)
        self.themes.append(theme)
        self._write_index()
        return theme

    def delete_theme(self, theme_id: str) -> bool:
        if theme_id == DEFAULT_ID:
            return False
        theme = self.get(theme_id)
        if theme is None or theme.locked:
            return False
        self.themes = [item for item in self.themes if item.id != theme_id]
        names = {theme.image, theme.thumb, f"wp_{theme_id}.jpg", f"wp_{theme_id}_thumb.jpg"}
        for name in names:
            if not name:
                continue
            path = self.root / name
            if path.is_file():
                path.unlink()
        for extra in self.root.glob(f"wp_{theme_id}*"):
            if extra.is_file():
                extra.unlink()
        if self.active_id == theme_id:
            self.active_id = DEFAULT_ID
        self._write_index()
        return True

    def _write_index(self) -> None:
        payload: dict[str, Any] = {
            "active": self.active_id,
            "last_browse": self.last_browse,
            "themes": [
                {"id": theme.id, "name": theme.name, "image": theme.image, "thumb": theme.thumb}
                for theme in self.themes
            ],
        }
        self.index_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def _theme_from_mapping(raw: Any) -> WallpaperTheme | None:
    if not isinstance(raw, dict):
        return None
    theme_id = str(raw.get("id") or "").strip()
    if not theme_id or theme_id == DEFAULT_ID:
        return None
    return WallpaperTheme(
        id=theme_id,
        name=str(raw.get("name") or theme_id),
        image=str(raw.get("image") or ""),
        thumb=str(raw.get("thumb") or ""),
    )
