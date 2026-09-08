from pathlib import Path


def resolve_path(path: str | Path, data_root: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path(data_root) / path).resolve()


def video_files_from_file_list(file_list: str | Path, data_root: str | Path | None = None) -> list[str]:
    file_list_path = resolve_path(file_list, data_root) if data_root is not None else Path(file_list)
    if not file_list_path.exists():
        raise FileNotFoundError(f"File list not found: {file_list_path}")

    label_to_video: dict[int, str] = {}
    with file_list_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid file_list line {line_num}: expected '<path> <label> ...', got: {line!r}")

            video_fp = parts[0]
            if data_root is not None and not Path(video_fp).is_absolute():
                video_fp = str(resolve_path(video_fp, data_root))
            label = int(parts[1])
            if label < 0:
                raise ValueError(f"file_list labels must be nonnegative, got {label}")
            if label in label_to_video and label_to_video[label] != video_fp:
                raise ValueError(
                    f"Duplicate label {label} with conflicting paths in {file_list_path}: "
                    f"{label_to_video[label]!r} vs {video_fp!r}"
                )
            label_to_video[label] = video_fp

    if not label_to_video:
        raise ValueError(f"Empty file list: {file_list_path}")

    max_label = max(label_to_video)
    missing_labels = [idx for idx in range(max_label + 1) if idx not in label_to_video]
    if missing_labels:
        raise ValueError(f"file_list labels must be contiguous from 0; missing labels: {missing_labels[:10]}")

    return [label_to_video[idx] for idx in range(max_label + 1)]


def parse_file_list_frame_ranges(
    file_list: str | Path, data_root: str | Path | None = None
) -> list[tuple[str, int, int, int]]:
    """
    Parse a DALI video file list with per-clip frame spans: ``path label start end`` per line.

    Frame indices follow DALI's convention: frames read are ``start <= f < end`` (end exclusive).
    The number of frames per clip is ``end - start`` (= ``last_idx - start_idx + 1`` with
    ``last_idx = end - 1``).
    """
    file_list_path = resolve_path(file_list, data_root) if data_root is not None else Path(file_list)
    if not file_list_path.exists():
        raise FileNotFoundError(f"File list not found: {file_list_path}")

    rows: list[tuple[str, int, int, int]] = []
    with file_list_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid file_list line {line_num}: expected '<path> <label> <start> <end>', got: {line!r}"
                )
            video_fp = parts[0]
            if data_root is not None and not Path(video_fp).is_absolute():
                video_fp = str(resolve_path(video_fp, data_root))
            label = int(parts[1])
            start_f = int(parts[2])
            end_f = int(parts[3])
            if label < 0 or start_f < 0 or end_f <= start_f:
                raise ValueError(f"Invalid label or frame range on file_list line {line_num}: {line!r}")
            rows.append((video_fp, label, start_f, end_f))
    if not rows:
        raise ValueError(f"Empty file list: {file_list_path}")
    return rows


def write_file_list_frame_ranges(rows, destination: str | Path) -> None:
    """Write resolved DALI rows, retaining labels and half-open frame ranges."""
    with Path(destination).open("w", encoding="utf-8") as handle:
        for video_fp, label, start, end in rows:
            handle.write(f"{Path(video_fp).resolve()} {label} {start} {end}\n")
