import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fix mislabels: convert Person (3) -> Vehicle (0) for specified YOLO label files"
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        required=True,
        help="Path to labels directory (e.g., datasets/traffic_ai_balanced_11class_processed/labels/train)",
    )
    parser.add_argument(
        "--file_glob",
        type=str,
        default="balanced_inter_*.txt",
        help="Glob pattern for files to process",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only show what would change without writing files",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .bak copy before modifying each file",
    )
    return parser.parse_args()


def fix_file(path: Path, dry_run: bool, backup: bool) -> tuple[int, int]:
    """
    Returns (total_lines, changed_lines)
    """
    content = path.read_text(encoding="utf-8").splitlines()
    changed = 0
    new_lines = []
    for line in content:
        if not line.strip():
            new_lines.append(line)
            continue
        parts = line.split()
        try:
            cls_id = int(parts[0])
        except Exception:
            new_lines.append(line)
            continue

        # Convert Person (3) -> Vehicle (0)
        if cls_id == 3:
            parts[0] = "0"
            changed += 1
            new_lines.append(" ".join(parts))
        else:
            new_lines.append(line)

    if changed > 0 and not dry_run:
        if backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return len(content), changed


def main():
    args = parse_args()
    labels_dir = Path(args.labels_dir)
    files = sorted(labels_dir.glob(args.file_glob))
    total_files = 0
    total_changed = 0
    total_lines = 0
    total_lines_changed = 0

    for f in files:
        total_files += 1
        lines, changed = fix_file(f, args.dry_run, args.backup)
        total_lines += lines
        total_lines_changed += changed
        if changed:
            print(f"FIXED {f}  (changed {changed}/{lines} lines)")
        else:
            print(f"OK    {f}  (no change)")

    print("\nSummary:")
    print(f"Files scanned: {total_files}")
    print(f"Label lines scanned: {total_lines}")
    print(f"Label lines changed: {total_lines_changed}")
    if args.dry_run:
        print("Dry-run mode: no files were modified. Re-run without --dry_run to apply changes.")


if __name__ == "__main__":
    main()


