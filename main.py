import argparse
import glob
import os
import sys

import cv2

from detector import decode_qr, decode_qr_adaptive, draw_results

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def run_single(args):
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: cannot read image '{args.image}'")
        sys.exit(1)

    category = args.category
    if category:
        results = decode_qr_adaptive(args.image, category)
    else:
        results = decode_qr(args.image)

    if not results:
        print("No QR code detected.")
        sys.exit(0)

    print(f"Detected {len(results)} QR code(s):\n")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] Type: {r['type']}")
        print(f"      Data: {r['data']}")
        print(f"      Rect: x={r['rect']['x']}, y={r['rect']['y']}, "
              f"w={r['rect']['w']}, h={r['rect']['h']}")
        print()

    annotated = draw_results(image, results)
    cv2.imwrite(args.output, annotated)
    print(f"Annotated image saved to: {args.output}")

    if not args.no_display:
        cv2.imshow("QR Detection Result", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def collect_images(directory: str) -> list[str]:
    files = []
    for f in sorted(os.listdir(directory)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTS:
            files.append(os.path.join(directory, f))
    return files


def run_batch(args):
    base = args.batch
    if not os.path.isdir(base):
        print(f"Error: '{base}' is not a directory")
        sys.exit(1)

    categories = []
    if args.category:
        cat_dir = os.path.join(base, args.category)
        if not os.path.isdir(cat_dir):
            print(f"Error: category directory '{cat_dir}' not found")
            sys.exit(1)
        categories = [args.category]
    else:
        categories = sorted([
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d)) and not d.startswith(".")
        ])

    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_detected = 0
    total_images = 0
    summary = []

    for cat in categories:
        cat_dir = os.path.join(base, cat)
        images = collect_images(cat_dir)
        if not images:
            continue

        detected = 0
        for idx, img_path in enumerate(images, 1):
            fname = os.path.basename(img_path)
            try:
                results = decode_qr_adaptive(img_path, cat)
            except Exception as e:
                print(f"  [{cat}] {idx}/{len(images)} {fname} — ERROR: {e}")
                results = []

            if results:
                detected += 1
                data_preview = results[0]["data"][:60]
                print(f"  [{cat}] {idx}/{len(images)} {fname} — OK ({len(results)} QR): {data_preview}")
            else:
                print(f"  [{cat}] {idx}/{len(images)} {fname} — FAIL")

            if output_dir:
                try:
                    image = cv2.imread(img_path)
                    if image is not None:
                        annotated = draw_results(image, results)
                        out_path = os.path.join(output_dir, cat, os.path.basename(img_path))
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        cv2.imwrite(out_path, annotated)
                except Exception:
                    pass

        total = len(images)
        total_detected += detected
        total_images += total
        pct = 100 * detected / total if total else 0
        summary.append((cat, detected, total, pct))

    col_w = max(len(s[0]) for s in summary) if summary else 8
    print(f"\n{'Category':<{col_w}}  Detected  Total   Rate")
    print("-" * (col_w + 30))
    for cat, det, tot, pct in summary:
        print(f"{cat:<{col_w}}  {det:>6d}/{tot:<4d}  {pct:5.1f}%")
    print("-" * (col_w + 30))
    overall_pct = 100 * total_detected / total_images if total_images else 0
    print(f"{'TOTAL':<{col_w}}  {total_detected:>6d}/{total_images:<4d}  {overall_pct:5.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="QR code detection and decoding with category-adaptive preprocessing"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a single input image")
    group.add_argument("--batch", metavar="DIR", help="Path to dataset directory (e.g. qrcodes/detection/)")

    parser.add_argument("-c", "--category", help="Category name (folder name). For single mode: sets preprocessing. For batch mode: process only this category.")
    parser.add_argument("-o", "--output", default="result.png", help="Output path for single-image mode (default: result.png)")
    parser.add_argument("--output-dir", metavar="DIR", help="Directory to save annotated images in batch mode")
    parser.add_argument("--no-display", action="store_true", help="Skip GUI window display")

    args = parser.parse_args()

    if args.image:
        run_single(args)
    else:
        run_batch(args)


if __name__ == "__main__":
    main()
