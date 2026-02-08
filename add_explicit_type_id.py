"""
Add MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID to all MLIR pass classes
that inherit from impl:: base classes but are missing the macro.

This fixes Windows DLL boundary crashes in FallbackTypeIDResolver where
implicit TypeIDs (based on template instantiation addresses) differ across
DLLs, causing abort() at runtime.

Usage: python add_explicit_type_id.py [--dry-run]
"""

import subprocess
import re
import sys
import os


def find_files_missing_macro(root):
    """Find .cpp/.h files that have impl:: base classes but no explicit TypeID macro."""
    search_dirs = [os.path.join(root, d) for d in ("lib", "third_party") if os.path.isdir(os.path.join(root, d))]

    files_with_impl = set()
    files_with_macro = set()

    for search_dir in search_dirs:
        result = subprocess.run(
            ["grep", "-rl", r"public.*impl::", "--include=*.cpp", "--include=*.h", search_dir],
            capture_output=True,
            text=True,
        )
        files_with_impl.update(f.strip() for f in result.stdout.strip().split("\n") if f.strip())

        result = subprocess.run(
            [
                "grep", "-rl", "MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID", "--include=*.cpp", "--include=*.h",
                search_dir
            ],
            capture_output=True,
            text=True,
        )
        files_with_macro.update(f.strip() for f in result.stdout.strip().split("\n") if f.strip())

    return sorted(files_with_impl - files_with_macro)


def add_macro_to_file(filepath, dry_run=False):
    """Add MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID to pass classes in a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Pattern 1: single-line inheritance
    #   struct FooPass : public impl::FooBase<FooPass> {
    pattern1 = r"((?:struct|class)\s+(\w+)\s*:\s*public\s+\S*impl::\S+<\w+>\s*\{)"

    # Pattern 2: multi-line inheritance
    #   struct FooPass
    #       : public some::impl::FooBase<FooPass> {
    #   or
    #   struct FooPass
    #       : public some::impl::FooBase<
    #           FooPass> {
    pattern2 = r"((?:struct|class)\s+(\w+)\s*\n\s*:\s*public\s+\S*impl::\S+<\s*\n?\s*\w+>\s*\{)"

    matches = list(re.finditer(pattern1, content))
    if not matches:
        matches = list(re.finditer(pattern2, content))

    if not matches:
        return []

    added = []
    # Process in reverse order to preserve earlier match positions
    for m in reversed(matches):
        keyword = m.group(0).split()[0]  # "struct" or "class"
        class_name = m.group(2)
        end_pos = m.end()
        macro_line = f"\n  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID({class_name})"

        if keyword == "class":
            # For "class" (default private), we must insert after "public:".
            # Find the first "public:" after the opening brace.
            public_match = re.search(r"public\s*:", content[end_pos:])
            if public_match:
                insert_pos = end_pos + public_match.end()
                content = content[:insert_pos] + macro_line + content[insert_pos:]
            else:
                # No public: section found; add "public:" then the macro.
                content = content[:end_pos] + "\npublic:" + macro_line + content[end_pos:]
        else:
            # For "struct" (default public), insert right after the opening brace.
            content = content[:end_pos] + macro_line + content[end_pos:]

        added.append(class_name)

    if not dry_run:
        with open(filepath, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)

    added.reverse()
    return added


def main():
    dry_run = "--dry-run" in sys.argv
    root = os.path.dirname(os.path.abspath(__file__))

    missing_files = find_files_missing_macro(root)

    if not missing_files:
        print("All files already have MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID.")
        return

    print(f"Found {len(missing_files)} file(s) missing the macro:")
    total_added = 0
    warnings = []

    for filepath in missing_files:
        added = add_macro_to_file(filepath, dry_run=dry_run)
        if added:
            for class_name in added:
                action = "Would add" if dry_run else "Added"
                print(f"  {action} macro for {class_name} in {filepath}")
                total_added += 1
        else:
            warnings.append(filepath)

    if warnings:
        print(f"\nWARNING: Could not find pass class pattern in {len(warnings)} file(s):")
        for w in warnings:
            print(f"  {w}")
            # Show the impl:: line for manual inspection
            result = subprocess.run(
                ["grep", "-n", "impl::", w],
                capture_output=True,
                text=True,
            )
            for line in result.stdout.strip().split("\n"):
                print(f"    {line}")

    action = "Would modify" if dry_run else "Modified"
    print(f"\n{action} {total_added} class(es) in {len(missing_files) - len(warnings)} file(s).")
    if warnings:
        print(f"{len(warnings)} file(s) need manual inspection.")


if __name__ == "__main__":
    main()
