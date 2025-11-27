import os
import shutil

def deploy_assets():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    SOURCE_DIR = os.path.join(ROOT_DIR, "src")
    DEST_DIR = os.path.join(ROOT_DIR, "frontend", "src")
    FILES_TO_COPY = [
        "preprocess.py",
        "core.py",
        "config.py",
        "utils/types.py"
    ]

    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' does not exist.")
        exit(1)

    try:
        os.makedirs(DEST_DIR, exist_ok=True)
        with open(os.path.join(DEST_DIR, "__init__.py"), 'w') as f:
            pass
        print(f"Ensure destination directory exists: {DEST_DIR}")
    except OSError as e:
        print(f"Error creating directory: {e}")
        exit(1)

    success_count = 0
    for filename in FILES_TO_COPY:
        dir_des = os.path.dirname(filename)
        if dir_des and not os.path.exists(dir_des):
            os.makedirs(dir_des)
        src_file = os.path.join(SOURCE_DIR, filename)
        dest_file = os.path.join(DEST_DIR, filename)

        if os.path.exists(src_file):
            try:
                shutil.copy2(src_file, dest_file)
                print(f"   └── Copied: {filename}")
                success_count += 1
            except Exception as e:
                print(f"   └── Failed to copy {filename}: {e}")
        else:
            print(f"   └── Warning: Source file not found: {filename}")

    print("-" * 30)
    if success_count == len(FILES_TO_COPY):
        print("Deployment complete! All files copied successfully.")
    else:
        print(f"Deployment finished with warnings. ({success_count}/{len(FILES_TO_COPY)} files copied)")

if __name__ == "__main__":
    deploy_assets()
