import os
import re
import shutil
from tqdm import tqdm

def listdir(root: str,
            hidden: bool = False,
            full: bool = False,
            sort: bool = True,
            case_sensitive_sort: bool = False,
            exclude_patterns: list[str] = None,
            include_patterns: list[str] = None,
            any_exclude_patterns: bool = False,
            any_include_patterns: bool = False,
            dirs_only: bool = False,
            files_only: bool = False,
            recursive: bool = False,
            max_depth: bool = None,
            depth: bool = 0,
            follow_symlinks: bool = False,
            file_types: bool = None,
            return_attributes: bool = False) -> list:
    """
    Enhanced directory listing function with additional options for filtering, sorting, and attribute retrieval.

    Args:
        root (str): The directory to list.
        hidden (bool, optional): Whether to include hidden files. Defaults to False.
        full (bool, optional): Whether to include the full path. Defaults to False.
        sort (bool, optional): Whether to sort the output. Defaults to True.
        case_sensitive_sort (bool, optional): Whether the sorting is case sensitive. Defaults to False.
        exclude_patterns (list, optional): List of regular expression patterns to exclude. Defaults to None.
        include_patterns (list, optional): List of regular expression patterns to include. Defaults to None.
        any_exclude_patterns (list, optional): List of regular expression patterns to exclude. Defaults to None.
        any_include_patterns (list, optional): List of regular expression patterns to include. Defaults to None.
        dirs_only (bool, optional): Whether to only list directories. Defaults to False.
        files_only (bool, optional): Whether to only list files. Defaults to False.
        recursive (bool, optional): Whether to list subdirectories. Defaults to False.
        max_depth (int, optional): Maximum depth to list subdirectories. Defaults to None.
        depth (int, optional): Current depth in the recursive search. Defaults to 0.
        follow_symlinks (bool, optional): Whether to follow symbolic links. Defaults to False.
        file_types (list, optional): List of file extensions to include. Defaults to None.
        return_attributes (bool, optional): Whether to return file attributes. Defaults to False.

    Returns:
        list: List of directories or files (or both) in the directory, with optional attributes.
    """
    if depth == max_depth:
        return []

    try:
        entries = os.listdir(root)
    except OSError as e:
        print(f"Error accessing directory {root}: {e}")
        return []

    results = []
    for entry in entries:
        full_path = os.path.join(root, entry)
        if not hidden and entry.startswith('.'):
            continue
        if dirs_only and not os.path.isdir(full_path):
            continue
        if files_only and not os.path.isfile(full_path):
            continue
        if follow_symlinks and os.path.islink(full_path):
            full_path = os.path.realpath(full_path)
        if file_types and not any(full_path.endswith(f'.{ext}') for ext in file_types):
            continue
        if exclude_patterns and all(re.search(re.escape(pattern), entry) for pattern in exclude_patterns):
            continue
        if include_patterns and not all(re.search(re.escape(pattern), entry) for pattern in include_patterns):
            continue
        if any_exclude_patterns and any(re.search(re.escape(pattern), entry) for pattern in any_exclude_patterns):
            continue
        if any_include_patterns and not any(re.search(re.escape(pattern), entry) for pattern in any_include_patterns):
            continue

        if return_attributes:
            attributes = {
                'name': full_path if full else entry,
                'size': os.path.getsize(full_path),
                'mtime': os.path.getmtime(full_path)
            }
            results.append(attributes)
        else:
            results.append(full_path if full else entry)

    if sort:
        results.sort(key=lambda x: x.lower() if not case_sensitive_sort else x)

    if recursive:
        subdirs = [d for d in results if os.path.isdir(os.path.join(root, d))]
        for subdir in subdirs:
            subdir_path = os.path.join(root, subdir)
            results.extend(listdir(subdir_path,
                                   hidden,
                                   full,
                                   sort,
                                   case_sensitive_sort,
                                   exclude_patterns,
                                   include_patterns,
                                   dirs_only,
                                   files_only,
                                   recursive,
                                   max_depth,
                                   depth + 1,
                                   follow_symlinks,
                                   file_types,
                                   return_attributes))

    return results


def compress(root, *args):
    # Decide which folders to compress
    if not args:
        folders = listdir(root, dirs_only=True)
    else:
        folders = args

    for folder in folders:
        folder = os.path.join(root, folder)
        # Make sure folder exists
        if not os.path.isdir(folder):
            print(f"Folder '{folder}' does not exist, skipping.")
            continue
        
        # Create a zip archive: folder -> folder.zip
        archive_name = shutil.make_archive(folder, 'zip', folder)
        print(f"Compressed '{os.path.basename(folder)}' -> '{os.path.basename(archive_name)}'")
        # print(f"Compressed '{folder}' -> '{archive_name}'")

        # Remove the original folder after successful compression
        shutil.rmtree(folder)
        # print(f"Removed folder '{folder}'")

def decompress(root, *args):
    # Decide which archives to decompress
    if not args:
        archives = listdir(root, files_only=True, file_types=['zip'])
    else:
        # Convert folder names -> archive filenames
        archives = [f'{folder}.zip' for folder in args]

    for archive in archives:
        archive = os.path.join(root, archive)
        # Make sure archive exists
        if not os.path.isfile(archive):
            print(f"Archive '{archive}' does not exist, skipping.")
            continue
        
        # Extract archive: folder.zip -> folder/
        extract_folder = archive.replace('.zip', '')
        shutil.unpack_archive(archive, extract_folder)
        print(f"Decompressed '{os.path.basename(archive)}' -> '{os.path.basename(extract_folder)}'")
        # print(f"Decompressed '{archive}' -> '{extract_folder}'")

        # Remove the zip file after successful extraction
        os.remove(archive)
        # print(f"Removed archive '{archive}'")

def get_total_size(directory, use_progress=False):
    """
    Computes the total size of all non-symlink files in the given directory,
    including all files in its subdirectories.
    
    Parameters:
        directory (str): The path of the directory to scan.
        use_progress (bool): If True, display a progress bar while processing.
    
    Returns:
        int: The total size in bytes of all files found.
    """
    total_size = 0
    if use_progress:
        # First pass: count the total number of files to process.
        total_files = sum(len(files) for _, _, files in os.walk(directory))
        with tqdm(total=total_files, desc="Calculating total file size") as pbar:
            for dirpath, _, filenames in os.walk(directory):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if not os.path.islink(file_path):
                        try:
                            total_size += os.path.getsize(file_path)
                        except Exception as e:
                            print(f"Error getting size for '{file_path}': {e}")
                    pbar.update(1)
    else:
        for dirpath, _, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if not os.path.islink(file_path):
                    try:
                        total_size += os.path.getsize(file_path)
                    except Exception as e:
                        print(f"Error getting size for '{file_path}': {e}")
    return total_size

def recursive_walk_with_stopping_content(root, stopping_content):
    """
    Recursively walk a directory and return all paths that contain the stopping content
    without descending into the stopping content directories themselves.
    """
    result = []

    def walk_directory(current_dir):
        entries = os.listdir(current_dir)
        if any(entry in stopping_content for entry in entries):
            result.append(current_dir)
        for entry in entries:
            path = os.path.join(current_dir, entry)
            if os.path.isdir(path) and entry not in stopping_content:
                walk_directory(path)

    walk_directory(root)
    return result