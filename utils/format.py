import json
import re
from pathlib import Path
from utils.io import read_json, save_json, get_files

def pretty_print_json(input_path, output_path=None, indent=2):
    with open(input_path, 'r') as f:
        data = json.load(f)

    # If no output_path is given, overwrite the original file
    if output_path is None:
        output_path = input_path

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=indent)

    print(f"Formatted JSON saved to {output_path}")


def clean_descriptor_text(desc, class_name):
    desc = desc.strip()
    class_name_lower = class_name.lower()

    # Step 1: Check for "which is" and extract everything after
    match = re.search(r"which is (.+)", desc, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Step 2: Handle exact class name templates like "An image of a bird: a <class_name>"
    regex_exact = rf"^a(n)? (image|photo|picture|closeup|drawing) of (a|an|the)?( bird:)? ?{re.escape(class_name_lower)}[.]?$"
    if re.match(regex_exact, desc, flags=re.IGNORECASE):
        return class_name

    # Step 3: Remove leading "A photo of a <class>, ..." up to and including class name
    regex_template = rf"^a(n)? (image|photo|picture|closeup|drawing) of (a|an|the)?( bird:)? ?{re.escape(class_name_lower)},?"
    desc = re.sub(regex_template, "", desc, flags=re.IGNORECASE).strip()

    # Step 4: Remove prefix "class_name has a ..." or "class_name has ..." case-insensitively
    regex_has_clause = rf"^(a[n]? )?{re.escape(class_name_lower)} has( a)? "
    desc = re.sub(regex_has_clause, "", desc, flags=re.IGNORECASE).strip()

    # Step 5: Strip prefix "class_name, which ..." if it still exists
    if desc.lower().startswith(class_name_lower):
        desc = desc[len(class_name):].lstrip(", ").strip()

    # Step 6: Strip common "which ..." prefixes
    for prefix in ["which has", "which is", "which"]:
        if desc.lower().startswith(prefix):
            desc = desc[len(prefix):].strip()
            break

    # # Final step: Remove all remaining occurrences of class name
    # desc = re.sub(re.escape(class_name), "", desc, flags=re.IGNORECASE).strip()
    desc = re.sub(r"\s+", " ", desc)  # Collapse extra spaces

    return desc


def remove_class_names_from_descriptors(folder_path, output_folder_path, global_descriptors=False):
    folder_path = Path(folder_path)
    output_folder_path = Path(output_folder_path)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    for file_path in get_files(folder_path, recursive=False):
        if file_path.suffix != ".json":
            continue

        data = read_json(file_path)

        if global_descriptors:
            deduped_descriptors = set()
            for class_name, descriptors in data.items():
                for desc in descriptors:
                    cleaned_desc = clean_descriptor_text(desc, class_name)
                    deduped_descriptors.add(cleaned_desc)
            output_data = {"descriptors": sorted(deduped_descriptors)}
            output_filename = "global_" + file_path.name
        else:
            cleaned_data = {}
            for class_name, descriptors in data.items():
                cleaned_set = set()
                for desc in descriptors:
                    cleaned_desc = clean_descriptor_text(desc, class_name)
                    cleaned_set.add(cleaned_desc)
                cleaned_data[class_name] = sorted(cleaned_set)
            output_data = cleaned_data
            output_filename = file_path.name

        output_path = output_folder_path / output_filename
        print(f"Saved cleaned descriptor set to: {output_path}")
        save_json(output_data, output_path)