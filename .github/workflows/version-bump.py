import toml
from pathlib import Path


def increment_version(version_str):
    major, minor, patch = map(int, version_str.split('.'))
    patch += 1  # Increment the patch version
    return f"{major}.{minor}.{patch}"


def main():
    # Path to your pyproject.toml
    pyproject_path = Path('pyproject.toml')

    # Read pyproject.toml
    with pyproject_path.open('r') as file:
        data = toml.load(file)

    # Increment version
    current_version = data['tool']['poetry']['version']
    new_version = increment_version(current_version)
    data['tool']['poetry']['version'] = new_version

    # Write back to pyproject.toml
    with pyproject_path.open('w') as file:
        toml.dump(data, file)

    print(f"Version updated: {current_version} -> {new_version}")


if __name__ == "__main__":
    main()
