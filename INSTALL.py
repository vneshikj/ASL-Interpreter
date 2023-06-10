import subprocess
import time

# Read the requirements.txt file
with open('requirements.txt', 'r') as file:
    requirements = file.readlines()

# Install packages in the order they appear in requirements.txt
for requirement in requirements:
    # Remove leading/trailing whitespaces and newlines
    requirement = requirement.strip()

    # Skip empty lines or comments starting with '#'
    if not requirement or requirement.startswith('#'):
        continue

    # Retry installation up to 3 times if there is an error
    for i in range(3):
        try:
            # Run pip command to install the package
            subprocess.run(['pip', 'install', requirement])
            break  # Installation successful, break out of retry loop
        except Exception as e:
            print(f"Error installing {requirement}: {str(e)}")
            print(f"Retrying in 5 seconds... (Attempt {i+1}/3)")
            time.sleep(5)
    else:
        print(f"Failed to install {requirement} after 3 attempts.")


