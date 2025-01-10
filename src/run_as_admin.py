import ctypes
import sys
import os
import psutil
import win32process  # part of pywin32
import win32con
import win32api
import win32security

# Constants to match the original code’s behavior
SE_DEBUG_NAME = "SeDebugPrivilege"

def user_is_admin():
    """
    Returns True if the current user context has admin privileges.
    Equivalent to the C code's UserIsAdmin() using IsUserAnAdmin().
    """
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        return False

def set_current_privilege(privilege_name, enable_privilege=True):
    """
    Replicates the SetCurrentPrivilege() logic from your C code.
    Acquires or relinquishes the specified privilege on the current process token.
    Returns True on success, False otherwise.
    """
    try:
        # Open the process token
        hToken = win32security.OpenProcessToken(
            win32api.GetCurrentProcess(),
            win32con.TOKEN_ADJUST_PRIVILEGES | win32con.TOKEN_QUERY
        )
        
        # Lookup the LUID for the privilege
        luid = win32security.LookupPrivilegeValue(None, privilege_name)
        
        # Now we need to build a TOKEN_PRIVILEGES structure
        if enable_privilege:
            new_privileges = [(luid, win32con.SE_PRIVILEGE_ENABLED)]
        else:
            new_privileges = [(luid, 0)]
        
        # Adjust the token
        win32security.AdjustTokenPrivileges(hToken, 0, new_privileges)
        
        # Check for error
        if win32api.GetLastError() != 0:
            return False
        
        return True
    except Exception as e:
        print(f"[set_current_privilege] Error: {e}")
        return False

def get_proc_name_by_id(pid):
    """
    Equivalent to GetProcnameByID in C code using psutil.
    Returns the process name (e.g., 'notepad.exe') or '' if not found.
    """
    if pid <= 0:
        return ""
    try:
        p = psutil.Process(pid)
        return p.name()
    except psutil.NoSuchProcess:
        return ""
    except Exception as e:
        print(f"[get_proc_name_by_id] Error: {e}")
        return ""

def get_parent_pid_by_pid(pid):
    """
    Equivalent to GetParentPIDByPID from the C code, but using psutil.
    Returns the parent PID or -1 on error.
    """
    if pid <= 0:
        return -1
    try:
        p = psutil.Process(pid)
        return p.ppid()
    except psutil.NoSuchProcess:
        return -1
    except Exception as e:
        print(f"[get_parent_pid_by_pid] Error: {e}")
        return -1

def run_as_admin(argv=None, wait=True):
    """
    Attempt to relaunch the current script with admin privileges using ShellExecuteEx with the 'runas' verb.
    Similar to the original code's ShellExecuteEx block.
    If wait=True, we wait until the new process finishes.
    """
    if argv is None:
        argv = sys.argv
    # Construct the command line
    cmd = f'"{sys.executable}"'
    # Join the rest of the arguments
    params = " ".join(f'"{arg}"' for arg in argv[1:])
    
    # ShellExecuteEx setup
    seh = ctypes.windll.shell32.ShellExecuteExW
    # Create a SHELLEXECUTEINFO struct
    # In Python, we can do a simpler approach with ShellExecute
    # but let's be close to your C code approach:
    class SHELLEXECUTEINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_ulong),
            ("fMask", ctypes.c_ulong),
            ("hwnd", ctypes.c_void_p),
            ("lpVerb", ctypes.c_wchar_p),
            ("lpFile", ctypes.c_wchar_p),
            ("lpParameters", ctypes.c_wchar_p),
            ("lpDirectory", ctypes.c_wchar_p),
            ("nShow", ctypes.c_int),
            ("hInstApp", ctypes.c_void_p),
            ("lpIDList", ctypes.c_void_p),
            ("lpClass", ctypes.c_wchar_p),
            ("hKeyClass", ctypes.c_ulong),
            ("dwHotKey", ctypes.c_ulong),
            ("hIconOrMonitor", ctypes.c_void_p),
            ("hProcess", ctypes.c_void_p)
        ]
    SEE_MASK_NOCLOSEPROCESS = 0x00000040
    
    sei = SHELLEXECUTEINFO()
    sei.cbSize = ctypes.sizeof(sei)
    sei.fMask = SEE_MASK_NOCLOSEPROCESS
    sei.hwnd = None
    sei.lpVerb = "runas"        # 'runas' is the admin elevation verb
    sei.lpFile = cmd            # program to run
    sei.lpParameters = params   # command line
    sei.lpDirectory = None
    sei.nShow = 1  # SW_SHOWNORMAL
    sei.hInstApp = None
    
    # Attempt execution
    success = seh(ctypes.byref(sei))
    if not success:
        # ShellExecuteEx returns >32 for success, or an error code <= 32
        error_code = ctypes.windll.kernel32.GetLastError()
        print(f"[run_as_admin] ShellExecuteEx failed: {error_code}")
        return False
    
    # Optionally wait for the launched process to finish
    if wait and sei.hProcess:
        ctypes.windll.kernel32.WaitForSingleObject(sei.hProcess, -1)
        ctypes.windll.kernel32.CloseHandle(sei.hProcess)
    
    return True

def main():
    # Replicates your sample main logic

    # 1. Set debug privilege
    set_current_privilege(SE_DEBUG_NAME, True)

    # 2. Check if admin
    is_admin = user_is_admin()
    privileges_str = "Admin" if is_admin else "!Admin"

    # 3. Get process/parent process info
    curr_pid = os.getpid()
    parent_pid = get_parent_pid_by_pid(curr_pid)
    parent_exe_name = get_proc_name_by_id(parent_pid)
    curr_exe_name = get_proc_name_by_id(curr_pid)

    print(f"[main] Privs: {privileges_str}, "
          f"Proc: {curr_exe_name}({curr_pid}), "
          f"Parent: {parent_exe_name}({parent_pid})")

    # 4. We are not admin and the parent is not an instance of ourself
    if not is_admin and (curr_exe_name.lower() != parent_exe_name.lower()):
        # re-run as admin
        print("[main] Attempting to run as admin...")
        success = run_as_admin(sys.argv, wait=True)
        if success:
            print("[main] Elevated process completed.")
        else:
            print("[main] Could not elevate privileges.")
        sys.exit(0)

    # 5. If we are admin
    elif is_admin:
        print("[main] We are admin. Doing admin stuff here...")
        # Wait for user input or do your privileged tasks
        input("Press Enter to continue...")

    # 6. We are still not admin
    else:
        print("[main] We are still not admin. Possibly continuing without elevation...")

if __name__ == "__main__":
    main()
