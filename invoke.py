from sys import platform

if platform in ["win32", "win64"]:
    power_shell = True
else:
    power_shell = False
