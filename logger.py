class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    ORANGE = "\033[38;5;208m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_warn(text):
    print(bcolors.WARNING + "[!] " + bcolors.ENDC + text)


def print_info(text):
    print(bcolors.OKBLUE + "[ℹ] " + bcolors.ENDC + text)


def print_success(text):
    print(bcolors.OKGREEN + "[✔] " + bcolors.ENDC + text)


def print_fail(text):
    print(bcolors.FAIL + "[✖] " + bcolors.ENDC + text)
