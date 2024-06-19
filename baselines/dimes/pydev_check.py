import os


def check_pydevd():
    try:
        FILENAME = '/root/.pycharm_helpers/pydev/pydevd.py'

        with open(FILENAME, 'r') as f:
            lines = f.readlines()

        state = 0
        modified = False
        for idx, line in enumerate(lines):
            if state == 0:
                if "port = setup['port']" in line:
                    state = 1
                    continue
            elif state == 1:
                state = 2 if "host = setup['client']" in line else 0

            elif state == 2:
                with open(FILENAME + '.backup', 'w') as f:
                    f.writelines(lines)
                lines = lines[:idx-1] + ["    os.environ['PYDEV_PORT'] = str(port)\n"] + lines[idx-1:]
                modified = True
                break

        if modified:
            print("PYDEV_PORT not set. PLEASE RERUN the program")
            with open(FILENAME, 'w') as f:
                f.writelines(lines)
    except:
        pass


if __name__ == '__main__':
    check_pydevd()
