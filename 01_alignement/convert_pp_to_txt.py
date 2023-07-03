from tkinter import filedialog
from tkinter import *
import re

def convert_pp_to_txt(ppFile):
    with open(ppFile, 'r', encoding='utf-8') as myppfile:
        with open(ppFile + '.txt', 'w', encoding='utf-8') as mytxtfile:
            for myppline in myppfile:
                if "<point" in myppline:
                    x = re.search(r'(?:[, ])x="([-+]?\d*\.\d+|\d+)', myppline)
                    y = re.search(r'(?:[, ])y="([-+]?\d*\.\d+|\d+)', myppline)
                    z = re.search(r'(?:[, ])z="([-+]?\d*\.\d+|\d+)', myppline)
                    mytxtfile.write("%s %s %s\n" % (x.group(1), y.group(1), z.group(1)))


if __name__ == '__main__':
    # window for choosing a directory
    root = Tk()
    root.withdraw()  # use to hide tkinter window
    ppFile = filedialog.askopenfilename(initialdir="~/", title="Select the pp file containing points")

    convert_pp_to_txt(ppFile)


