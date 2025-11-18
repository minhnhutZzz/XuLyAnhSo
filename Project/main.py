import tkinter as tk
from gui import ImageProcessorGUI


def main():
    root = tk.Tk()
    app = ImageProcessorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
    

