import tkinter as tk
from datetime import datetime

def add_entry():
    """
    Add two entries, one for a description, one for an amount,
    along with some labels
    """
    # create the label and entry widgets
    label1 = tk.Label(entry_frame, text="Volume of base added:")
    #label2 = tk.Label(entry_frame, text="Amount:")
    entry1 = tk.Entry(entry_frame)
    #entry2 = tk.Entry(entry_frame)

    # lay them out on the screen
    column, row = entry_frame.grid_size()
    label1.grid(row=row, column=0, sticky="e", pady=2)
    entry1.grid(row=row, column=1, sticky="ew", pady=2, padx=4)
    #label2.grid(row=row, column=2, sticky="e", pady=2)
    #entry2.grid(row=row, column=3, sticky="ew", pady=2, padx=4)
    entry_frame.grid_rowconfigure(row, weight=0)
    #entry_frame.grid_rowconfigure(row+1, weight=1)

    # save the entries, so we can retrieve the values
    entries.append((entry1))

    # give focus to the new entry widget
    entry1.focus_set()

def save():
    # iterate over the entries, printing the values
    for description_entry in entries:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("description: %s value: %s" %
              (description_entry.get(), current_time))


# this is our global list of entries
entries = []

# create the main window and buttons
root = tk.Tk()

entry_frame = tk.Frame(root)
button_frame = tk.Frame(root)

entry_frame.pack(side="top", fill="both", expand=True)
button_frame.pack(side="bottom", fill="x")

add_button = tk.Button(button_frame, text="add another entry", command=add_entry)
save_button = tk.Button(button_frame, text="Save", command=save)

add_button.pack(side="left")
save_button.pack(side="right")

# create the first entry
add_entry()

# start the main loop -- this is where the GUI starts waiting,
# and why you don't need to add your own loop.
root.mainloop()