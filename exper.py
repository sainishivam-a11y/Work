def entry_focus_in(event):
    if search_entry1.get() == "Enter City Name..":
        search_entry1.delete(0, 'end')
        search_entry1.config(fg="Black")


def entry_focus_out(event):
    if search_entry1 == "":
        search_entry1.insert(0, "Enter City Name..")
        search_entry1.config(fg="Gray")
        search_entry1.insert(0, "Enter City Name..")
        search_entry1.bind("<FocusIn>", entry_focus_in)
        search_entry1.bind("<FocusOut>", entry_focus_out)