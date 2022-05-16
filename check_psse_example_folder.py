def check_psse_example_folder():
    # if called from PSSE's Example Folder,
    # create report in subfolder 'Output_Pyscript'
    # and return the path for the same
    import os
    outdir = os.getcwd()  # returns the string containing the current directory
    cwd = outdir.lower()  # returns the path but in all lowercase
    # find returns the index of the first occurrence
    # of the string
    i = cwd.find('psse_files')
    j = cwd.find('python_practice_psse')
    k = cwd.find('e0040')
    # print(i, j, k)
    # The if statement below basically checks if the current directory
    # is the EXAMPLES folder of PSSE and makes a folder called
    # Output_Pyscript if it is not already made
    if i > 0 and j > i and k > j:  # called from Example folder
        # outdir = os.path.join(outdir, 'Output_Pyscript1')
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        return outdir
