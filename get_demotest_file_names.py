def get_demotest_file_names(outpath, filename_outfile, filenameProgress):

    if outpath:
        outdir = outpath
    else:
        from check_psse_example_folder import check_psse_example_folder
        outdir = check_psse_example_folder()

    # outfile = os.path.join(outdir, 'outfile.out')
    outfile = filename_outfile
    progressFile = filenameProgress
    return outfile, progressFile
