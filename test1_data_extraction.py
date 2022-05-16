import numpy as np

def test1_data_extraction(show, filename_outfile, \
filenameProgress, filename_xlsx, \
 printScale, printCommand, chosen_channels, outpath=None):
    print('Hi! Commencing test1_data_extraction.\n')
    import dyntools
    import numpy as np

    from get_demotest_file_names import get_demotest_file_names
    outfile, progressFile = \
    get_demotest_file_names(outpath, filename_outfile, filenameProgress)
    print(outfile)

    # create object
    chnfobj = dyntools.CHNF(outfile)

    print('\n Testing call to get_data')
    sh_ttl, ch_id, ch_data = chnfobj.get_data()
    if printCommand:
        print(sh_ttl)
        print(ch_id)

    print('\n Testing call to get_id')
    sh_ttl, ch_id = chnfobj.get_id()
    if printCommand:
        print(sh_ttl)
        print(ch_id)

    print('\n Testing call to get_range')
    ch_range = chnfobj.get_range()
    if printCommand:
        print(ch_range)

    print('\n Testing call to get_scale')
    ch_scale = chnfobj.get_scale()
    if printCommand:
        print(ch_scale)

    print('\n Testing call to print_scale')
    if printCommand or printScale:
        chnfobj.print_scale()

    print('\n Testing call to txtout')

    chnfobj.txtout(channels=list(chosen_channels))

    print('\n Testing call to xlsout')

    import os
    if os.path.exists(filename_xlsx):
        os.remove(filename_xlsx)

    chnfobj.xlsout(channels=list(chosen_channels), show=show)
