import os
import sys  # noqa: F401
import contextlib
import collections
import pssepath
pssepath.add_pssepath()
import numpy as np
import psspy
import excelpy
import tail
import time
import winsound
duration = 1000  # milliseconds duration of alert sound
freq = 440  # also for alert sound


start = time.time()

def check_psse_example_folder():
    # if called from PSSE's Example Folder,
    # create report in subfolder 'Output_Pyscript'
    # and return the path for the same
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

# =============================================================================================


def lines_that_contain(string, fp):
    return [line for line in fp if string in line]

def get_demotest_file_names(outpath):

    if outpath:
        outdir = outpath
    else:
        outdir = check_psse_example_folder()

    # outfile = os.path.join(outdir, 'outfile.out')
    progressFile = filenameProgress
    outfile = filename_outfile
    return outfile, progressFile

# =============================================================================================
# Run Dynamic simulation on system_name to generate .out files

def run_simulation(datapath, savfile, snpfile, outfile, progressFile):

    import psspy
    psspy.psseinit(50000)

    if datapath:
        savfile = os.path.join(datapath, savfile)
        snpfile = os.path.join(datapath, snpfile)

    psspy.lines_per_page_one_device(1, 10000000)
    psspy.lines_per_page_one_device(2, 10000000)
    psspy.progress_output(2, progressFile, [0, 0])

    ierr = psspy.case(savfile)
    if ierr:
        psspy.progress_output(1, "", [0, 0])
        print(" psspy.case Error")
        return

    import numpy as np

    ierr, numBuses = psspy.abuscount(-1, 2)
    global volts
    volts = np.empty((0, numBuses))

    ierr, numBranches = psspy.aflowcount(-1, 1, 1, 2)
    global currents
    currents = np.empty((0, numBranches))

    ierr = psspy.rstr(snpfile)
    if ierr:
        psspy.progress_output(1, "", [0, 0])
        print(" psspy.rstr Error")
        return

    psspy.lines_per_page_one_device(1, 10000000)

    # psspy.dynamics_solution_param_2(realar3=0.001)
    psspy.dynamics_solution_param_2(realar3=0.01)
    # psspy.dynamics_solution_param_2(realar3=0.0333)

    ierr, mbase1 = psspy.macdat(1, '1', 'MBASE')
    print('mbase1 =', mbase1)
    ierr, mbase2 = psspy.macdat(2, '1', 'MBASE')
    ierr, mbase3 = psspy.macdat(3, '1', 'MBASE')
    # add dummy governor without droop and time constants
    ierr = psspy.add_plant_model(1,'1',7,r"""IEESGO""",0,"",0,[],[],\
    11,[0.0,0.0,0.05,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0])
    # print('ierr for add_plant_model =', ierr)
    ierr = psspy.add_plant_model(2,'1',7,r"""IEESGO""",0,"",0,[],[],\
    11,[0.0,0.0,0.05,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0])
    ierr = psspy.add_plant_model(3,'1',7,r"""IEESGO""",0,"",0,[],[],\
    11,[0.0,0.0,0.05,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0])
    psspy.machine_array_channel([-1,6, 1], '1' ,"")
    psspy.machine_array_channel([-1,6, 2], '1' ,"")
    psspy.machine_array_channel([-1,6, 3], '1' ,"")

    psspy.strt(0, outfile)
    psspy.run(0, 0.99, 1, 1, 0)

    import numpy as np
    import random as rn

    t_start = 1.00
    t_end = 600.00
    # t_increment = 3.0/30
    # t_increment = 0.1
    if system_name == 'ieee9':
        load_increment_bus001 = 27.0
        load_increment_bus002 = 29.0
        load_increment_bus003 = 21.0
        # load_increment_bus001 = 7.0
        # load_increment_bus002 = 9.0
        # load_increment_bus003 = 8.0

        ierr = psspy.bsyso(1, 5)
        ierr = psspy.bsyso(2, 6)
        ierr = psspy.bsyso(3, 8)
        ierr = psspy.bsyso(4, 1)
        ierr = psspy.bsyso(5, 2)
        ierr = psspy.bsyso(6, 3)

    elif system_name == 'ieee39':
        load_increment_bus001 = 22.0
    else:
        load_increment_bus001 = 22.0

    if run_number == 3:
        ierr, xarray = psspy.abrncplx(-1, 1, 1, 1, 1, 'RX')
        print('ierr for abrncplx = ', ierr)
        print('xarray for abrncplx = ', xarray)
        checkpointString = ' Ohaiyo!? ' + str(ierr) + " " + str(xarray[0]) +'\n'
        psspy.progress(checkpointString)

    for t in np.arange(t_start, t_end + t_increment, t_increment):
        print('The time is now:', round(t, 2))

        std_white_noise = 0.01 # white noise standard deviation
        t_increment_whiteNoise = 0.1

        if round(((t - t_start)/t_increment),2) % 3.0 == 0:
            # print('White noise is added at this time.')
            white_noise1 = \
            np.random.normal(0, std_white_noise, size = 1)[0]*100
            print('white_noise1 = ', round(white_noise1, 3))
            white_noise2 = \
            np.random.normal(0, std_white_noise, size = 1)[0]*100
            white_noise3 = \
            np.random.normal(0, std_white_noise, size = 1)[0]*100
        else:
            white_noise1 = 0
            white_noise2 = 0
            white_noise3 = 0

        if system_name == 'ieee9':

            ierr, current_load1 = psspy.aloadreal(1, 4, "TOTALACT")
            current_load1 = current_load1[0]
            print('current_load1 =', round(current_load1[0], 3))

            change_percent1 = load_increment_bus001/60 * t_increment \
            + white_noise1 * t_increment_whiteNoise

            ierr, totals, moto = psspy.scal_2(1, 1, 0, \
            [psspy._i, 2, 0, 1, 0], \
            [change_percent1, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0])

            # dp1 = current_load1[0]/60 * t_increment
            if current_load1[0] >= 120 and run_number == 2:
                dp1 = 0
            else:
                dp1 = current_load1[0] * change_percent1/100

            print('dp1 = ', dp1)
            ierr = psspy.increment_gref(1,'1', dp1*1.8/mbase1)
            # print('ierr for psspy.increment_gref =', ierr)

            ierr, current_load2 = psspy.aloadreal(2, 4, "TOTALACT")
            current_load2 = current_load2[0]
            print('current_load2 = ', current_load2)

            change_percent2 = load_increment_bus002/60 * t_increment \
            + white_noise2 * t_increment_whiteNoise

            ierr, totals, moto = psspy.scal_2(2, 1, 0, \
            [psspy._i, 2, 0, 1, 0], \
            [change_percent2, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0])

            # dp2 = current_load2[0]/60 * t_increment
            dp2 = current_load2[0] * change_percent2/100
            print('dp2 = ', dp2)
            psspy.increment_gref(2,'1', dp2*1.8/mbase2)

            ierr, current_load3 = psspy.aloadreal(3, 4, "TOTALACT")
            current_load3 = current_load3[0]
            change_percent3 = load_increment_bus003/60 * t_increment \
            + white_noise3 * t_increment_whiteNoise

            ierr, totals, moto = psspy.scal_2(3, 1, 0, \
            [psspy._i, 2, 0, 1, 0], \
            [change_percent3, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0])

            # dp3 = current_load3[0]/60 * t_increment
            dp3 = current_load3[0] * change_percent3/100
            print('dp3 = ', dp3)
            psspy.increment_gref(3,'1', dp3*1.8/mbase3)

            ierr, loss = psspy.aflowreal(-1, 2, 1, 1, 'PLOSS')
            losses = loss[0]
            print('Total loss = ', losses[0:8])

        elif system_name == 'ieee39':
            ierr, current_load1 = psspy.aloadreal(-1, 4, "TOTALACT")
            current_load1 = current_load1[0]
            print('current_load1 =', round(current_load1[0], 3))

            change_percent1 = load_increment_bus001/60 * t_increment \
            + white_noise1 * t_increment_whiteNoise

            ierr, totals, moto = psspy.scal_2(-1, 1, 0, \
            [psspy._i, 2, 0, 1, 0], \
            [change_percent1, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0])

        else:
            ierr, current_load1 = psspy.aloadreal(-1, 4, "TOTALACT")
            current_load1 = current_load1[0]
            print('current_load1 =', round(current_load1[0], 3))

            change_percent1 = load_increment_bus001/60 * t_increment \
            + white_noise1 * t_increment_whiteNoise

            ierr, totals, moto = psspy.scal_2(-1, 1, 0, \
            [psspy._i, 2, 0, 1, 0], \
            [change_percent1, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0])


        psspy.run(0, t, 1, 1, 0)

        checkpointString = ' Yebuseyo!? ' + str(round(t, 2)) +'\n'
        psspy.progress(checkpointString)

        with open(filenameProgress, "r") as f:
            latest_lines = tail.tail(filenameProgress, 1000)
            allowLimitBreach = 1
            exitFlag = 0

            for item in latest_lines:
                if "beyond" in item:
                    anomalies = \
                    lines_that_contain("beyond", latest_lines)
                    # print(anomalies)
                    if allowLimitBreach:
                        print('We''ll let it slide.\n')
                        exitFlag = 0
                    else:
                        print('\nThis simulation stops right here! \
                         This is because:\n')
                        exitFlag = 1

                if "Network not converged" in item:
                    print('\nThis simulation stops right here! This is because:\n')
                    anomalies = \
                    lines_that_contain('Network not converged', latest_lines)
                    # print(anomalies)
                    exitFlag = 1
                    break

            if exitFlag:
                break
            # else:
            #     print('all good! Still no divergence or limit breaches.')



    psspy.lines_per_page_one_device(2, 10000000)
    psspy.progress_output(1, "", [0, 0])


def test0_run_simulation(datapath=None, outpath=None):

    outfile, progressFile = get_demotest_file_names(outpath)

    run_simulation(datapath, savfile, snpfile, outfile, progressFile)

    print('\nDone', system_name, 'dynamics simulation')


import numpy as np

def test1_data_extraction(printCommand, show, outpath=None):
    print('Hi! Commencing test1_data_extraction.\n')
    import dyntools
    import numpy as np
    outfile, progressFile = get_demotest_file_names(outpath)
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

    if os.path.exists(filename_xlsx):
        os.remove(filename_xlsx)

    chnfobj.xlsout(channels=list(chosen_channels), show=show)

if __name__ == '__main__':

    import psse34  #noqa: F401
    simulation_inputs_folder_name = 'simulation_inputs/'
    simulation_outputs_folder_name = 'simulation_outputs/'
    run_number = 1;
    system_name = 'ieee9'
    # system_name = 'ieee39'
    printCommand = 0
    printScale = 1
    allowLimitBreach = 1 #allow machine generations to cross limits
    t_increment = 0.1

    if system_name == 'ieee9':
        chosen_channels = np.arange(1, 42)
    elif system_name == 'ieee39':
        chosen_channels = np.arange(1, 140)
    else:
        chosen_channels = np.arange(1, 140)


    savfile = simulation_inputs_folder_name \
    + system_name + '_cnv.sav'
    # savfile = system_name + '_cnv1.sav' #MBase and
    # PMax increased as per OPAL RT ieee9 bus system parameters



    # snpfile = system_name + '_snp.snp'
    # snpfile = system_name + '_snp1.snp' #D=0 for all generators
    # snpfile = system_name + '_snp2.snp' #D=0, No stabilizers
    snpfile = simulation_inputs_folder_name + \
    system_name + '_snp3.snp' #No Governors

    filename_xlsx = simulation_outputs_folder_name + \
    system_name + \
    '_outfile_' + str(run_number) + '.xlsx'
    filename_outfile = simulation_outputs_folder_name + \
    system_name + \
    '_outfile_' + str(run_number) + '.out'
    filenameProgress = simulation_outputs_folder_name + \
    system_name + \
    '_progress_dynamic_' + str(run_number) + '.txt'
    simulationType = 'Dynamic State Simulation'

    datapath = None
    show = True     # True  --> create, save
    outpath = None

    print('\nAttempting to run test0 : Simulation of', system_name)
    test0_run_simulation(datapath, outpath)
    print('\nAttempting to run test1')
    # printCommand = 0
    test1_data_extraction(printCommand, show, outpath)

    end = time.time()
    print('Time elapsed = ', end-start)
    # winsound.Beep(freq, duration)
