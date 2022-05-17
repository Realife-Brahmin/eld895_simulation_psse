import os
import sys
import contextlib
import pssepath
pssepath.add_pssepath()
import numpy as np
import psspy
import excelpy

import time
import winsound
duration = 1000  # milliseconds duration of alert sound
freq = 440  # also for alert sound

start = time.time()

def run_simulation(datapath, savfile, snpfile, outfile, \
progressFile, load_increments):

    import psspy

    from apply_simulation_parameters import apply_simulation_parameters
    mbase1, mbase2, mbase3 = \
    apply_simulation_parameters(datapath, savfile, snpfile, \
    outfile, progressFile)

    psspy.strt(0, outfile)
    psspy.run(0, 0.99, 1, 1, 0)

    import numpy as np
    import random as rn

    t_start = 1.00
    t_end = 600.00

    if system_name == 'ieee9':

        load_increment_bus001 = load_increments[0]
        load_increment_bus002 = load_increments[1]
        load_increment_bus003 = load_increments[2]

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
            print(change_percent1)

            ierr, totals, moto = psspy.scal_2(1, 1, 0, \
            [psspy._i, 2, 0, 1, 0], \
            [change_percent1, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0])

            if current_load1[0] >= 120 and run_number == 2:
                dp1 = 0
            else:
                dp1 = current_load1[0] * change_percent1/100

            print('dp1 = ', dp1)
            ierr = psspy.increment_gref(1,'1', dp1*1.8/mbase1)

            ierr, current_load2 = psspy.aloadreal(2, 4, "TOTALACT")
            current_load2 = current_load2[0]
            print('current_load2 = ', current_load2)

            change_percent2 = load_increment_bus002/60 * t_increment \
            + white_noise2 * t_increment_whiteNoise

            ierr, totals, moto = psspy.scal_2(2, 1, 0, \
            [psspy._i, 2, 0, 1, 0], \
            [change_percent2, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0])

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

        checkpointString = ' Yebuseyo!? ' + str(round(t, 2)) + '\n'
        psspy.progress(checkpointString)

        from check_simulation_health import check_simulation_health
        exitFlag = check_simulation_health(filenameProgress, allowLimitBreach)
        if exitFlag:
            break

    psspy.lines_per_page_one_device(2, 10000000)
    psspy.progress_output(1, "", [0, 0])


def test0_run_simulation(load_increments, datapath=None, outpath=None):

    from get_demotest_file_names import get_demotest_file_names
    outfile, progressFile = \
    get_demotest_file_names(outpath, filename_outfile, filenameProgress)

    run_simulation(datapath, savfile, snpfile, outfile, \
    progressFile, load_increments)

    print('\nDone', system_name, 'dynamics simulation')

if __name__ == '__main__':

    import psse34  #noqa: F401
    simulation_inputs_folder_name = 'simulation_inputs/'
    simulation_outputs_folder_name = 'simulation_outputs/'
    run_number = 2;
    system_name = 'ieee9'
    # system_name = 'ieee39'
    printCommand = 0
    printScale = 1
    allowLimitBreach = 1 #allow machine generations to cross limits
    t_increment = 0.1
    import numpy as np
    load_increments = np.array([27, 29, 21], dtype = float)

    if system_name == 'ieee9':
        chosen_channels = np.arange(1, 43)
    elif system_name == 'ieee39':
        chosen_channels = np.arange(1, 140)
    else:
        chosen_channels = np.arange(1, 140)


    savfile = simulation_inputs_folder_name \
    + system_name + '_cnv.sav'

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
    test0_run_simulation(load_increments, datapath, outpath)

    print('\nAttempting to run test1')
    from test1_data_extraction import test1_data_extraction
    test1_data_extraction(show, filename_outfile,\
     filenameProgress, filename_xlsx, \
      printScale, printCommand, chosen_channels, outpath)

    end = time.time()
    print('Time elapsed = ', end-start)
    # winsound.Beep(freq, duration)
