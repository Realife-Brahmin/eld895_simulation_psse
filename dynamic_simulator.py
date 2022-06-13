import os
import sys
import contextlib
import pssepath
pssepath.add_pssepath()

import numpy as np
# import psse34
import psspy
import excelpy

import time
import winsound
duration = 1000  # milliseconds duration of alert sound
freq = 440  # also for alert sound

start = time.time()

def run_simulation(datapath, savfile, snpfile, outfile, \
progressFile, load_increments, impedanceChanged):

    import psspy

    from apply_simulation_parameters import apply_simulation_parameters
    mbase1, mbase2, mbase3 = \
    apply_simulation_parameters(datapath, savfile, snpfile, \
    outfile, progressFile)

    psspy.strt(0, outfile)
    t_normal_run = 25
    psspy.run(0, 0.99 + t_normal_run, 1, 1, 0)

    import numpy as np
    import random as rn

    t_start = t_normal_run + 1.00
    t_end = t_normal_run + 300.00

    if system_name == 'ieee9':

        ierr = psspy.bsyso(1, 5)
        ierr = psspy.bsyso(2, 6)
        ierr = psspy.bsyso(3, 8)
        ierr = psspy.bsyso(4, 1)
        ierr = psspy.bsyso(5, 2)
        ierr = psspy.bsyso(6, 3)
        ierr = psspy.bsyso(7, 7)

        load_increment_bus001 = load_increments[0]
        load_increment_bus002 = load_increments[1]
        load_increment_bus003 = load_increments[2]

    elif system_name == 'ieee39':
        load_increment_bus001 = 22.0
    else:
        load_increment_bus001 = 22.0


    for t in np.arange(t_start, t_end + t_increment, t_increment):
        print('The time is now:', round(t, 2))



        if impedanceChanged == 0 and \
        round(t, 2) == t_start + 500*t_increment and run_number == 3:
            ierr, xarray = psspy.abrncplx(7, 2, 3, 2, 1, 'RX')
            print('ierr for abrncplx = ', ierr)
            print('xarray for abrncplx = ', xarray)
            xarray = xarray[0]
            z = xarray[0]
            r = z.real
            x = z.imag
            checkpointString = ' Ohaiyo!? ierr =  ' \
            + str(ierr) + " and impedance array is " + str(xarray) +'\n'
            psspy.progress(checkpointString)
            ierr = psspy.branch_data_3(7, 5, realar1 = 1000*r, realar3 = 1000*x)
            print('ierr for branch_data_3 = ', ierr)
            ierr, xarray = psspy.abrncplx(7, 2, 3, 2, 1, 'RX')
            print('ierr for abrncplx = ', ierr)
            print('xarray for abrncplx = ', xarray)
            xarray = xarray[0]
            z = xarray[0]
            print('New value of impedance is: ', z)
            # print("Type of retrieved z is: ", type(z))
            checkpointString = ' Arigato Gozaimasu! ierr =  ' \
            + str(ierr) + " and impedance array is " + str(xarray) +'\n'
            psspy.progress(checkpointString)
            impedanceChanged = 1


        std_white_noise = 0.01 # white noise standard deviation
        t_increment_whiteNoise = 0.1

        if round(((t - t_start)/t_increment), 2) % 3.0 == 0:
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


        change_percent1 = load_increment_bus001/60 * t_increment \
        + white_noise1 * t_increment_whiteNoise
        change_percent2 = load_increment_bus002/60 * t_increment \
        + white_noise2 * t_increment_whiteNoise
        change_percent3 = load_increment_bus003/60 * t_increment \
        + white_noise3 * t_increment_whiteNoise

        ierr, current_load1 = psspy.aloadreal(1, 4, "TOTALACT")
        current_load1 = current_load1[0]
        print('current_load1 =', round(current_load1[0], 3))

        ierr, current_load2 = psspy.aloadreal(2, 4, "TOTALACT")
        current_load2 = current_load2[0]

        ierr, current_load3 = psspy.aloadreal(3, 4, "TOTALACT")
        current_load3 = current_load3[0]

        ierr, totals, moto = psspy.scal_2(1, 1, 0, \
        [psspy._i, 2, 0, 1, 0], \
        [change_percent1, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0])

        ierr, totals, moto = psspy.scal_2(2, 1, 0, \
        [psspy._i, 2, 0, 1, 0],
        [change_percent2, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0])

        dp2 = current_load2[0] * change_percent2/100

        ierr, totals, moto = psspy.scal_2(3, 1, 0, \
        [psspy._i, 2, 0, 1, 0], \
        [change_percent3, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0])

        dp3 = current_load3[0] * change_percent3/100

        if current_load1[0] >= 120 and run_number == 2:
            dp1 = 0
        else:
            dp1 = current_load1[0] * change_percent1/100


        print('dp1 = ', dp1)
        print('dp2 = ', dp2)
        print('dp3 = ', dp3)
        dpTotal = dp1 + dp2 + dp3

        if run_number == 6:
            psspy.increment_gref(1, '1', dpTotal*1.8/mbase1)
            psspy.increment_gref(2, '1', dp2*0.0/mbase2)
            psspy.increment_gref(3, '1', dp3*0.0/mbase3)
        elif run_number == 7:
            psspy.increment_gref(1, '1', dp1*0.0/mbase1)
            psspy.increment_gref(2, '1', dpTotal*1.8/mbase2)
            psspy.increment_gref(3, '1', dp3*0.0/mbase3)
        elif run_number == 8:
            psspy.increment_gref(1, '1', dp1*0.0/mbase1)
            psspy.increment_gref(2, '1', dp2*0.0/mbase1)
            psspy.increment_gref(1, '1', dpTotal*1.8/mbase3)
        else:
            psspy.increment_gref(1, '1', dp1*1.8/mbase1)
            psspy.increment_gref(2, '1', dp2*1.8/mbase2)
            psspy.increment_gref(3, '1', dp3*1.8/mbase3)

        ierr, loss = psspy.aflowreal(-1, 2, 1, 1, 'PLOSS')
        losses = loss[0]
        stringLoss = "Current Losses in all branches are: " + str(losses)
        psspy.progress(stringLoss)
        print('Total loss = ', losses[0:8])

        psspy.run(0, t, 1, 1, 0)

        checkpointString = ' Yebuseyo!? ' + str(round(t, 2)) + '\n'
        psspy.progress(checkpointString)

        from check_simulation_health import check_simulation_health
        exitFlag = check_simulation_health(filenameProgress, allowLimitBreach)
        if exitFlag:
            break

        trip = 0
        ierr = psspy.set_osscan(1, trip)

    return impedanceChanged
    psspy.lines_per_page_one_device(2, 10000000)
    psspy.progress_output(1, "", [0, 0])


def test0_run_simulation(load_increments, impedanceChanged, \
datapath=None, outpath=None):

    from get_demotest_file_names import get_demotest_file_names
    outfile, progressFile = \
    get_demotest_file_names(outpath, filename_outfile, filenameProgress)

    impedanceChanged = run_simulation(datapath, savfile, snpfile, outfile, \
    progressFile, load_increments, impedanceChanged)

    print('\nDone', system_name, 'dynamics simulation')

if __name__ == '__main__':

    import psse34  #noqa: F401
    psse_version = 'xplore'
    # psse_version = 'full'
    simulation_inputs_folder_name = 'simulation_inputs/'
    simulation_outputs_folder_name = 'simulation_outputs/'
    run_number = 8;
    impedanceChanged = 0;
    system_name = 'ieee9'
    # system_name = 'ieee39'
    printCommand = 0
    printScale = 1
    allowLimitBreach = 1 #allow machine generations to cross limits
    t_increment = 0.1
    import numpy as np

    if run_number == 4:
        load_increments = np.array([7, 9, 11], dtype = float)
    elif run_number == 5:
        load_increments = np.array([33, 35, 27], dtype = float)
    else:
        load_increments = np.array([27, 29, 21], dtype = float)


    if system_name == 'ieee9':
        if psse_version == 'xplore':
            snpfile_suffix = '_snp3_xplore.snp'
            chosen_channels = np.arange(1, 42)
        else:
            snpfile_suffix = '_snp3_full.snp'
            chosen_channels = np.arange(1, 114)

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
    system_name + snpfile_suffix #No Governors

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
    test0_run_simulation(load_increments, impedanceChanged, \
     datapath, outpath)

    print('\nAttempting to run test1')
    from test1_data_extraction import test1_data_extraction
    test1_data_extraction(show, filename_outfile,\
     filenameProgress, filename_xlsx, \
      printScale, printCommand, chosen_channels, outpath)

    end = time.time()
    print('Time elapsed = ', end-start)
    # winsound.Beep(freq, duration)
