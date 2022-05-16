def apply_simulation_parameters(datapath, savfile, snpfile, \
  outfile, progressFile):
      import psspy
      psspy.psseinit(50000)

      import os
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

      ierr = psspy.rstr(snpfile)
      if ierr:
          psspy.progress_output(1, "", [0, 0])
          print(" psspy.rstr Error")
          return

      psspy.lines_per_page_one_device(1, 10000000)

      psspy.dynamics_solution_param_2(realar3=0.01)

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

      return mbase1, mbase2, mbase3
