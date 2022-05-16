def check_simulation_health(filenameProgress, allowLimitBreach):

    def lines_that_contain(string, fp):
        return [line for line in fp if string in line]

    with open(filenameProgress, "r") as f:
        from tail import tail
        latest_lines = tail(filenameProgress, 1000)
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
                    print('\nThis simulation stops right here!\n')
                    exitFlag = 1

            if "Network not converged" in item:
                print('\nThis simulation stops right here!')
                anomalies = \
                lines_that_contain('Network not converged', latest_lines)
                # print(anomalies)
                exitFlag = 1
                break


    return exitFlag
