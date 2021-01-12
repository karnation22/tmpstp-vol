import os 


def main():
    for item in os.listdir():
        cwd = os.getcwd()
        if("_misc_data" in item):
            os.chdir(item)
            for subitem in os.listdir():
                if("textDump" in subitem):
                    with open(subitem, "r") as f_subitm:
                        lines = f_subitm.readlines()
                        relLine = lines[0].strip()
                    writeFile = subitem[:subitem.index("textDump")]+"textString.txt"
                    with open(writeFile, "w") as f_itm:
                        f_itm.write(relLine)
                    ## print("mv {} {}".format(writeFile,cwd))
                    os.system("mv {} '{}'".format(writeFile,cwd))
                    break
            break        

main()