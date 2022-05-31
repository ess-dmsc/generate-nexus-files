import fileinput
import time


def time_reading_file(read_function):
    # time at the start of program is noted
    start = time.time()

    # keeps a track of number of lines in the file
    count = read_function()

    # time at the end of program execution is noted
    end = time.time()

    # total time taken to print the file
    print("Execution time in seconds: ", (end - start))
    print("No. of lines printed: ", count)


if __name__ == '__main__':
    # time the read function using fileinput.
    file_name = 'DREAM_Mantel.off'


    def read_func_1():
        count = 0
        for lines in fileinput.input([file_name]):
            count = count + 1
        return count


    time_reading_file(read_func_1)


    def read_func_2():
        count = 0
        with open(file_name, 'r') as f:
            for line in f:
                count = count + 1
        return count


    time_reading_file(read_func_2)
