import parser
import plotter
import sys

if __name__ == '__main__':

    # get input file from user input
    try:
        in_file = sys.argv[1]
    except IndexError:
        print('Error: No input file given! Enter the .in file as first argument!')
        sys.exit(1)

    # create plate
    plate = parser.Plate(in_file)
    # create plotter
    plotter = plotter.Plotter(plate, in_file)

    # visualize data
    plotter.plot()