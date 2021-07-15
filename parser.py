import pandas
import numpy


NUMPY_STD_DDOF = 0

PLATE_SIZE = (0, 0)

FILE_FORMAT_AUTO = 'auto'

FILE_FORMAT_EML = 'eml'
FILE_FORMAT_CSV = 'csv'

FILE_FORMAT_IN = 'in'
FILE_IN_COMMENT = '#'
FILE_IN_VARS = 'VARS'
FILE_IN_SEPARATOR = '\t'
FILE_IN_END = '@'
FILE_IN_DATA_FILE = 'DATA_FILE'
FILE_IN_HAS_HEADER = 'HAS_HEADER'
FILE_IN_TABLE_INDEX = 'TABLE_INDEX'
FILE_IN_X = 'X'
FILE_IN_Y = 'Y'
FILE_IN_LABEL = 'LABEL'
FILE_IN_WELL = 'WELL'
FILE_IN_WELL_COL_VAL = 'WELL_COL_VAL'
FILE_IN_WELL_ROW_VAL = 'WELL_ROW_VAL'
FILE_IN_ROW = 'ROW'
FILE_IN_ROW_VAL = 'ROW_VAL'
FILE_IN_COL_VAL = 'COL_VAL'
FILE_IN_COL = 'COL'
FILE_IN_IGNORE_COL = 'IGNORE_COL'
FILE_IN_IGNORE_ROW = 'IGNORE_ROW'
FILE_IN_IGNORE_WELL = 'IGNORE_WELL'
FILE_IN_DONT_FIT_COL_VAL = 'DONT_FIT_COL_VAL'
FILE_IN_DONT_FIT_ROW_VAL = 'DONT_FIT_ROW_VAL'
FILE_IN_DONT_FIT_DATA_POINT = 'DONT_FIT_DATA_POINT'
FILE_IN_DATA_POINT_SEP = ';'
FILE_IN_YES = 'yes'
FILE_IN_NO = 'no'

FILE_EML_TYPE = ''
FILE_CSV_PLATE_INFORMATION = 'Plate information'
FILE_CSV_SEPARATOR = '\t'
FILE_TABLE_INDEX = 0


PLATE_ALL_ROWS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
PLATE_ALL_COLS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,26, 27, 28]


ROW_TO_INDEX = dict()
for col, index in zip(PLATE_ALL_ROWS, range(len(PLATE_ALL_ROWS))):
    ROW_TO_INDEX[col] = index


def read_eml(file_name):
    file = open(file_name)

    all_str = file.read().replace('\n', '')
    print(all_str)


def read_csv(file_name, header, table_index=FILE_TABLE_INDEX):

    # list which contains all the the tables found in the .csv file
    # "table" refers to the section in the file introduced by FILE_CSV_PLATE_INFORMATION
    # and where all rows and columns are numbers

    plate_array = []

    print('read_csv: Reading csv file...')
    # read file as string
    file = open(file_name, 'r')
    all_str = file.read()
    file.close()

    # get all lines
    lines = all_str.split('\n')

    # boolean that indicates if the section is a table
    read_plate = False

    # iterate through lines
    for line in lines:
        line = line.replace('  ', ' ')

        # watch out for "Plate information"
        # if there is a plate inforamtion entry, create new list in plate_array and go to next line
        if FILE_CSV_PLATE_INFORMATION in line:
            read_plate = True
            plate_array.append([])
            continue

        # if a table was identified, try to parse ALL values to float numbers
        # if everything is parsable, add values to a row and then, add this row to the table list
        first_char = line.split(FILE_CSV_SEPARATOR)[0]
        
        if read_plate:

            try:
                row = None

                if header and first_char in PLATE_ALL_ROWS:
                    row = [float(x) for x in line.split(FILE_CSV_SEPARATOR)[1:] if x]

                elif not header:
                    row = [float(x) for x in line.split(FILE_CSV_SEPARATOR) if x]

                if row:
                    plate_array[-1].append(row)

            except ValueError:
                # Ignore line if not all values could be parsed to float values
                
                pass

    # notify the user that the the file contains more than one table
    if len(plate_array) >= 2:
        print('read_csv: Warning: Parser found two tables in {}! Took table {} (table_index = {})!'.format(file_name, table_index, table_index))

    # convert the messy plate_array to an dictionary
    plate_dict = dict()

    for col in range(len(plate_array[table_index][0])):

        c = PLATE_ALL_COLS[col]
        plate_dict[c] = []

        for row, vals in zip(range(len(plate_array[table_index])), plate_array[table_index]):
            r = PLATE_ALL_ROWS[row]
            plate_dict[c].append(vals[col])

    # finally create a pandas DataFrame which is more usefull in this case
    plate_df = pandas.DataFrame(plate_dict)

    print('read_csv: File read successfully!\n')
    return plate_df


def read_in(file_name):

    print('read_in: Reading input file...')
    # read file as string
    file = open(file_name, 'r')
    all_str = file.read()
    file.close()

    # get all lines
    lines = all_str.split('\n')

    # iterate through lines
    output_dict = {}
    section_dict = {}

    in_vars = False
    read_keys = False

    for line in lines:
        if not line:
            continue

        # remove comments in line
        if FILE_IN_COMMENT in line:
            line = line.split(FILE_IN_COMMENT)[0]

        tokens = [t for t in line.split(FILE_IN_SEPARATOR)]

        if not len(tokens):
            continue

        # if line starts with VAR token, read column names

        section_end = FILE_IN_END + FILE_IN_VARS

        if read_keys:
            for token in tokens:
                 section_dict[token] = []
            read_keys = False
            continue

        if tokens[0] == FILE_IN_VARS:
            in_vars = True
            read_keys = True
        elif tokens[0] == section_end:

            output_dict.update(section_dict)

            section_dict = {}
            in_vars = False
            read_keys = False
        else:
            read_keys = False

        if in_vars:

            for key, token in zip(section_dict.keys(), tokens):
                if not token:
                    continue
                # try:
                #     section_dict[key].append(float(token))
                # except Exception:
                #     section_dict[key].append(token)
                section_dict[key].append(token)

    print('read_in: File read sucessfully!\n')
    return output_dict


FILE_READER_DICT = {FILE_FORMAT_EML: read_eml,
                    FILE_FORMAT_CSV: read_csv,
                    FILE_FORMAT_IN: read_in}


class Plate():

    def __init__(self, input_file_name, file_format=FILE_FORMAT_AUTO):
        self._plate_df = None
        self._data_dict = None
        self._fit_data_dict = None
        self._file_name = None

        # read plate from file and save it in the _plate_df
        self.parse_to_data(input_file_name, file_format)

    def parse_to_data(self, input_file_name, data_file_format=FILE_FORMAT_AUTO):

        print('Reading input file!')
        # read annotation file; always .in format
        suffix = input_file_name.split('.')[-1]

        if suffix != FILE_FORMAT_IN:
            # ERROR
            print('Error: Input file has wrong suffix! Suffix should be {}'.format(FILE_FORMAT_IN))
            exit(1)

        input_dict = FILE_READER_DICT[suffix](input_file_name)

        print('Variables from {} file:'.format(FILE_FORMAT_IN))
        for key in input_dict.keys():
            print('{}:\t{}'.format(key, input_dict[key]))

        print('Input file sucessfully read!')
        print('-----------------------------\n')

        print('Reading data file!')

        # get data file name and data file parameters given in the input file
        data_file_name = input_dict[FILE_IN_DATA_FILE][0]
        has_header = input_dict[FILE_IN_HAS_HEADER][0] == FILE_IN_YES
        table_index = 0
        try:
            table_index = int(input_dict[FILE_IN_TABLE_INDEX][0])
        except ValueError:
            print('Error: {} from input file has to be an integer! {} is no integer!'.format(FILE_IN_TABLE_INDEX, input_dict[FILE_IN_TABLE_INDEX][0]))
            exit(1)

        self._file_name = data_file_name

        # create a data frame
        file_df = None

        # read data file
        if data_file_format == FILE_FORMAT_AUTO:
            suffix = data_file_name.split('.')[-1]
            try:
                file_df = FILE_READER_DICT[suffix](data_file_name, has_header, table_index)
            except IndexError:
                print('Error: .{} file is not readable!'.format(suffix))
                exit(1)
        else:
            try:
                file_df = FILE_READER_DICT[data_file_format](data_file_name)
            except IndexError:
                print('Error: .{} file is not readable!'.format(data_file_name))
                exit(1)

        self._plate_df = file_df

        # output data frame
        print('Data frame:')
        print(self._plate_df)
        print('Data file successfully read!')
        print('-----------------------------\n')

        # extract data from plate like described in the input file
        data_dict = {}
        # c_val = concentration, row_val Compound
        # data_dict = {concentration : {cmpd1:[[value1,..value4], [value1,...,value4]]}}
        # initiate data_dict
        for s in range(1, 10):
            col_val_key = FILE_IN_COL_VAL + str(s)
            col_key = FILE_IN_COL + str(s)
            if not col_key in input_dict.keys() or not col_val_key in input_dict.keys():
                continue

            for c_val, c in zip(input_dict[col_val_key], input_dict[col_key]):
                if c_val not in data_dict.keys():
                    data_dict[c_val] = dict()

                row_val_key = FILE_IN_ROW_VAL + str(s)
                row_key = FILE_IN_ROW + str(s)
                if not row_key in input_dict.keys() or not row_val_key in input_dict.keys():
                    continue

                for r_val, r in zip(input_dict[row_val_key], input_dict[row_key]):
                    if r_val not in data_dict[c_val].keys():
                        data_dict[c_val][r_val] = []

                    # check if well is set as 'ignore' in the annotation file, e.g. if wrong pipetting or so :(
                    if c in input_dict[FILE_IN_IGNORE_COL]:
                        print('Note: Column {} was not added into data set!'.format(str(c)))
                        continue
                    if r in input_dict[FILE_IN_IGNORE_ROW]:
                        print('Note: Row {} was not added into data set!'.format(str(r)))
                        continue
                    if '{}{}'.format(r, c) in input_dict[FILE_IN_IGNORE_WELL]:
                        print('Note: Well {} was not added into data set!'.format(str(r) + str(c)))
                        continue

                    data_dict[c_val][r_val].append(self.read_well(r, c))

        data_label = input_dict[FILE_IN_LABEL][0]

        # dictionary with all x, y_mean and y_std data
        mean_data_dict = dict()
        # mean_data_dict = {Cmpd : [mean_x_conc1,...], [mean_z_conc1,mean_x_conc2,...], [std x_conc1,...]}
        # dictionary with x, y_mean and y_std data for fitting

        mean_fit_data_dict = dict()
        # mean_fitdata_dict = {Cmpd : [mean_x_conc1,...], [mean_z_conc1,mean_x_conc2,...], [std x_conc1,...]}

        for c_val in data_dict.keys():

            for r_val in data_dict[c_val].keys():

                x = None

                if data_label == FILE_IN_COL:
                    key = c_val
                    x = r_val

                elif data_label == FILE_IN_ROW:
                    key = r_val
                    x = c_val

                else:
                    # ERROR!
                    print('Error: only ROW or COL can be assigned to LABEL!')
                    exit(1)

                # create data set in dictionary
                if key not in mean_data_dict.keys():
                    # data set content:    x y_mean y_std
                    mean_data_dict[key] = [[], [], []]
                if key not in mean_fit_data_dict.keys():
                    mean_fit_data_dict[key] = [[], [], []]

                # check if there is any data for each ROW and COL
                if not data_dict[c_val][r_val]:
                    continue

                y = numpy.mean(data_dict[c_val][r_val])
                y_std = numpy.std(data_dict[c_val][r_val], ddof=NUMPY_STD_DDOF)

                mean_data_dict[key][0].append(x)
                mean_data_dict[key][1].append(y)
                mean_data_dict[key][2].append(y_std)

                # check if value is used for fitting
                dp = '{}{}{}'.format(c_val, FILE_IN_DATA_POINT_SEP, r_val)

                if dp in input_dict[FILE_IN_DONT_FIT_DATA_POINT]:
                    print('Note: Data point ({}) is not used for fitting!'.format(dp))
                    continue
                if c_val in input_dict[FILE_IN_DONT_FIT_COL_VAL]:
                    print('Note: All values in ({}) are not used for fitting!'.format(c_val))
                    continue
                if r_val in input_dict[FILE_IN_DONT_FIT_ROW_VAL]:
                    print('Note: All values in ({}) are not used for fitting!'.format(r_val))
                    continue

                mean_fit_data_dict[key][0].append(x)
                mean_fit_data_dict[key][1].append(y)
                mean_fit_data_dict[key][2].append(y_std)

        print('Data sets (all data points):')
        for key in mean_data_dict.keys():
            print('{}:'.format(key))
            for i, j in zip(['x', 'y_mean', 'y_std'], mean_data_dict[key]):
                print('\t{}:\t{}'.format(i, j))

        print('Data sets successfully extracted from plate!')
        print('--------------------------------------------\n')

        self._data_dict = mean_data_dict
        self._fit_data_dict = mean_fit_data_dict

    def read_well(self, row, col):
        """
        Method to read a well at a certain position
        :param row: a string: A, B, C, ...
        :param col: an integer: 1, 2, 3, 4, 5,
        :return:
        """
        try:
            col = int(col)
        except ValueError:
            print('Error! COLS have to be integers!')
            exit(1)

        w = self._plate_df.iloc[ROW_TO_INDEX[row]][col]
        return w

    def labels(self):
        return list(self._data_dict.keys())

    def x(self, label):
        try:
            return numpy.array([float(i) for i in self._data_dict[label][0]])
        except ValueError:
            print('Error! Data set {} has to contain numerical values!'.format(label))
            exit(1)

    def y(self, label):
        try:
            return numpy.array([float(i) for i in self._data_dict[label][1]])
        except ValueError:
            print('Error! Data set {} has to contain numerical values!'.format(label))
            exit(1)

    def y_std(self, label):
        try:
            return numpy.array([float(i) for i in self._data_dict[label][2]])
        except ValueError:
            print('Error! Data set {} has to contain numerical values!'.format(label))
            exit(1)

    def fit_x(self, label):
        try:
            return numpy.array([float(i) for i in self._fit_data_dict[label][0]])
        except ValueError:
            print('Error! Data set {} has to contain numerical values!'.format(label))
            exit(1)

    def fit_y(self, label):
        try:
            return numpy.array([float(i) for i in self._fit_data_dict[label][1]])
        except ValueError:
            print('Error! Data set {} has to contain numerical values!'.format(label))
            exit(1)

    def fit_y_std(self, label):
        try:
            return numpy.array([float(i) for i in self._fit_data_dict[label][2]])
        except ValueError:
            print('Error! Data set {} has to contain numerical values!'.format(label))
            exit(1)

    @property
    def file_name(self):
        return self._file_name
